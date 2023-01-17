import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn import svm
train_set = pd.read_csv('../input/train.csv')
train_set.columns
#initial Data exploration
train_set.shape
train_set.isnull().sum()
train_set.describe()
train_set.describe(include=['O'])
train_set['touch_screen'].value_counts()
train_set['bluetooth'].value_counts()
train_set['dual_sim'].value_counts()
train_set['wifi'].value_counts()
train_set['4g'].value_counts()
train_set['3g'].value_counts()
train_set['price_range'].value_counts()
#Range Flag creation
for price in train_set['price_range'].unique():
    train_set[price]=0
    train_set[price][train_set['price_range']==price]=1
# train_set.to_csv('train.csv',index=False)
train_set.head()
def information_value(variable,dfm,n=10):
    """
    function to calculate the information value for a given target variable
    parameters
    -----------
        variable = the target variable (string)
        n = number of total features to be returned, default=10 (int)
        dfm = the dataframe of all the columns for which information value is to be calculated (Dataframe)
    returns
    --------
        required_features = list of top n features in case of information value (List)
    
    """
    variables=list(dfm.columns)
    iv=dict((el,0) for el in variables)
    variables.remove(variable)
    for i in variables:
        if dfm[i].dtypes!='O':
                temp=dfm.groupby(pd.qcut(dfm[i],10,duplicates='drop'))[variable].value_counts().reset_index(name='counts')
        else:
            temp=dfm.groupby(dfm[i])[variable].value_counts().reset_index(name='counts')
        newdf=pd.pivot_table(temp, values = 'counts', index=[i], columns =variable)
        newdf['ACCTS']=newdf[0]+newdf[1]
        newdf['GOODS']=newdf[0]/(newdf[0].sum())
        newdf['BADS']=newdf[1]/(newdf[1].sum())
        newdf['TOTAL']=newdf['ACCTS']/(newdf['ACCTS'].sum())    
        newdf['ODDS']=newdf['GOODS']/newdf['BADS']
        newdf['LN_ODDS']=np.log(newdf['ODDS'])
        newdf['VALUE']=(newdf['GOODS']-newdf['BADS'])*newdf['LN_ODDS']
        iv[i]=float(newdf['VALUE'].sum())
    iv = sorted(iv.items(), key=operator.itemgetter(1),reverse=True)
    required_features=[s[0] for s in iv[0:n]]
    return required_features
    
#getting the union of top n features for all the price_range, change the n accordingly to get good model.
variables=train_set['price_range'].unique()
total_features=[]
for f in variables:
    var=[v for v in variables if v!=f]
    features=list(set(train_set.columns)-set(var)-set(['Id','price_range']))
    dfnw=train_set[features]
    required_features=information_value(f,dfnw,12)
    total_features.extend(required_features)
total_features= list(set(total_features))
print (total_features)
#Lets convert these categorical values into numeric
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    train_set[i] = train_set[i].replace({'yes':1,'no':0})
train_set['price_range'] = train_set['price_range'].replace({'very low':0,'low':1,'medium':2,'high':3})
X = train_set[total_features]
Y = train_set['price_range']
#splitting test and train set
trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3)
def svm_gridsearch(X, Y, nfolds):
    """
    function to find the optimum value of gamma and c in svm
    Parameters
    -----------
        X = The features of model (Dataframe)
        Y = The target variable
        nfolds = number of cross validation required
    Returns
    ---------
        gamma and C values for the best model (Dictionary)
    """
    Cs = [0.0001,0.001,0.005,0.006,0.008, 0.01,0.05,0.08, 0.1]
    gammas = [0.0001,0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, Y)
    grid_search.best_params_
    return grid_search.best_params_
params=svm_gridsearch(trainX,trainY,5)
print (params)
#final model with best accuracy
model = svm.SVC(kernel='linear', decision_function_shape='ovr',gamma=params['gamma'],C=params['C']) 
model.fit(trainX, trainY)
model.score(testX, testY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
print (classification_report(testY,preds))
test_set = pd.read_csv('../input/test.csv')
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    test_set[i] = test_set[i].replace({'yes':1,'no':0})
test_set['price_range'] = model.predict(test_set[total_features])
test_set['price_range'] = test_set['price_range'].replace({0:'very low',1:'low',2:'medium',3:'high'})
test_set[['Id','price_range']].to_csv('submission98.9.csv',index=False)
