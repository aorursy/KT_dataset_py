# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import StratifiedKFold,GridSearchCV
seed =123
% matplotlib inline
# Read data sets, set missings, show dimensions
path = '../input/'
train = pd.read_csv(path+'train.csv',na_values=-1)
test = pd.read_csv(path+'test.csv',na_values=-1)
print('Number of rows and columns:',train.shape)
print('Number of rows and columns:',test.shape)
# Let's inspect the first data lines visually
train.head(9).T
# Explore the target variable
plt.figure(figsize=(10,3))
sns.countplot(train['target'],palette='rainbow')
plt.xlabel('Target')
train['target'].value_counts()
# Plot correlations
cor = train.corr()
plt.figure(figsize=(12,9))
sns.heatmap(cor,cmap='RdBu')
# Drop ps_calc* variables (see above)
ps_cal = train.columns[train.columns.str.startswith('ps_calc')] 
train = train.drop(ps_cal,axis =1)
test = test.drop(ps_cal,axis=1)
train.shape
# Plot missing values of train and test data 
k= pd.DataFrame()
k['train']= train.isnull().sum()
k['test'] = test.isnull().sum()
fig,ax = plt.subplots(figsize=(16,5))
k.plot(kind='bar',ax=ax)
# Replace missing with mode
def missing_value(df):
    col = df.columns
    for i in col:
        if df[i].isnull().sum()>0:
            df[i].fillna(df[i].mode()[0],inplace=True)

missing_value(train)
missing_value(test)

# Count categories 
def basic_details(df):
    b = pd.DataFrame()
    #b['Missing value'] = df.isnull().sum()
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
basic_details(train)
# change data type to category (if not more than 104 categories, see ps_car_11_cat*egory)
def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=104:
            df[i] = df[i].astype('category')
category_type(train)
category_type(test)
#basic_details(train)
# generate some variables lists
cat_col = [col for col in train.columns if '_cat' in col]
#print(cat_col)
bin_col = [col for col in train.columns if 'bin' in col]
#print(bin_col)
tot_cat_col = list(train.select_dtypes(include=['category']).columns)
#print(tot_cat_col)
other_cat_col = [c for c in tot_cat_col if c not in cat_col+ bin_col]
other_cat_col
# Calculate Median and mean for categorical data
def transform_df(df):
    df = pd.DataFrame(df)
    dcol= [c for c in train.columns if train[c].nunique()>2]
    dcol.remove('id')   
    d_median = df[dcol].median(axis=0)
    d_mean = df[dcol].mean(axis=0)
    q1 = df[dcol].apply(np.float32).quantile(0.25)
    q2 = df[dcol].apply(np.float32).quantile(0.5)
    q3 = df[dcol].apply(np.float32).quantile(0.75)
    
    #Add mean and median column to data set having more then 2 categories
    for c in dcol:
        df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)
        df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)
        df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)
        df[c+str('_q2')] = (df[c].astype(np.float32).values < q2[c]).astype(np.int8)
        df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)
    return df

train = transform_df(train)
test = transform_df(test)
basic_details(train)
# Correlation plot of the numeric or "many level" features (for ideas about their meaning have a look in discussions)
num_col = ['ps_reg_03','ps_car_12','ps_car_13','ps_car_14']
cor = train[num_col].corr()
plt.figure(figsize=(10,4))
sns.heatmap(cor,annot=True)
plt.tight_layout()
# Determine outliers in dataset
def outlier(df,columns):
    for i in columns:
        quartile_1,quartile_3 = np.percentile(df[i],[25,75])
        quartile_f,quartile_l = np.percentile(df[i],[1,99])
        IQR = quartile_3-quartile_1
        lower_bound = quartile_1 - (1.5*IQR)
        upper_bound = quartile_3 + (1.5*IQR)
        print(i,lower_bound,upper_bound,quartile_f,quartile_l)
                
        df[i].loc[df[i] < lower_bound] = quartile_f
        df[i].loc[df[i] > upper_bound] = quartile_l
        
outlier(train,num_col)
outlier(test,num_col) 
# One Hot Encoding (generate dummy variables for categories)
def OHE(df1,df2,column):
    cat_col = column
    #cat_col = df.select_dtypes(include =['category']).columns
    len_df1 = df1.shape[0]
    
    df = pd.concat([df1,df2],ignore_index=True)
    c2,c3 = [],{}
    
    print('Categorical feature',len(column))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2

train1,test1 = OHE(train,test,tot_cat_col)
# Split data set
X = train1.drop(['target','id'],axis=1)
y = train1['target'].astype('category')
x_test = test1.drop(['target','id'],axis=1)
del train1,test1
# Logistic regression model (L2-penalty), 5-fold CV, C determined by grid-search
kf = StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
pred_test_full=0
cv_score=[]
i=1
for train_index,test_index in kf.split(X,y):    
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    lr = LogisticRegression(class_weight='balanced',C=0.003)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:,1]
    score = roc_auc_score(yvl,pred_test)
    print('roc_auc_score',score)
    cv_score.append(score)
    pred_test_full += lr.predict_proba(x_test)[:,1]
    i+=1
# Model performance
print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))
print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))
# Area Under ROC-Curve (Normalized gini = 2*AUC-1)
proba = lr.predict_proba(xvl)[:,1]
fpr,tpr, threshold = roc_curve(yvl,proba)
auc_val = auc(fpr,tpr)

plt.figure(figsize=(14,8))
plt.title('Area Under ROC-Curve')
plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
# Make prediction and create submission file
y_pred = pred_test_full/5
submit = pd.DataFrame({'id':test['id'],'target':y_pred})
submit.to_csv('lr_porto.csv',index=False) 
#conda install jupyter
#conda install scipy
#pip install sklearn
#pip install pandas
#pip install pandas-datareader
#pip install matplotlib
#pip install pillow
#pip install requests
#pip install h5py
#pip install tensorflow==1.4.0
#pip install keras==2.1.2
#conda install –c rdonnelly py-xgboost
#conda install –c conda-forge lightgbm