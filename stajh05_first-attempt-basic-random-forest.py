#Purpose: To predict the levels of poverty on the dataset. 
#Background: This is my first attempt and one of my early machine learning projects with python individually(I have used python professionally 6 or 7 times and took a 
#            couple classes on it.I will focus on writing decent code with a good model. My next submission will be more detailed as I will read more about feature selection
#            /evaluation with python or I will use R which is my strongest language. Feedback is welcome.       
#Process: Load libraries and requried methods. Then clean the data set by removing varaibles that are non-numeric, missing too much data or are indexes. Then we
#         will test for which variable are the most "important" through feature ranking. Then split the training set data into test and train. The next step is to 
#         create a model, check for overfitting, and submit. Easy enough :)
#Notes: I did some analysis with some python functions and outside of python to make some of my vairable decisions. I also wanted to see where I was at and did not 
#       do any additional feature  engineering yet.  
#Load the required libraries and functions for this analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.max_columns = None

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Load the data, clean and impute. I did analysis earlier to determine which variables I wanted to 
# exclude due to there being too much missing data or the type being string. Later iterations will 
# explore other strategies for incorporating these variables in the analysis 
data_set= pd.read_csv('../input/train.csv')
variables_missingOrstr= ['v2a1','v18q1','rez_esc','elimbasu5', 'Id', 'idhogar','dependency','edjefe','edjefa']
data_set=data_set.drop(columns=variables_missingOrstr)
imput=SimpleImputer(strategy='mean')
imput.fit(data_set)
data_set_np=imput.transform(data_set)
names=data_set.columns
data_set= pd.DataFrame(data_set_np)
data_set.columns=names
def rank_features(data_set, target='Target',numVars=35):
    '''
    This function ranks the variables and returns a list of important features. User decides the number to return. The default is 35. 
    data_set= The data set to be predicted (pandas.Dataframe object)
    target= The variable that is to be targeted in the analysis (str type)
    numVars= The number of variables to be returned (numeric formulas to get numeric outputs welcome)
    '''
    random_forest_rank= RandomForestClassifier(n_estimators=100)
    random_forest_rank.fit(data_set.drop(columns='Target'),data_set.loc[:,'Target'])
    importances= np.array(random_forest_rank.feature_importances_)
    features= np.array(data_set.drop(columns='Target').columns)
    ranking_df= pd.DataFrame( importances,features).reset_index()
    ranking_df.columns=[ 'features','importances']
    ranking_df=ranking_df.sort_values(by='importances',ascending=False)
    return list(ranking_df.iloc[0:numVars:,0])
def compute_accuracy_m(algo,x_train, y_train, x_test, name=' '):
    '''
    This function generates accuracy scores for a given multiclass classification algorithm that is already fit.
    It the needed packages are in the function. 

    algo    = The algorithm used to compute your predictions (skleran object)
    x_train = Predictor variables from the training set (array of numeric predictor variables)
    y_train = Target varaibles from the training set (numeric multiclass target)
    name    = Name of the algorithm used to differentate the output. (str object)
    '''
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    y_pred_t= algo.predict(x_train)
    acc_train= accuracy_score(y_train,y_pred_t)
    f1_train= f1_score(y_train,y_pred_t, average='macro')
    y_pred_test= algo.predict(x_test)
    acc_test= accuracy_score(y_test, y_pred_test)
    f1_test= f1_score(y_test,y_pred_test, average='macro')
    print('Computed Accuracy Metrics: '+ name )
    print('|'+'Train Accuracy = '+str(round(acc_train,3))+'|')
    print('|'+'Test Accuracy = '+' '+str(round(acc_test,3))+'|')
    print('|----------------------|')
    print('|'+'Train Macro F1 = ' +str(round(f1_train,3))+'|')
    print('|'+'Test Macro F1 = ' +' '+str(round(f1_test,3))+'|')
    print('')
    print('')
def create_submission(algo, importantVars, sub='submission 1'):
    '''
    This function load the test data and imputes the missing values with the same strategy as before. Then it takes the algorithm,
    fits it to the test data, makes predictions and sends it to the location for submission.
    
    algo= The fit algorithm used to make projections. (fit sklearn object)
    importantVars= Variables that were fit in the final model choice. 
    sub=The name that will be 
    '''
    submit_set= pd.read_csv('../input/test.csv')
    X= submit_set.loc[:,importantVars]
    imput=SimpleImputer(strategy='mean')
    imput.fit(X)
    X_np=imput.transform(X)
    names=X.columns
    X= pd.DataFrame(X_np)
    X.columns=names
    
    y_pred= pd.DataFrame(algo.predict(X)).astype(int)
    submission= pd.concat([submit_set.loc[:,'Id'],pd.DataFrame(y_pred)], axis=1)
    names= ['Id', 'Target']
    submission.columns=names
    return submission.to_csv(sub+'.csv', index = False)
#Now I will use the rank_features function I built to filter the dataframe by the important variables and then split the set into train and test. 

vars_i= rank_features(data_set)
vars_i.append('Target')
data_set= data_set.loc[:,vars_i]
np.random.seed(4672)
X_train, X_test, y_train, y_test = train_test_split(data_set.drop(columns='Target'),data_set.loc[:,'Target'],
                                                    test_size=0.30, random_state=None)
#Fit the Random Forest (I played around with the parameters and I believe it is still overfit. I will fix this in future iterations.)
random_forest= RandomForestClassifier(n_estimators=155, criterion='entropy', max_features=20, bootstrap=False)
random_forest.fit(X_train,y_train)

#Calculate the accuracy metrics
compute_accuracy_m(random_forest,X_train, y_train, X_test, name='First Random Forest')
#Try Randomforest with bootstrap
random_forest_boot= RandomForestClassifier(n_estimators=105, criterion='entropy', max_features=20, bootstrap=True)
random_forest_boot.fit(X_train,y_train)
#Modify the varlist for prediction on the test set and write the file to a csv for submission. Most of the code and explaination is in the function above. 
vars_i.remove('Target')
create_submission(random_forest, vars_i, 'random_forest')
create_submission(random_forest_boot, vars_i, 'random_forest_boot')