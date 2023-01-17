# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats

from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
#train = pd.read_csv('train.csv',parse_dates=[['weblog_date'],['date_of_advert'],['last_advert_online']])

#test = pd.read_csv('test.csv',parse_dates=[weblog_date],[[date_of_advert],[last_advert_online])

train = pd.read_csv('/kaggle/input/intercampusai2019/train.csv')

test = pd.read_csv('/kaggle/input/intercampusai2019/test.csv')
#save and drop the target varriable

y_train=train['Promoted_or_Not']
#Save the 'Id' column

train_ID = train['EmployeeNo']

test_ID = test['EmployeeNo']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("EmployeeNo", axis = 1, inplace = True)

test.drop("EmployeeNo", axis = 1, inplace = True)



#dropping the target varriable

train.drop('Promoted_or_Not',axis=1,inplace=True)
#Data PreProcessing

# check number & percentage of missing value in the columns

def missing_values_table(df):

  mis_val = df.isnull().sum() #total missing values

  mis_val_percent = 100 * df.isnull().sum() / len(df) #percentage of missing values

  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) #make a table with the results

  mis_val_table_ren_columns = mis_val_table.rename(

  columns = {0 : 'Missing Values', 1 : '% of Total Values'}) #rename the columns

     # sort the table by percentage of missing value

  mis_val_table_ren_columns = mis_val_table_ren_columns[

  mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)



        #print same summary information

  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")



          # return the dataframe with missing information

  return mis_val_table_ren_columns

  

missing_values = missing_values_table(train)

missing_values.head()
#Handling Missing values

train['Qualification'].fillna('uncertified',inplace=True)

test['Qualification'].fillna('uncertified',inplace=True)
#function to defined the school ranking based on foreign school and qualification

def School_rank(Foreign_schooled,Qualification):

    if Foreign_schooled == 'Yes' and Qualification == 'MSc, MBA and PhD':

     return 5

    if Foreign_schooled == 'Yes' and Qualification == 'First Degree or HND':

     return 4

    if Foreign_schooled == 'No' and Qualification == 'MSc, MBA and PhD':

     return 3

    if Foreign_schooled == 'No' and Qualification == 'First Degree or HND':

     return 2

    else:

     return 1
#we could add hirarchical feature of people that foreigned schooled and PHD:4,with First degree :3,Local(PHD):3,Local Bsc:2,uneducated:1,noinfo:1 

train['School_rank']=train.apply(lambda x:School_rank(x['Foreign_schooled'],x['Qualification']),axis=1)

test['School_rank']=test.apply(lambda x:School_rank(x['Foreign_schooled'],x['Qualification']),axis=1)
#adding new division description

def ConvertDivisionToFeature(desc):

  Division={

      'Commercial Sales and Marketing':'CSM',

      'Customer Support and Field Operations':'CSFO',

      'Information and Strategy':'IS',

      'Information Technology and Solution Support':'ITSS',

      'Sourcing and Purchasing':'SP',

      'Business Finance Operations':'BFO',

      'People/HR Management':'PHM',

      'Research and Innovation':'RI',

      'Regulatory and Legal services':'RLS'

      }

  return Division[desc]
train['Division']=train['Division'].apply(ConvertDivisionToFeature)

test['Division']=test['Division'].apply(ConvertDivisionToFeature)
#function for channel 

def convertChannelToFeature(desc):

  Channel={

      'Direct Internal process':'DIP',

      'Agency and others':'AO',

      'Referral and Special candidates':'RSC'

      }

  return Channel[desc]



train['Channel_of_Recruitment']=train['Channel_of_Recruitment'].apply(convertChannelToFeature)

test['Channel_of_Recruitment']=test['Channel_of_Recruitment'].apply(convertChannelToFeature)
#categorized state into six geo-political zones

def ConvertToGeoPoliticalZone(desc):

  

  State={

      

      'BENUE':'NC',

      'KOGI':'NC',

      'KWARA':'NC',

      'NASSARAWA':'NC',

      'NIGER':'NC',

      'PLATEAU':'NC',

      'FCT':'NC',

      

      'ADAMAWA':'NE',

      'BAUCHI':'NE',

      'BORNO':'NE',

      'GOMBE':'NE',

      'TARABA':'NE',

      'YOBE':'NE',

      

      

          

      'JIGAWA':'NW',

      'KADUNA':'NW',

      'KANO':'NW',

      'KATSINA':'NW',

      'KEBBI':'NW',

      'SOKOTO':'NW',

      'ZAMFARA':'NW',

      

          

      'ABIA':'SE',

      'ANAMBRA':'SE',

      'EBONYI':'SE',

      'ENUGU':'SE',

      'IMO':'SE',

      

      'AKWA IBOM':'SS',

      'BAYELSA':'SS',

      'CROSS RIVER':'SS',

      'RIVERS':'SS',

      'DELTA':'SS',

      'EDO':'SS',

     

        

      'EKITI':'SW',

      'LAGOS':'SW',

      'OGUN':'SW',

      'ONDO':'SW',

      'OSUN':'SW',

      'OYO':'SW'

      }

  return State[desc]



#one hot encode/label encode

train['State_Of_Origin']=train['State_Of_Origin'].apply(ConvertToGeoPoliticalZone)

test['State_Of_Origin']=test['State_Of_Origin'].apply(ConvertToGeoPoliticalZone)
#function to handle number of previous employers rank

def ConvertNumberOfPreviousEmployerFeature(desc):

  Past={

      '0':'0',

      '1':'1',

      '2':'2',

      '3':'3',

      '4':'4',

      '5':'5',

      'More than 5':'7'

      }

  return Past[desc]



#This column is not actually numerical col change to numerical and retry for other models like LGB,XGboost,Randomforest

train['No_of_previous_employers']=train['No_of_previous_employers'].apply(ConvertNumberOfPreviousEmployerFeature)

test['No_of_previous_employers']=test['No_of_previous_employers'].apply(ConvertNumberOfPreviousEmployerFeature)
#function to calculate diff in year

from datetime import date

def CalculateYear(year):

  today=date.today()

  age=today.year-year

  return age
train['No_Of_Year_Spent']=train['Year_of_recruitment'].apply(CalculateYear)

test['No_Of_Year_Spent']=test['Year_of_recruitment'].apply(CalculateYear)



train['Age_in_years']=train['Year_of_birth'].apply(CalculateYear)

test['Age_in_years']=test['Year_of_birth'].apply(CalculateYear)
#dropping Year_of_recruitment and Year_of_birth

train.drop('Year_of_recruitment', axis = 1, inplace = True)

test.drop('Year_of_recruitment', axis = 1, inplace = True)



train.drop('Year_of_birth', axis = 1, inplace = True)

test.drop('Year_of_birth', axis = 1, inplace = True)
#rounding up last performance score to integer

train['Last_performance_score']=train['Last_performance_score'].round().astype(int)

test['Last_performance_score']=test['Last_performance_score'].round().astype(int)
#label encode one hot encode categorical varribles





qualitative_new=['Division',

 'Qualification',

 'Gender',

 'Channel_of_Recruitment',

 'State_Of_Origin',

 'Foreign_schooled',

 'Marital_Status',

 'Past_Disciplinary_Action',

 'Previous_IntraDepartmental_Movement',

  'No_of_previous_employers',

  ]



#LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for column in qualitative_new :

    train[column] = le.fit_transform(train[column])

    



for column in qualitative_new :

    test[column] = le.fit_transform(test[column])

    
#creating copies of train and test set

train_processed_copy=train.copy()

test_processed_copy=test.copy()
train
import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_validate

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV   #Perforing grid search



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4



from sklearn.model_selection import train_test_split
train.dtypes
def modelfit(alg,dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        #x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 42)

        xgtrain = xgb.DMatrix(train.values, label=y_train.values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors],y_train,eval_metric='auc')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

                    

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols

predictors =train.columns

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb1,train,predictors)
param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[predictors],y_train)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
'''param_test2 = {

 'max_depth':[4,5,7],

 'min_child_weight':[6,8,10,12]

}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=200, max_depth=5,

 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(train[predictors],y_train)

gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_'''
'''param_test2b = {

 'min_child_weight':[2,4,6,8]

}

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=1000, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2b.fit(train[predictors],y_train)

gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_'''
#modelfit(gsearch3.best_estimator_, train, predictors)

#gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
'''param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=4,

 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(train[predictors],y_train)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_'''
'''xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=7,

 min_child_weight=2,

 gamma=0.2,

 subsample=0.9,

 colsample_bytree=0.7,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb2, train, predictors)'''
'''param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=7,

 min_child_weight=2, gamma=0.2, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(train[predictors],y_train)

gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_'''
'''param_test5 = {

 'subsample':[i/100.0 for i in range(75,90,5)],

 'colsample_bytree':[i/100.0 for i in range(75,90,5)]

}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=7,

 min_child_weight=2, gamma=0.2, subsample=0.9, colsample_bytree=0.7,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch5.fit(train[predictors],y_train)

gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_'''
'''param_test6 = {

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=7,

 min_child_weight=2, gamma=0.2, subsample=0.9, colsample_bytree=0.7,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(train[predictors],y_train)

gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_'''
'''

xgb3 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=7,

 min_child_weight=2,

 gamma=0.2,

 subsample=0.9,

 colsample_bytree=0.7,

 reg_alpha=1,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb3, train, predictors)'''
'''xgb4 = XGBClassifier(

 learning_rate =0.03,

 n_estimators=5000,

 max_depth=7,

 min_child_weight=2,

 gamma=.2,

 subsample=0.9,

 colsample_bytree=0.7,

 reg_alpha=1,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb4, train, predictors)'''