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
#unzippig the us-data file

#from shutil import unpack_archive

#unpack_archive('all.zip')

#unpack_archive('train.csv.zip')

#unpack_archive('test.csv.zip')
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

train = pd.read_csv('/kaggle/input/dsn-staff-promotion-algorithm/train.csv')

test = pd.read_csv('/kaggle/input/dsn-staff-promotion-algorithm/test.csv')
sample=pd.read_csv('/kaggle/input/dsn-staff-promotion-algorithm/sample_submission2.csv')
sample.head()
train_copy=train.copy()

test_copy=test.copy()
##display the first five rows of the train dataset.

train.head()
##display the first five rows of the test dataset.

test.head()
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'Id' column

train_ID = train['EmployeeNo']

test_ID = test['EmployeeNo']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("EmployeeNo", axis = 1, inplace = True)

test.drop("EmployeeNo", axis = 1, inplace = True)
#save and drop the target varriable

y_train=train['Promoted_or_Not']
y_train.value_counts()
#dropping the target varriable

train.drop('Promoted_or_Not',axis=1,inplace=True)
train.dtypes
#separate categorical varriable from numerical varriable

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

#quantitative.remove('SalePrice')

#quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
quantitative
qualitative
#missing values

sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
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
cols_with_missing = [col for col in train.columns if train[col].isnull().any()]



reduced_train =train.drop(cols_with_missing, axis=1)

reduced_test = test.drop(cols_with_missing, axis=1)
#Handling missing values

train['Qualification'].unique()

train['Qualification'].fillna('uncertified',inplace=True)

test['Qualification'].fillna('uncertified',inplace=True)
#function for adding School rank

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
'''train/'''

test['School_rank'].value_counts()
'''def ConvertQualificationToFeature(desc):

  Qualification={

      'MSc, MBA and PhD':5,

      'First Degree or HND':3,

      'Non-University Education':1,

      'uncertified':1

      }

  return Qualification[desc]'''
#train['Qualification']=train['Qualification'].apply(ConvertQualificationToFeature)

#test['Qualification']=test['Qualification'].apply(ConvertQualificationToFeature)
train.head()

#qualification done
train['Gender'].unique()
#one hot encode gender since it does not show hirarchy male cant be said to be more importance than female

train['Gender'].value_counts()
#one hot encode/label encode

train['Division'].value_counts()
train['Division'].unique()
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
train.head()
#one encode/label encode

train['Channel_of_Recruitment'].value_counts()
train['Channel_of_Recruitment'].unique()
#function for handling channel of recruitment

def convertChannelToFeature(desc):

  Channel={

      'Direct Internal process':'DIP',

      'Agency and others':'AO',

      'Referral and Special candidates':'RSC'

      }

  return Channel[desc]



train['Channel_of_Recruitment']=train['Channel_of_Recruitment'].apply(convertChannelToFeature)

test['Channel_of_Recruitment']=test['Channel_of_Recruitment'].apply(convertChannelToFeature)
train.head()
qualitative
train['State_Of_Origin'].value_counts()
#categorized into six geo-political zones

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
train.head()
#hirarchy coding 2,1 

#one hot encode first later perform hirarchical encoding

train['Foreign_schooled'].value_counts()
#one ehot encode

train['Marital_Status'].value_counts()
#Do Hirarchical encoding NO:1  ,yes :0

train['Past_Disciplinary_Action'].value_counts()
'''def ConvertDisciplinaryToFeature(desc):

  DisciplinaryAction={

      'Yes':0,

      'No':1

      

  }

  return DisciplinaryAction[desc]'''
#it is logical to rate those without dicsciplinary action higher than those that have

#train['Past_Disciplinary_Action']=train['Past_Disciplinary_Action'].apply(ConvertDisciplinaryToFeature)

#test['Past_Disciplinary_Action']=test['Past_Disciplinary_Action'].apply(ConvertDisciplinaryToFeature)
train.head()
'''def Previous_IntraDepartmental_Movement(desc):

  Movement={

      'Yes':1,

      'No':0

      

  }

  return Movement[desc]'''


#You might want to try rating those with more intra deptmental movement high cos looks like they might understand the flow of the business more

#will try one hot encoding first

train['Previous_IntraDepartmental_Movement'].value_counts()
#train['Previous_IntraDepartmental_Movement']=train['Previous_IntraDepartmental_Movement'].apply(Previous_IntraDepartmental_Movement)

#test['Previous_IntraDepartmental_Movement']=test['Previous_IntraDepartmental_Movement'].apply(Previous_IntraDepartmental_Movement)
train.head()
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


train['No_of_previous_employers'].value_counts()
train['No_of_previous_employers'].unique()
test.head()
#no of years spent in establishment can be calculated

train['Year_of_recruitment'].value_counts()


#age feature could be added

train['Year_of_birth'].value_counts()
#function for handling year difference

from datetime import date

def CalculateYear(year):

  today=date.today()

  age=today.year-year

  return age

#Number of years spent

train['No_Of_Year_Spent']=train['Year_of_recruitment'].apply(CalculateYear)

test['No_Of_Year_Spent']=test['Year_of_recruitment'].apply(CalculateYear)



#age

train['Age_in_years']=train['Year_of_birth'].apply(CalculateYear)

test['Age_in_years']=test['Year_of_birth'].apply(CalculateYear)
train.head()

#dropping Year_of_recruitment and Year_of_birth

train.drop('Year_of_recruitment', axis = 1, inplace = True)

test.drop('Year_of_recruitment', axis = 1, inplace = True)



train.drop('Year_of_birth', axis = 1, inplace = True)

test.drop('Year_of_birth', axis = 1, inplace = True)

train.head()
train.dtypes
train['Last_performance_score']=train['Last_performance_score'].round().astype(int)

test['Last_performance_score']=test['Last_performance_score'].round().astype(int)

#train['Last_performance_score']=np.log1p(train['Last_performance_score'])

#sns.distplot(np.log1p(train['Last_performance_score']))
#train['Training_score_average']=np.log1p(train['Training_score_average'])

#sns.distplot(np.log1p(train['Training_score_average']))
#test['Last_performance_score']=np.log1p(test['Last_performance_score'])

#test['Training_score_average']=np.log1p(test['Training_score_average'])



'''0 Division                                object

1 Qualification                           int64

2 Gender                                  object

3 Channel_of_Recruitment                  object

4 Trainings_Attended     *                  int64

5 Last_performance_score *               float64

6 Targets_met     *                         int64

7 Previous_Award    *                       int64

8 Training_score_average  *                 int64

9 State_Of_Origin                         object

10 Foreign_schooled                        object

11 Marital_Status                          object

12 Past_Disciplinary_Action                 int64

13 Previous_IntraDepartmental_Movement      int64

14 No_of_previous_employers                 int64

15 No_Of_Year_Spent  *                       int64

16 Age_in_years  *'''



[0,1,2,3,9,10,11,12,13,14]
['Trainings_Attended',

 'Year_of_birth',

 'Last_performance_score',

 'Year_of_recruitment',

 'Targets_met',

 'Previous_Award',

 'Training_score_average']
qualitative
#extracting the index of categorical varriable

#categorical_var = np.where((train.dtypes != np.float)&(train.dtypes != np.int))[0]

#train.dtypes[f] != 'object']
#categorical_var
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

    

 

#one-hot encoder

'''for column in qualitative_new :

    dummies = pd.get_dummies(train[column], prefix=column[:5])

    train = pd.concat([train, dummies], axis=1)



for column in qualitative_new:

    dummies = pd.get_dummies(test[column], prefix=column[:5])

    test = pd.concat([test, dummies], axis=1)'''





'''

#LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

qualitative_4_labelEncode=['Division',

 'Qualification',

 'Channel_of_Recruitment',

 'State_Of_Origin'

  ]





for column in qualitative_4_labelEncode :

    train[column] = le.fit_transform(train[column])

    



for column in qualitative_4_labelEncode :

    test[column] = le.fit_transform(test[column])'''



'''#LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

qualitative_4_labelEncode=['Division',

 'Qualification',

 'Channel_of_Recruitment',

 'State_Of_Origin'

  ]





for column in qualitative_4_labelEncode :

    train[column] = le.fit_transform(train[column])

    



for column in qualitative_4_labelEncode :

    test[column] = le.fit_transform(test[column])'''











train.dtypes
#one hot encode categorical varriable

'''qualitative_4_one_hot=[

 'Gender',

 'Foreign_schooled',

 'Marital_Status',

 'Past_Disciplinary_Action',

 'Previous_IntraDepartmental_Movement',

  'No_of_previous_employers',

  ]



for column in qualitative_4_one_hot :

    dummies = pd.get_dummies(train[column], prefix=column[:5])

    train = pd.concat([train, dummies], axis=1)



for column in qualitative_4_one_hot:

    dummies = pd.get_dummies(test[column], prefix=column[:5])

    test = pd.concat([test, dummies], axis=1)

    

train.drop(qualitative_4_one_hot,axis=1,inplace=True)



test.drop(qualitative_4_one_hot,axis=1,inplace=True)'''
train.dtypes
'''#one hot encoding the labeled encoded feature

qualitative_4_labelEncode=['Division',

 'Qualification',

 'Channel_of_Recruitment',

 'State_Of_Origin'

  ]



for column in qualitative_4_labelEncode :

    dummies = pd.get_dummies(train[column], prefix=column[:5])

    train = pd.concat([train, dummies], axis=1)



for column in qualitative_4_labelEncode:

    dummies = pd.get_dummies(test[column], prefix=column[:5])

    test = pd.concat([test, dummies], axis=1)

    

train.drop(qualitative_4_labelEncode,axis=1,inplace=True)



test.drop(qualitative_4_labelEncode,axis=1,inplace=True)'''
train.head()
#drop all categorical column

#train.drop(qualitative_new,axis=1,inplace=True)
#test.drop(qualitative_new,axis=1,inplace=True)
#train.drop('Qualification',axis=1,inplace=True)

#test.drop('Qualification',axis=1,inplace=True)
#converting from uint8 to int64

'''train[['Divis_BFO',

       'Divis_CSFO',

       'Divis_CSM','Divis_IS',

       'Divis_ITSS',

       'Divis_PHM',

       'Divis_RI',

       'Divis_RLS',

       'Divis_SP',                                

       'Gende_Female',                            

       'Gende_Male',                              

       'Chann_AO',                               

       'Chann_DIP',                             

       'Chann_RSC',                               

       'State_NC',                               

       'State_NE',                                

       'State_NW',                               

       'State_SE',                               

       'State_SS',                               

       'State_SW',                                

       'Forei_No',                               

       'Forei_Yes',                              

       'Marit_Married',                           

       'Marit_Not_Sure',                         

       'Marit_Single']].astype(np.int64)'''
#train.dtypes
'''

0-Qualification  *                        int64

1-Trainings_Attended                     int64

2-Last_performance_score                 int64

3-Targets_met    *                        int64

4-Previous_Award  *                       int64

5-Training_score_average                 int64

6-Past_Disciplinary_Action  *             int64

7-Previous_IntraDepartmental_Movement *   int64

8-No_of_previous_employers               int64

9-No_Of_Year_Spent                       int64

10-Age_in_years                           int64

Divis_BFO                              int64

Divis_CSFO                             int64

Divis_CSM                              int64

Divis_IS                               int64

Divis_ITSS                             int64

Divis_PHM                              int64

Divis_RI                               int64

Divis_RLS                              int64

Divis_SP                               int64

Gende_Female                           int64

Gende_Male                             int64

Chann_AO                               int64

Chann_DIP                              int64

Chann_RSC                              int64

State_NC                               int64

State_NE                               int64

State_NW                               int64

State_SE                               int64

State_SS                               int64

State_SW                               int64

Forei_No                               int64

Forei_Yes                              int64

Marit_Married                          int64

Marit_Not_Sure                         int64

Marit_Single'''



[0,3,4,6,7]



['Division',

 'Qualification',

 'Gender',

 'Channel_of_Recruitment',

 'State_Of_Origin',

 'Foreign_schooled',

 'Marital_Status',

 'Past_Disciplinary_Action',

 'Previous_IntraDepartmental_Movement',

 'No_of_previous_employers']

#This column is actually categorical column change to category and retry with other algorithm

#train['Targets_met'].unique()
#This column is actually categorical column change to category and retry with other algorithm

#train['Previous_Award'].unique()
train_processed_copy=train.copy()

test_processed_copy=test.copy()
#train=train_processed_copy.copy()

#test=test_processed_copy.copy()
'''#Buiding Our Model

from sklearn.ensemble import (

    RandomForestClassifier,

    ExtraTreesClassifier,

    AdaBoostClassifier,

)



from lightgbm import LGBMClassifier 

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import KFold,train_test_split, cross_val_score

models = []

models.append(( 'Adab' , AdaBoostClassifier()))

models.append(( 'ET' , ExtraTreesClassifier()))

models.append(( 'XGB' , XGBClassifier()))

models.append(( 'RF' , RandomForestClassifier()))

models.append(( 'LGBM' , LGBMClassifier()))





results = []

names = []



for name, model in models:

    Kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, x_trn, y_trn, cv=Kfold, scoring= 'accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());

    print(msg)'''
'''#Hyper-parameter tuning xgboost

#import numpy as np

#import pandas as pd



from hyperopt import hp, tpe

from hyperopt.fmin import fmin



from sklearn.model_selection import cross_val_score, StratifiedKFold

#from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer



#import xgboost as xgb



#import lightgbm as lgbm'''
'''#gini functions

def gini(truth, predictions):

    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))]

    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(truth) + 1) / 2.

    return gs / len(truth)



def gini_xgb(predictions, truth):

    truth = truth.get_label()

    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)



def gini_lgb(truth, predictions):

    score = gini(truth, predictions) / gini(truth, truth)

    return 'gini', score, True



def gini_sklearn(truth, predictions):

    return gini(truth, predictions) / gini(truth, truth)



gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)'''
'''#tuning xgboost

def objective(params):

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf = XGBClassifier(

        n_estimators=200,

        learning_rate=0.05,

        n_jobs=4,

        **params

    )

    

    score = cross_val_score(clf,x_trn, y_trn, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'max_depth': hp.quniform('max_depth', 2, 8, 1),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

    'min_child_weight': hp.uniform('min_child_weight', 1, 6),

    'gamma': hp.uniform('gamma', 0.0, 0.5),

    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.07)

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)



'''
#optimum parameter for xgboost

#print("Hyperopt estimated optimum {}".format(best))
'''#tuning  for lightgbm

def objective(params):

    params = {

        'num_leaves': int(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf = LGBMClassifier(

        n_estimators=500,

        learning_rate=0.01,

        **params

    )

    

    score = cross_val_score(clf, x_trn, y_trn, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)'''
#optimum parameter for lgbm

#print("Hyperopt estimated optimum {}".format(best))
#tuning for rf

'''def objective(params):

    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}

    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)

    score = cross_val_score(clf, x_trn, y_trn, scoring=gini_scorer, cv=StratifiedKFold()).mean()

    print("Gini {:.3f} params {}".format(score, params))

    return score



space = {

    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),

    'max_depth': hp.quniform('max_depth', 1, 10, 1)

}



best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)'''
#optimum parameter for rf

#print("Hyperopt estimated optimum {}".format(best))
train=train_processed_copy.copy()

test=test_processed_copy.copy()
from sklearn.model_selection import train_test_split

x_trn,x_valid, y_trn, y_valid = train_test_split(train,

                                                   y_train,

                                                   test_size=0.3,

                                                   random_state=0)

#stacking the models

#Buiding Our Model

from sklearn.ensemble import (

    RandomForestClassifier,

    ExtraTreesClassifier,

    AdaBoostClassifier,

)



#from vecstack import stacking

from mlxtend.classifier import StackingCVClassifier

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier 

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import KFold,train_test_split, cross_val_score

random_state=1

max_features=min(30,train.columns.size)



ab_clf = AdaBoostClassifier(n_estimators=200,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=random_state),

                            random_state=random_state)



et_clf = ExtraTreesClassifier(n_estimators=300,

                              min_samples_leaf=2,

                              min_samples_split=2,

                              max_depth=10,

                              max_features=max_features,

                              random_state=random_state,

                              n_jobs=1)



lg_clf = LGBMClassifier(n_estimators=200,

                        num_leaves=126,

                        colsample_bytree=0.7,

                        verbose=-1,

                        random_state=random_state,

                        n_jobs=1)



rf_clf = RandomForestClassifier(n_estimators=250,

                                max_depth=7,

                                random_state=random_state,

                                n_jobs=1)





xg_clf= XGBClassifier(

 learning_rate =0.1,

 n_estimators=200,

 max_depth=7,

 min_child_weight=2,

 gamma=0.2,

 subsample=0.9,

 colsample_bytree=0.7,

 reg_alpha=0.05,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



 #estimated optimum {'colsample_bytree': 0.33751943330694145, 'gamma': 0.4139723161576815, 'max_depth': 4.0, 'min_child_weight': 5.872975495460509, 'reg_alpha': 0.030289941526500358}







    

    

    













models = []

models.append(( 'Adab' ,ab_clf))

models.append(( 'ET' , et_clf))

models.append(( 'XGB' ,xg_clf ))

models.append(( 'RF' , rf_clf ))

models.append(( 'LGBM' , lg_clf ))





results = []

names = []



print('> Cross-validating classifiers')

for name, model in models:

    Kfold = KFold(n_splits=10, random_state=0)

    cv_results = cross_val_score(model, train, y_train, cv=Kfold, scoring= 'accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());

    print(msg)
print('> Fitting stack')

# - https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17

# - https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python



stack = StackingCVClassifier(classifiers=[clf for label, clf in models],

                             meta_classifier=xg_clf ,

                             cv=3,

                             use_probas=True, 

                             use_features_in_secondary=True,

                             verbose=-1,

                             )





stack = stack.fit(train.as_matrix(), y_train.as_matrix())



#Our Score on Validation Data

stack.score(x_valid.as_matrix(),y_valid.as_matrix())
#Our Score on Training data

stack.score(train.as_matrix(),y_train.as_matrix())
#Prediction on Test Set

predict_y_stack = stack.predict(test.as_matrix())
print('> Creating submission')

#Prepare our Submission file

my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_stack })

my_submission.to_csv('submission_ayanlola.csv', index=False)



print('> Done !')
!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
!kaggle competitions submissions -c intercampusai2019
# applying SMOTE cos of  large difference between 0s and 1s in target varriable



#from imblearn.over_sampling import SMOTE



#x_resample, y_resample = SMOTE().fit_sample(train, y_train.values.ravel()) 



# checking the shape of x_resample and y_resample

#print("Shape of x:", x_resample.shape)

#print("Shape of y:", y_resample.shape)



from imblearn.over_sampling import ADASYN 

sm = ADASYN()

train_resample,y_train_resample = sm.fit_sample(train,y_train)

train_resample = pd.DataFrame(train_resample, columns = train.columns)

y_train_resample=pd.DataFrame(y_train_resample)
# checking the shapes

print(train.shape)

print(train_resample.shape)

print(y_train_resample.shape)
# train and valid sets from train

#from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



x_trn, x_valid, y_trn, y_valid = train_test_split(train_resample, y_train_resample, test_size = 0.2, random_state = 0)



# checking the shapes

#print(x_trn.shape)

#print(y_train.shape)

#print(x_valid.shape)

#print(y_valid.shape)
#y_trn.head()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

train = pd.DataFrame(sc.fit_transform(train),columns=train.columns).astype(int)

test = pd.DataFrame(sc.transform(test),columns=test.columns).astype(int)
test.dtypes
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

x_trn, x_valid, y_trn, y_valid = train_test_split(train, y_train, test_size = 0.2, random_state = 0)

model_RF=RandomForestClassifier(n_estimators=200,random_state=200,max_features=0.5,min_samples_leaf=3,oob_score=True,n_jobs=-1)

model_RF.fit(x_trn,y_trn)
#Our Score on Validation Data

model_RF.score(x_valid,y_valid)
#Prediction on Test Set

predict_y_RF = model_RF.predict(test)
#Our Score on Training data

model_RF.score(x_trn,y_trn)
#Prepare our Submission file

#my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_RF })

#my_submission.to_csv('submission_ayanlola.csv', index=False)
#!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
#!kaggle competitions submissions -c intercampusai2019
#Applying ADABoost

#from sklearn.ensemble import AdaBoostClassifier

#from sklearn.model_selection import train_test_split



#x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 0)
'''model_ada = AdaBoostClassifier(n_estimators=200,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=random_state),

                            random_state=random_state)'''

#model_ada.fit(x_trn,y_trn)
#Our Score on Validation Data

#model_ada.score(x_valid,y_valid)
#Prediction on Test Set

#predict_y_ada = model_ada.predict(test)
#Our Score on Training data

#model_ada.score(x_trn,y_trn)
#Prepare our Submission file

#my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_ada })

#my_submission.to_csv('submission_ayanlola.csv', index=False)
#!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
#!kaggle competitions submissions -c intercampusai2019
'''import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_validate

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV   #Perforing grid search



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4



from sklearn.model_selection import train_test_split'''



#y_train.values
def modelfit(alg,dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        #x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 42)

        xgtrain = xgb.DMatrix(train.values, label=y_train.values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds,show_progress=None)

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
'''#Choose all predictors except target & IDcols

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

modelfit(xgb1,train,predictors)'''
'''param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[predictors],y_train)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_'''
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
#gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_
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
train=train_processed_copy.copy()

test=test_processed_copy.copy()
from sklearn.model_selection import train_test_split

x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 42)
#standardization techniques

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_trn = sc.fit_transform(x_trn)

x_valid = sc.transform(x_valid)

test = sc.transform(test)
#Applying XGBOOST

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import average_precision_score

#from sklearn.model_selection import train_test_split



#x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 42)

#x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 42)

print(x_trn.shape)

print(y_trn.shape)

print(x_valid.shape)

print(y_valid.shape)
#parameter tuning

'''from sklearn import metrics

from sklearn.model_selection import GridSearchCV



model_xg =XGBClassifier()

param_dist = {"max_depth": [3,5,7],

              "min_child_weight" : [1,3,6],

              "n_estimators": [140],

              "learning_rate": [0.05, 0.1,0.16],

              "gamma":[0]}

grid_search = GridSearchCV(model_xg, param_grid=param_dist, cv = 3, 

                                   verbose=10, n_jobs=-1)

grid_search.fit(train, y_train)



grid_search.best_estimator_'''


model_xg = XGBClassifier(

 learning_rate =0.1,

 n_estimators=200,

 max_depth=7,

 min_child_weight=2,

 gamma=0.2,

 subsample=0.9,

 colsample_bytree=0.7,

 reg_alpha=0.05,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

model_xg.fit(x_trn,y_trn)



''' estimated optimum {'colsample_bytree': 0.33751943330694145, 'gamma': 0.4139723161576815, 

                    'max_depth': 4.0, 'min_child_weight': 5.872975495460509, 

                    'reg_alpha': 0.030289941526500358}'''



print(train.shape)

print(y_train.shape)
'''model_xg =XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,

                        min_child_weight=1,gamma=0.3,subsample=0.8,colsample_bytree=0.8,

                        objective= 'binary:logistic',nthread=4,scale_pos_weight=1,reg_alpha=1e-05,

                        reg_lambda=1,seed=27)



model_xg.fit(train,y_train)'''
#Our Score on Validation Data

model_xg.score(x_valid,y_valid)
#Prediction on Test Set

predict_y_xg = model_xg.predict(test)
#Our Score on Training data

model_xg.score(x_trn,y_trn)
# making a classification report

cr = classification_report(y_trn,predict_y_xg)

print(cr)



# making a confusion matrix

cm = confusion_matrix(y_trn,predict_y_xg)

sns.heatmap(cm, annot = True)
# Calculating the avg precision score

from sklearn.metrics import average_precision_score



apc = average_precision_score(y_train,predict_y_xg)

print('Average Precision Score :', apc)




precision, recall, _ = precision_recall_curve(y_train,predict_y_xg)



step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



plt.step(recall, precision, color = 'pink', alpha = 0.6, where = 'post')

plt.fill_between(recall, precision, color = 'pink', alpha = 0.6, **step_kwargs)



plt.title('Precision Recall Curve')

plt.xlabel('Recall', fontsize = 15)

plt.ylabel('Precision', fontsize =15)

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])
#Prepare our Submission file

my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_xg })

my_submission.to_csv('submission_ayanlola.csv', index=False)
!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
!kaggle competitions submissions -c intercampusai2019
#Applying light boost classifier

from lightgbm import LGBMClassifier

#from sklearn.model_selection import train_test_split



#x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 0)





















#model_lgb =LGBMClassifier(n_estimators=400,num_leaves=100,verbosity=0)

#model_lgb.fit(train,y_train)
#Our Score on Validation Data

#model_lgb.score(x_valid,y_valid)
#Prediction on Test Set

#predict_y_lgb = model_lgb.predict(test)


#Our Score on Training data

#model_lgb.score(x_trn,y_trn)
#Prepare our Submission file

#my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_lgb })

#my_submission.to_csv('submission_ayanlola.csv', index=False)
#!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
#!kaggle competitions submissions -c intercampusai2019
!pip install catboost==0.7.2
#applying catboost

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.model_selection import train_test_split



x_trn,x_valid, y_trn, y_valid = train_test_split(train,y_train, test_size = 0.2, random_state = 0)

#categorical_var = np.where(train_x.dtypes != np.float)[0]
cat_features=[0,3,4,6,7]

model_cat = CatBoostClassifier(iterations = 5000,

                                

                                eval_metric='AUC',

                                loss_function= 'Logloss',

                                learning_rate=0.003,

                                depth=5,

                                leaf_estimation_iterations = 10

                                )



#model_cat=CatBoostClassifier()



model_cat.fit(train,y_train,cat_features)
#Our Score on Validation Data

model_cat.score(x_valid,y_valid)
#Prediction on Test Set

predict_y_cat = model_cat.predict(test)
#Our Score on Training data

model_cat.score(x_trn,y_trn)
#Prepare our Submission file

my_submission = pd.DataFrame({'EmployeeNo':test_ID, 'Promoted_or_Not': predict_y_cat })

my_submission.to_csv('submission_ayanlola.csv', index=False)
!kaggle competitions submit -c intercampusai2019 -f submission_ayanlola.csv -m "BOOTCAMP"
!kaggle competitions submissions -c intercampusai2019