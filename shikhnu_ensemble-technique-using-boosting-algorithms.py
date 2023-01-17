import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
# loading training data and reading top 5 records



df = pd.read_csv('../input/hr-analytics-analytics-vidya/train.csv')

df.head()
### Reading bottom 5 records



df.tail()
print("There are {} rows and {} columns in the training dataset.".format(df.shape[0],df.shape[1]))
# To know the datatypes of the column



df.info()
print("There are {} duplicate records.".format(df.shape[0] - len(df['employee_id'].unique())))
# Droping employee_id column as it doesnot provide any information



df.drop('employee_id',axis=1,inplace=True)
# Name of the columns



print("Column Names: {}".format(list(df.columns)))
# Column names into list



col_name = df.columns.to_list()
# To find out number of unique values and unique vales of a perticular column



for i in col_name:

    print("In the column - {}:".format(i))

    print("There are {0} Unique values".format(len(df[i].unique())))

    print("Unique vales in the column are - \n{}".format(list(df[i].unique())))

    print("")
df.info()
# Correlation Matrix



plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot=True)

plt.show()
# Count of each values in column

for i in col_name:

    plt.figure(figsize=(15,5))

    plt.title("Count of each values in column '{}'".format(i))

    sns.countplot(df[i])

    plt.show()
# Pair plot



sns.pairplot(df)

plt.show()
print("There are totally {} missing values in the dataset.".format(df.isnull().sum().sum()))
# Count of missing values in column



for i in col_name:

    if df[i].isnull().sum() > 0:

        print("There are {} missing values in the '{}' column.\n".format(df[i].isnull().sum(),i))
# Imputing missing values in column education with forwardfill



df['education'] = df['education'].ffill()
# Value count for column "length_of_service" when "previous_year_rating" isnull



df[df["previous_year_rating"].isnull() == True]['length_of_service'].value_counts()
# Imputing missing values in column "previous_year_rating" with "0" as length of service is 1 for missing values 



df['previous_year_rating'] = df['previous_year_rating'].fillna(0.0)
# Binning the age column



df['age'] = pd.cut(x=df['age'], bins=[20, 29, 39, 49], 

                    labels=['20 to 30', '30 to 40', '40+']) 
# Changing datatype 'category' to 'object'



df['age'] = df['age'].astype('object')
X = df.drop('is_promoted',axis=1)

y = df['is_promoted']
X_encode = pd.get_dummies(X,drop_first=True)
from sklearn import preprocessing 



scaler = preprocessing.RobustScaler() 

X_standard = scaler.fit_transform(X_encode) 

X_standard = pd.DataFrame(X_standard, columns =X_encode.columns) 
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier





Classifiers = {'0._XGBoost' : XGBClassifier(learning_rate =0.1, n_estimators=500, max_depth=5,subsample = 0.70,

                                            verbosity = 0, scale_pos_weight = 2.5,updater ="grow_histmaker",

                                            base_score  = 0.2),

               

               '1.CatBoost' : CatBoostClassifier(learning_rate=0.15, n_estimators=500, subsample=0.085, max_depth=5,

                                                 scale_pos_weight=2.5),

               

               '2.LightGBM' : LGBMClassifier(subsample_freq = 2, objective ="binary",importance_type = "gain",verbosity = -1,

                                             max_bin = 60,num_leaves = 300, boosting_type = 'dart',learning_rate=0.15, 

                                             n_estimators=500, max_depth=5, scale_pos_weight=2.5)}
from sklearn.ensemble import VotingClassifier



vc_model = VotingClassifier(estimators=[('XGBoost_Best', list(Classifiers.values())[0]), 

                                        ('CatBoost_Best', list(Classifiers.values())[1]),

                                        ('LightGBM_Best', list(Classifiers.values())[2]),

                                       ], 

                            voting='soft',weights=[2, 1, 3])



vc_model.fit(X_standard,y)
# Loading test dataset



df1 = pd.read_csv('../input/hr-analytics-analytics-vidya/test.csv')

df1.head()
# Performing all the step on the unseen data that was performed on historical data



df2 = df1.copy()



df1.drop('employee_id',axis=1,inplace=True)



df1['education'] = df1['education'].ffill()



df1['previous_year_rating'] = df1['previous_year_rating'].fillna(0.0)



df1['age'] = pd.cut(x=df1['age'], bins=[20, 29, 39, 49], labels=['20 to 30', '30 to 40', '40+']) 

df1['age'] = df1['age'].astype('object')



df1_encode = pd.get_dummies(df1,drop_first=True)



scaler = preprocessing.RobustScaler() 

df_standard = scaler.fit_transform(df1_encode) 

df_standard = pd.DataFrame(df_standard, columns =df1_encode.columns)
df2['is_promoted'] = vc_model.predict(df_standard)



df1=df2[['employee_id','is_promoted']]

df1.to_csv('Predict19.csv')