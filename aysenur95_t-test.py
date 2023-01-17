d=[{'name': 'Decision Tree', 'precision': 0.42, 'recall': 0.40 , 'f1': 0.41},

{'name': 'Naive Bayes', 'precision': 0.18 , 'recall': 0.72 , 'f1': 0.29 },

{'name': 'SVC', 'precision':  0.84, 'recall': 0.14, 'f1': 0.25 },

 {'name': 'kNN', 'precision': 0.62, 'recall': 0.07, 'f1': 0.13 },

 {'name': 'Logistic Regression', 'precision': 0.65, 'recall': 0.07, 'f1': 0.13 },

 {'name': 'Random Forest', 'precision': 0.88, 'recall': 0.32, 'f1': 0.47 },

 {'name': 'Bagging', 'precision': 0.96, 'recall':  0.30, 'f1': 0.45 },

 {'name': 'ExtraTrees', 'precision': 0.91, 'recall': 0.31, 'f1': 0.46}, 

 {'name': 'GradientBoost', 'precision': 0.95, 'recall': 0.33 , 'f1': 0.49 },

 {'name': 'AdaBoost', 'precision':  0.71, 'recall': 0.20, 'f1': 0.31 },

 {'name': 'CatBoost', 'precision': 0.95, 'recall': 0.33, 'f1': 0.48},

 {'name': 'XgBoost', 'precision': 0.94, 'recall': 0.33, 'f1': 0.49},

 {'name': 'LightGBM', 'precision': 0.95, 'recall': 0.33, 'f1': 0.49}]
import pandas as pd 

df = pd.DataFrame().append(d, ignore_index=True)

df = df.set_index('name')

df
df.plot.bar()
# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
#read data

train_features_data = pd.read_csv('../input/hr-dataset/train_LZdllcl.csv')

test_features_data = pd.read_csv('../input/hr-dataset/test_2umaH9m.csv')

train_features_data.drop(['employee_id'], axis="columns", inplace=True)

train_features_data.head()
cat_fetaures_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtype == object:

        cat_fetaures_col.append(column)

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        print("-------------------------------------------")

        
#numeric-cat ==> discrete

disc_feature_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() <= 30:

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        disc_feature_col.append(column)

        print("-------------------------------------------")

        

disc_feature_col.remove('is_promoted')
cont_feature_col=[]

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() > 30:

        print(f"{column} : Minimum: {train_features_data[column].min()}, Maximum: {train_features_data[column].max()}")

        cont_feature_col.append(column)

        print("-------------------------------------------")
#eliminate null values(fill with mode of that column)



for column in train_features_data.columns:

    train_features_data[column].fillna(train_features_data[column].mode()[0], inplace=True)
#there are no missing values in our dataset anymore!!!

train_features_data.isnull().sum()
# find the IQR



q1 = train_features_data[cont_feature_col].quantile(.25)

q3 = train_features_data[cont_feature_col].quantile(.75)

IQR = q3-q1



print("         IQR")

print("------------------------------\n")

print(IQR)

print("         q1")

print("------------------------------\n")

print(q1)

print("         q3")

print("------------------------------\n")

print(q3)





lower_bound = q1 - 1.5*IQR

upper_bound = q3 + 1.5*IQR

print("\n--------lower bounds--------")

print(lower_bound)

print("\n--------upper bound---------")

print(upper_bound)
outliers_df = np.logical_or((train_features_data[cont_feature_col] < lower_bound), (train_features_data[cont_feature_col] > upper_bound)) 

outliers_df
outlier_det_age_df=train_features_data[cont_feature_col]['age']

print(type(outlier_det_age_df))



outlier_det_los_df=train_features_data[cont_feature_col]['length_of_service']
outlier_age=train_features_data[cont_feature_col]['age'] > upper_bound[0]

print(outlier_age.head())



outlier_los=train_features_data[cont_feature_col]['length_of_service'] > upper_bound[1]

print(outlier_los.head())
#fill outliers with mean

outlier_det_age_df[outlier_age]=upper_bound[0]

print(outlier_det_age_df[outlier_age])



print("=======================")

outlier_det_los_df[outlier_los]=upper_bound[1]

print(outlier_det_los_df[outlier_los])





#update original train set with fixed outlier values

train_features_data['age']=outlier_det_age_df

train_features_data['length_of_service']=outlier_det_los_df
train_features_data.isnull().sum()
#encode ediyoruzzz!!!



#encoding categorical features (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(train_features_data)

train_features_data_arr=enc.transform(train_features_data)



col_names_list=train_features_data.columns

encoded_categorical_df=pd.DataFrame(train_features_data_arr, columns=col_names_list)
binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() <= 2] 

binary_cols.remove('is_promoted')



non_binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() > 2]
encoded_categorical_df
from sklearn.model_selection import train_test_split



y = encoded_categorical_df.loc[:, 'is_promoted'].values

X = encoded_categorical_df.drop('is_promoted', axis=1)



# split data into 80-20 for training set / test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=100)
#normalization(make all values bet. 0-1)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train[non_binary_cols])



X_train_normalized_arr=scaler.transform(X_train[non_binary_cols])

X_train_normalized_df=pd.DataFrame(X_train_normalized_arr, columns=non_binary_cols)



X_test_normalized_arr=scaler.transform(X_test[non_binary_cols])

X_test_normalized_df=pd.DataFrame(X_test_normalized_arr, columns=non_binary_cols)
X_train_binary_cols_df = X_train[binary_cols]

X_train_binary_cols_df.reset_index(inplace=True, drop=True)



X_train_final_df = pd.concat([X_train_binary_cols_df,X_train_normalized_df], axis=1)



X_train_final_df.head()
X_test_binary_cols_df = X_test[binary_cols]

X_test_binary_cols_df.reset_index(inplace=True, drop=True)



X_test_final_df = pd.concat([X_test_binary_cols_df,X_test_normalized_df], axis=1)



X_test_final_df.head()
X_train_final_df.shape
X_test_final_df.shape
df_X=pd.concat([X_train_final_df,X_test_final_df], axis=0)
print(df_X.shape)

df_X
y_train=pd.DataFrame(y_train)

y_test=pd.DataFrame(y_test)
df_y=pd.concat([y_train,y_test], axis=0)
print(df_y.shape)

df_y
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import mean_absolute_error

from scipy import stats



from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

import lightgbm as lgb



#test_size=%20

n_splits = 5



sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=42, test_size=0.2)



model_1 = GradientBoostingClassifier(random_state=0, max_depth=5, max_features= None, n_estimators=100, subsample=1)

model_2 = XGBClassifier(random_state=0, gamma=5, max_depth=4, n_estimators=200, subsample=0.8)

model_3 = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01, metric='auc', 

                             lambda_l1=1.5, lambda_l2=1, min_data_in_leaf=30, num_leaves=31, reg_alpha=0.1)





cv_mae_1 = []

cv_mae_2 = []

cv_mae_3 = []







for X_train_list, X_test_list in sss.split(df_X,df_y):

    model_1.fit(X.loc[X_train_list], y[X_train_list])

    pred_1 = model_1.predict(X.loc[X_test_list])

    err_1 = mean_absolute_error(y[X_test_list], pred_1)

    cv_mae_1.append(err_1)





    model_2.fit(X.loc[X_train_list], y[X_train_list])

    pred_2 = model_2.predict(X.loc[X_test_list])

    err_2 = mean_absolute_error(y[X_test_list], pred_2)

    cv_mae_2.append(err_2)



    model_3.fit(X.loc[X_train_list], y[X_train_list])

    pred_3 = model_3.predict(X.loc[X_test_list])

    err_3 = mean_absolute_error(y[X_test_list], pred_3)

    cv_mae_3.append(err_3)

from scipy import stats

print(stats.ttest_rel(cv_mae_1,cv_mae_2))

print(stats.ttest_rel(cv_mae_3,cv_mae_2))

print(stats.ttest_rel(cv_mae_3,cv_mae_1))