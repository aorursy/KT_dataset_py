# # For Data reading 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For data visualization

import matplotlib.pyplot as plt

import seaborn as sns





# For Feature Scaling & Feature Importance

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesRegressor



# For model building & scoreing

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score





# others

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test_df = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')

train_dict = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data_dictionary.csv')
# Let's find unique values of dependent feature

train_df["Stay"].unique()
# Let's create a dictionary for dependent feature

encode = {

    '0-10' : 1, '11-20' : 2, '21-30' : 3, '31-40' : 4, '41-50' : 5, '51-60' : 6, '61-70' : 7, '71-80' : 8,

    '81-90' : 9, '91-100' : 10, 'More than 100 Days' : 11

}

train_df['Stay'] = train_df['Stay'].map(encode)
# Let's check missing values

print('Train Dataset:::::::::::::::')

print(train_df.isnull().sum())

print("=========================================")

print('Test Dataset::::::::::::::::')

print(test_df.isnull().sum())
# Find features of missing values 

def NaNFeature(df):

    nan_feature = [n for n in df.columns if df[n].isnull().sum()>=1]

    return nan_feature
# Let's fill missing values of train detaset 

nan_features_train = NaNFeature(train_df)

for fillnan in nan_features_train:

    train_df[fillnan].fillna(train_df[fillnan].mode()[0], inplace=True)
# Let's fill missing values of test detaset 

nan_features_test = NaNFeature(test_df)

for fillnan in nan_features_test:

    test_df[fillnan].fillna(test_df[fillnan].mode()[0], inplace=True)
# Lets check missing values percentage

print('Train Dataset:::::::::::::::')

print(np.round(train_df.isnull().sum() * 100 / len(train_df), 4))

print("=========================================")

print('Test Dataset:::::::::::::::')

print(np.round(test_df.isnull().sum() * 100 / len(test_df), 4))
# Let's Find out categorical features through a function

def CatFeatures(df):

    features = [feature for feature in df.columns if df[feature].dtypes == "O"]

    return features
# categorical features of train dataset

cat_features_train = CatFeatures(train_df)

cat_features_train
# Let's check unique value of categorical features of train data

for i in cat_features_train:

    print(train_df[i].unique())
# categorical features of test dataset

cat_features_test = CatFeatures(test_df)

cat_features_test
for i in cat_features_test:

    print(test_df[i].unique())
# # Let's create a function to handle categorical features 

def CatToNumaric():

    # Handle categorical feature of train dataset

    for n in cat_features_train:

        num_data = dict(zip(train_df[n].unique(), range(len(train_df[n].unique()))))

        train_df[n] = train_df[n].map(num_data) # or train_df[n].replace(num_data, inplace=True)

        

    # Handle categorical features of test dataset

    for n in cat_features_test:

        num_data = dict(zip(test_df[n].unique(), range(len(test_df[n].unique()))))

        test_df[n] = test_df[n].map(num_data) # or test_df[n].replace(num_data, inplace=True)
# Let's check features data types

CatToNumaric()

print('Train Dataset:::::::::::::::')

print(train_df.dtypes)

print("=====================================")

print('Test Dataset:::::::::::::::')

print(test_df.dtypes)
# Let's see the train dictionary data to drop un necessary features

train_dict
# Lets drop features those are necessary so much

def DropFeatures(df):

    drop_features = {'case_id', 'Hospital_code', 'Hospital_type_code', 'patientid'}

    df.drop(drop_features, axis=1, inplace=True)

    return df
# Show train dataset

train_data = DropFeatures(train_df)

train_data.head()
# Show test dataset

test_data = DropFeatures(test_df)

test_data.head()
# create X_train & X_test for feature scaling 

X_train = train_data.iloc[: , :-1]

X_test = test_data



# y_train (depended feature)

y_train = train_data.iloc[: , -1]
# create function for scaling X_ data 

def FeatureScaler(df):

    min_max = MinMaxScaler()

    df = pd.DataFrame(min_max.fit_transform(df), columns=df.columns)

    return df
# Let's show final train dataset

X_train_final = FeatureScaler(X_train)

X_train_final.head()
# Let's show final test dataset

X_test_final = FeatureScaler(X_test)

X_test_final.head()
# Let's call Extra Trees Regressor function

feature_imp = ExtraTreesRegressor()

feature_imp.fit(X_train_final, y_train)

# Let's show the list of feature importance

feature_imp.feature_importances_
# Let's show a plot of ten (10) features

feature_importance = pd.Series(feature_imp.feature_importances_, index=X_train_final.columns)

feature_importance.nlargest(10).plot(kind='barh')

plt.show()
# Create model

stay_predict = RandomForestClassifier()

stay_predict.fit(X_train_final, y_train)
# Let's test the model

y_test = stay_predict.predict(X_test_final)

y_test
# For submission file we need 'case_id' so read sample_submission file

sample_sub_df = test_df = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/sample_sub.csv')
predection_df = pd.DataFrame()

predection_df['case_id'] = sample_sub_df['case_id'] 

predection_df['Stay'] = y_test



decode_prediction = { 1 : '0-10', 2 : '11-20', 3 : '21-30', 4 : '31-40', 5 : '41-50', 6 : '51-60', 7 : '61-70'

            ,8 : '71-80', 9 : '81-90', 10 : '91-100', 11 : 'More than 100 Days'}



predection_df['Stay'] = predection_df['Stay'].map(decode_prediction)

predection_df.head()
# Model score

stay_predict.score(X_train_final, y_train)
# Cross Validation 

score = cross_val_score(stay_predict, X_train_final, y_train.ravel(), cv=10)

score.mean()
submission = predection_df.copy()

submission.head()