import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns



from IPython.core.interactiveshell import InteractiveShell

from IPython.display import display



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import gc

import lightgbm as lgb





import re
pd.options.display.max_columns = None

InteractiveShell.ast_node_interactivity = "all"
traindf = pd.read_csv('../input/titanic/train.csv')

testdf = pd.read_csv('../input/titanic/test.csv')
print('check missing value on training dataset')

print('-'*120)

missing_value_1 = traindf.isnull().sum()

missing_value_1



print('check missing value on test dataset')

print('-'*120)

missing_value_2 = testdf.isnull().sum()

missing_value_2
traindf['Survived'].value_counts().plot(kind='bar',title='Unbalance target variable')
ax = testdf["Age"].hist(bins=15, color='blue', alpha=0.9)

ax.set(xlabel='Age', ylabel='Count')

plt.show()
ax = testdf["Fare"].hist(bins=15, color='blue', alpha=0.9)

ax.set(xlabel='Fare', ylabel='Count')

plt.show()
print('Fill missing value on train dataset')

print('-'*120)

traindf.loc[traindf.Cabin.isnull(), 'Cabin'] = 'Unknown'

traindf['Age'].fillna(28, inplace=True)

traindf['Embarked'].fillna('S', inplace=True)



print('Fill missing value on test dataset')

print('-'*120)

testdf['Age'].fillna(27, inplace=True)

testdf['Fare'].fillna(14, inplace=True)

testdf.loc[testdf.Cabin.isnull(), 'Cabin'] = 'Unknown'
print('Feature engineering: create new feature and change cabin value')

print('-'*120)

traindf['FamilySize'] = traindf['SibSp'] + traindf['Parch'] + 1

traindf['Cabin'] = traindf['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())

testdf['FamilySize'] = testdf['SibSp'] + testdf['Parch'] + 1

testdf['Cabin'] = testdf['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
print('Check missing value on train dataset')

print('-'*120)

traindf.isnull().sum()



print('Check missing value on test dataset')

print('-'*120)

testdf.isnull().sum()
print('Check order of Cabin based on number of survived and unsurvived')

print('-'*120)

a = traindf[['PassengerId','Cabin','Survived']].groupby('Cabin', as_index=False).agg({

    'PassengerId': 'count',

    'Survived': 'sum'

})



a['CabinGroup'] = a.Cabin.str[0]

a = a[['PassengerId','Survived','CabinGroup']].groupby('CabinGroup').agg({

    'PassengerId' : 'sum',

    'Survived': 'sum'

})



a['Unsurvived'] = a['PassengerId'] - a['Survived']

a['Survived_perc'] = (a['Survived']/a['PassengerId'])*100

a['Unsurvived_perc'] = (a['Unsurvived']/a['PassengerId'])*100



a.head(10)
print('Check order of Title based on number of survived and unsurvived')

print('-'*120)



copy_df = traindf.copy()

copy_df['Title'] = copy_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



b = copy_df[['PassengerId','Title','Survived']].groupby('Title', as_index=False).agg({

    'PassengerId': 'count',

    'Survived': 'sum'

})





b['Unsurvived'] = b['PassengerId'] - b['Survived']

b['Survived_perc'] = (b['Survived']/b['PassengerId'])*100

b['Unsurvived_perc'] = (b['Unsurvived']/b['PassengerId'])*100



b.head(17)
print('Change Title value based on number of percentage survived and unsurvived')

print('-'*120)



unsurvived_title_order = {'Capt':8, 'Col':5, 'Don':8, 'Dr':6, 'Jonkheer':8, 'Lady':1, 'Major':5, 'Master':4,

       'Miss':3, 'Mlle':1, 'Mme':1, 'Mr':7, 'Mrs':2, 'Ms':1, 'Rev':8, 'Sir':1,

       'the Countess':1}

survived_title_order = {'Capt':1, 'Col':4, 'Don':1, 'Dr':3, 'Jonkheer':1, 'Lady':8, 'Major':4, 'Master':5,

       'Miss':6, 'Mlle':8, 'Mme':8, 'Mr':2, 'Mrs':7, 'Ms':8, 'Rev':1, 'Sir':8,

       'the Countess':8}

traindf['Title'] = traindf['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

testdf['Title'] = testdf['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

traindf['TitleSurvivedOrder'] = traindf['Title'].map(survived_title_order)

traindf['TitleUnsurvivedOrder'] = traindf['Title'].map(unsurvived_title_order)

testdf['TitleSurvivedOrder'] = testdf['Title'].map(survived_title_order)

testdf['TitleUnsurvivedOrder'] = testdf['Title'].map(unsurvived_title_order)
testdf['TitleSurvivedOrder'].fillna(0, inplace=True)

testdf['TitleUnsurvivedOrder'].fillna(0, inplace=True)
print('Change cabin value based on number of percentage survived and unsurvived')

print('-'*120)



#Survived : DEBFCGAUT

#Unsurvived: TUAGCFBED

unsurvived_cabin_order = {'A':7,'B':3,'C':5,'D':1,'E':2,'F':4,'G':6,'T':9,'U':8}

survived_cabin_order = {'A':3,'B':7,'C':5,'D':9,'E':8,'F':6,'G':4,'T':1,'U':2}



traindf['CabinSurvivedOrder'] = traindf['Cabin'].map(survived_cabin_order)

traindf['CabinUnsurvivedOrder'] = traindf['Cabin'].map(unsurvived_cabin_order)

testdf['CabinSurvivedOrder'] = testdf['Cabin'].map(survived_cabin_order)

testdf['CabinUnsurvivedOrder'] = testdf['Cabin'].map(unsurvived_cabin_order)
print('Transform and map sex feature on train dataset & test dataset')

print('-'*120)

sex_le = LabelEncoder()

sex_le_2 = LabelEncoder()

sex_labels = sex_le.fit_transform(traindf['Sex'])

sex_labels_2 = sex_le_2.fit_transform(testdf['Sex'])

traindf['Sex_Label'] = sex_labels

testdf['Sex_Label'] = sex_labels_2



print('Transform and map embarked feature on train dataset & test dataset')

print('-'*120)

embk_le = LabelEncoder()

embk_le_2 = LabelEncoder()

embk_labels = embk_le.fit_transform(traindf['Embarked'])

embk_labels_2 = embk_le_2.fit_transform(testdf['Embarked'])

traindf['Embk_Label'] = embk_labels

testdf['Embk_Label'] = embk_labels_2
# encode sex labels using one-hot encoding scheme

print('Encode sex labels using one-hot encoding scheme for train & test dataset')

print('-'*120)

sex_ohe = OneHotEncoder()

sex_ohe_2 = OneHotEncoder()

sex_feature_arr = sex_ohe.fit_transform(

                              traindf[['Sex_Label']]).toarray()

sex_feature_arr_2 = sex_ohe_2.fit_transform(

                              testdf[['Sex_Label']]).toarray()



sex_feature_labels = list(sex_le.classes_)

sex_features = pd.DataFrame(sex_feature_arr, 

                            columns=sex_feature_labels)

sex_features_2 = pd.DataFrame(sex_feature_arr_2, 

                            columns=sex_feature_labels)



print('Encode embarked labels using one-hot encoding scheme for train & test dataset')

print('-'*120)

embk_ohe = OneHotEncoder()

embk_ohe_2 = OneHotEncoder()

embk_feature_arr = embk_ohe.fit_transform(

                                traindf[['Embk_Label']]).toarray()

embk_feature_arr_2 = embk_ohe_2.fit_transform(

                                testdf[['Embk_Label']]).toarray()



embk_feature_labels = list(embk_le.classes_)

embk_feature_labels_2 = list(embk_le_2.classes_)



embk_features = pd.DataFrame(embk_feature_arr, 

                            columns=embk_feature_labels)

embk_features_2 = pd.DataFrame(embk_feature_arr_2, 

                            columns=embk_feature_labels_2)

print('Encode pclass labels using one-hot encoding scheme for train & test dataset')

print('-'*120)

status_ohe = OneHotEncoder()

status_ohe_2 = OneHotEncoder()

status_feature_arr = status_ohe.fit_transform(

                                traindf[['Pclass']]).toarray()

status_feature_arr_2 = status_ohe_2.fit_transform(

                                testdf[['Pclass']]).toarray()

status_feature_labels = ['Upper', 'Middle', 'Lower']

status_features = pd.DataFrame(status_feature_arr, 

                            columns=status_feature_labels)

status_features_2 = pd.DataFrame(status_feature_arr_2, 

                            columns=status_feature_labels)



traindf = pd.concat([traindf, sex_features, status_features, embk_features], axis=1)



testdf = pd.concat([testdf, sex_features_2, status_features_2, embk_features_2], axis=1)
print('Remove Unneccessary on train dataset')

print('-'*120)

traindf.drop(['Name','Ticket','PassengerId','Sex','SibSp','Parch','Embarked',

             'Sex_Label','Embk_Label','Cabin','Title'],axis=1,inplace=True)



print('Remove Unneccessary on test dataset')

print('-'*120)

testdf.drop(['Name','Ticket','Sex','SibSp','Parch','Embarked',

             'Sex_Label','Embk_Label','Cabin','Title'],axis=1,inplace=True)
traindf.corr()
cols=[c for c in traindf.columns if c not in ['Survived', 'PassengerId']]

y = traindf["Survived"]

x = traindf[cols]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
rb = RobustScaler(with_centering=True,copy=False)

x_train = rb.fit_transform(x_train)

x_test = rb.fit_transform(x_test)

transform_2 = rb.fit_transform(testdf[cols])

testdf[cols] = transform_2
params = {

    'num_iterations': 1000000,

    'learning_rate': 0.01,

    'early_stopping_rounds':2500,

    'objective': 'binary',

    'metric': 'binary_logloss',

    'boosting' : 'gbdt',

    'is_unbalance': True,

    'max_depth': 10

}
train_data = lgb.Dataset(x_train, label=y_train)

valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)



lgbmodel = lgb.train(params, train_data,                     

                 valid_sets=[valid_data],

                 valid_names=['valid'],

                 verbose_eval=1000)

y_pred = lgbmodel.predict(x_test)

print("Accuracy on test data:",metrics.accuracy_score(y_test, y_pred.round(0).astype(int))*100)

print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred.round(0).astype(int))*100))
regr = RandomForestClassifier(max_depth=5, random_state=42)



scores = cross_val_score(regr, x_train, y_train, cv=5)

print("Accuracy on train data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



regr.fit(x_train, y_train.ravel())

#Predict the response for test dataset

y_pred = regr.predict(x_test)



print("Accuracy on test data:",metrics.accuracy_score(y_test.ravel(), y_pred)*100)

print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred)*100))
dtclf = DecisionTreeClassifier(max_depth = None, criterion='gini')



scores = cross_val_score(dtclf, x_train, y_train, cv=5)

print("Accuracy on train data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



dtclf.fit(x_train, y_train.ravel())



#Predict the response for test dataset

y_pred = dtclf.predict(x_test)



print("Accuracy on test data:",metrics.accuracy_score(y_test.ravel(), y_pred)*100)

print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred)*100))
logreg = LogisticRegression(max_iter=10000, C=5)



scores = cross_val_score(logreg, x_train, y_train, cv=10)

print("Accuracy on train data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

logreg.fit(x_train, y_train.ravel())



y_pred = logreg.predict(x_test)

print("Accuracy on test data:",metrics.accuracy_score(y_test.ravel(), y_pred)*100)

print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred)*100))
y_pred_1 = logreg.predict(testdf[cols].to_numpy())

y_pred_2 = dtclf.predict(testdf[cols].to_numpy())

y_pred_3 = regr.predict(testdf[cols].to_numpy())

y_pred_4 = lgbmodel.predict(testdf[cols].to_numpy()).round(0).astype(int)



testdf['Survived'] = y_pred_1

testdf[['PassengerId','Survived']].to_csv('TitanicSubmission1.csv', index=False)



testdf['Survived'] = y_pred_2

testdf[['PassengerId','Survived']].to_csv('TitanicSubmission2.csv', index=False)



testdf['Survived'] = y_pred_3

testdf[['PassengerId','Survived']].to_csv('TitanicSubmission3.csv', index=False)



testdf['Survived'] = y_pred_4

testdf[['PassengerId','Survived']].to_csv('TitanicSubmission4.csv', index=False)