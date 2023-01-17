import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn import ensemble

import lightgbm as lgb

from xgboost import XGBClassifier



import eli5

from eli5.sklearn import PermutationImportance
train = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv.zip')

test = pd.read_csv('../input/forest-cover-type-kernels-only/test.csv.zip')

train.head()
# train.info()
# train.describe()
print(train.shape)

print(test.shape)
train.columns
train.isna().sum().sum()
# train.isinf().sum().sum()
# sns.countplot
sns.countplot(x='Cover_Type', data=train, palette='Paired')

plt.show()
# plt.figure(figsize=(8, 6))

# df_corr = train.drop('Id', axis=1).corr()

# sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu', vmax=0.8, vmin=-0.8)

# plt.show()
train['hf1'] = abs(train['Horizontal_Distance_To_Hydrology']+

                   train['Horizontal_Distance_To_Fire_Points'])

train['hf2'] = abs(train['Horizontal_Distance_To_Hydrology']-

                   train['Horizontal_Distance_To_Fire_Points'])

train['hr1'] = abs(train['Horizontal_Distance_To_Hydrology']+

                   train['Horizontal_Distance_To_Roadways'])

train['hr2'] = abs(train['Horizontal_Distance_To_Hydrology']-

                   train['Horizontal_Distance_To_Roadways'])

train['fr1'] = abs(train['Horizontal_Distance_To_Fire_Points']+

                   train['Horizontal_Distance_To_Roadways'])

train['fr2'] = abs(train['Horizontal_Distance_To_Fire_Points']-

                   train['Horizontal_Distance_To_Roadways'])



train['ele_vert'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']



train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+

                      train['Vertical_Distance_To_Hydrology']**2)**0.5



train['slope_hyd'] = train['slope_hyd'].map(lambda x: 0 if np.isinf(x) else x)



train['Mean_Amenities'] = (train['Horizontal_Distance_To_Fire_Points'] + 

                           train['Horizontal_Distance_To_Hydrology'] + 

                           train['Horizontal_Distance_To_Roadways']) / 3 



train['Mean_Fire_Hyd'] = (train['Horizontal_Distance_To_Fire_Points'] + 

                          train['Horizontal_Distance_To_Hydrology']) / 2 
test['hf1'] = abs(test['Horizontal_Distance_To_Hydrology']+

                   test['Horizontal_Distance_To_Fire_Points'])

test['hf2'] = abs(test['Horizontal_Distance_To_Hydrology']-

                   test['Horizontal_Distance_To_Fire_Points'])

test['hr1'] = abs(test['Horizontal_Distance_To_Hydrology']+

                   test['Horizontal_Distance_To_Roadways'])

test['hr2'] = abs(test['Horizontal_Distance_To_Hydrology']-

                   test['Horizontal_Distance_To_Roadways'])

test['fr1'] = abs(test['Horizontal_Distance_To_Fire_Points']+

                   test['Horizontal_Distance_To_Roadways'])

test['fr2'] = abs(test['Horizontal_Distance_To_Fire_Points']-

                   test['Horizontal_Distance_To_Roadways'])



test['ele_vert'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']



test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+

                      test['Vertical_Distance_To_Hydrology']**2)**0.5



test['slope_hyd'] = test['slope_hyd'].map(lambda x: 0 if np.isinf(x) else x)



test['Mean_Amenities'] = (test['Horizontal_Distance_To_Fire_Points'] + 

                           test['Horizontal_Distance_To_Hydrology'] + 

                           test['Horizontal_Distance_To_Roadways']) / 3 



test['Mean_Fire_Hyd'] = (test['Horizontal_Distance_To_Fire_Points'] + 

                          test['Horizontal_Distance_To_Hydrology']) / 2 
test.columns
feature = [col for col in train.columns if col not in ['Cover_Type','Id']]

X_train = train[feature]

X_test = test[feature]
preds = pd.DataFrame()
m1 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=500), 

                                 n_estimators=250, learning_rate=0.01, algorithm='SAMME')  

m1.fit(X_train, train['Cover_Type']) 

preds["Model1"] = m1.predict(X_test)
m2 = ensemble.ExtraTreesClassifier(n_estimators=550)  

m2.fit(X_train, train['Cover_Type'])

preds["Model2"] = m2.predict(X_test)
m3 = XGBClassifier(max_depth=20, n_estimators=1000)  

m3.fit(X_train, train['Cover_Type'])

preds["Model3"] = m3.predict(X_test)
# m4 = LGBMClassifier(n_estimators=2000, max_depth=15)

# m4.fit(X_train, train['Cover_Type'])

# preds["Model4"] = m4.predict(X_test)
m5 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=10), n_estimators=1000, learning_rate=0.01, algorithm="SAMME")

m5.fit(X_train, train['Cover_Type'])

preds["Model5"] = m5.predict(X_test)
m6 = SGDClassifier(loss='hinge')

m6.fit(X_train, train['Cover_Type'])

preds["Model6"] = m6.predict(X_test)
preds.head()
pred = preds.mode(axis=1)

pred
sub = pd.DataFrame({"Id": test['Id'],

                    "Cover_Type": pred[0].astype('int').values})

sub.to_csv("submission.csv", index=False)