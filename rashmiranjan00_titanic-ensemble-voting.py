import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix , classification_report

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV







# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
train_data.info()
train_data.describe()
#Let's check that the target is indeed 0 or 1:



train_data["Survived"].value_counts()
# Now let's take a quik look at all the catagorical values



train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]

    
#removing mentioned columns from dataset

train_data = train_data.drop(['Name','Ticket','Cabin','SibSp','Parch','PassengerId'],axis=1)

test_data = test_data.drop(['Name','Ticket','Cabin','SibSp','Parch'],axis=1)
# Let's build the pipeline for the numerical attributes:



# num_pipeline = Pipeline([

#     ("select_numeric", DataFrameSelector(["Age","Fare"])),

#     ("imputer", SimpleImputer(strategy="median"))

# ])
# num_pipeline.fit_transform(train_data)
class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
# cat_pipeline = Pipeline([

#     ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

#     ("imputer", MostFrequentImputer()),

#     ("cat_encoder", OneHotEncoder(sparse=False))

# ])
# cat_pipeline.fit_transform(train_data)
# Finally, let's join the numerical and categorical pipelines:



# preprocess_pipeline = FeatureUnion(transformer_list=[

#     ("num_pipeline", num_pipeline),

#     ("cat_pipeline", cat_pipeline)

# ])
#checking for any null values

train_data.isnull().any() # True means null value present
test_data.isnull().any()
# age columns

print('mean age in train data :',train_data['Age'].mean())

print('mean age in test data :',test_data['Age'].mean())
combined_data=[train_data,test_data]

#replacing null values with 30 in age column

for df in combined_data:

    df['Age'] = df['Age'].replace(np.nan,30).astype(int)
train_data['Embarked'].value_counts()
#most people embarked from 'S'. So, we'll replace the missing missing Embarked value by 'S'.

train_data['Embarked'] = train_data['Embarked'].replace(np.nan,'S')
#finding mean fare in test data

test_data['Fare'].mean()
#replace missing fare values in test data by mean

test_data['Fare'] = test_data['Fare'].replace(np.nan,36).astype(int)
combined_data=[train_data,test_data]

for df in combined_data:

    print(df.isnull().any()) #bool value = False means that there are no nulls in the column.
train_data.head()
train_onehot = train_data.copy()
train_onehot = pd.get_dummies(train_onehot, columns=['Embarked', 'Sex'], 

                              prefix=['Embarked', 'Sex'])
train_onehot = train_onehot.astype(int)

train_onehot.head()
test_data.head()
test_onehot = test_data.copy()
test_onehot = pd.get_dummies(test_onehot, columns=['Embarked', 'Sex'], 

                              prefix=['Embarked', 'Sex'])
test_onehot = test_onehot.astype(int)

train_onehot.head()
print(len(train_onehot.columns))

print(len(train_onehot))
print(len(test_onehot.columns))

print(len(test_onehot))
X_train, X_test, y_train, y_test = train_test_split(train_onehot.drop(['Survived'],axis=1), train_onehot['Survived'], test_size=0.2, random_state=42)
# X_train = train_onehot.drop(['Survived'], axis = 1)

# X_train.head()
# Let's not forget to get the labels:



# y_train = train_onehot["Survived"]
# X_test = test_onehot.drop(['PassengerId'], axis=1).copy()
xgb_model = xgb.XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=10,

                      eval_metric = 'auc')
xgb_model.fit(X_train,y_train)
preds_xgb = xgb_model.predict(X_test)
print(confusion_matrix(y_test,preds_xgb))

print(classification_report(y_test,preds_xgb))
params = {

 "learning_rate"    : list(np.arange(0.05,0.6,0.05)) ,

 "max_depth"        : list(np.arange(1,20,2)),

 "min_child_weight" : list(np.arange(1,9,1)),

 "gamma"            : list(np.arange(0.05,0.7,0.05)),

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}
grid = RandomizedSearchCV(xgb.XGBClassifier(objective='binary:logistic'),

                         param_distributions=params,

                         scoring='accuracy',cv=5,verbose=5)
grid.fit(X_train,y_train)
grid.best_params_, grid.best_score_
best_matrix = {'min_child_weight': 8,

  'max_depth': 13,

  'learning_rate': 0.5,

  'gamma': 0.15000000000000002,

  'colsample_bytree': 0.7}
pre_finalized_model = xgb.XGBClassifier(params=best_matrix, objective='binary:logistic')
pre_finalized_model.fit(X_train,y_train)
pre_final_preds = pre_finalized_model.predict(X_test)
print(confusion_matrix(y_test,pre_final_preds))

print(classification_report(y_test,pre_final_preds))
X = train_onehot.drop(['Survived'],axis=1)

y = train_onehot['Survived']
model_to_submit_from = xgb.XGBClassifier(params=best_matrix, objective='binary:logistic')
model_to_submit_from.fit(X,y)
final_test = test_onehot.drop(['PassengerId'], axis=1).copy()
final_prediction = model_to_submit_from.predict(final_test)
accu_rafo = model_to_submit_from.score(X, y)

round(accu_rafo*100,2)
svm_clf = SVC(gamma="auto")

svm_clf.fit(X, y)
svm_pred = svm_clf.predict(final_test)
svm_scores = cross_val_score(svm_clf, X, y, cv=10)

svm_scores.mean()
forest_clf = RandomForestClassifier(n_estimators=200, random_state=42)

forest_scores = cross_val_score(forest_clf, X, y, cv=10)

forest_scores.mean()
forest_clf.fit(X, y)
forest_pred = forest_clf.predict(final_test)

accu_rafo = forest_clf.score(X, y)

round(accu_rafo*100,2)
#first applying Logistic Regression



lg = LogisticRegression()

lg.fit(X, y)

lg_pred = lg.predict(final_test)

accu_lg = (lg.score(X, y))

round(accu_lg*100,2)
lg_scores = cross_val_score(lg, X, y, cv=10)

lg_scores.mean()
voting_clf = VotingClassifier(

estimators=[('xgb', model_to_submit_from), ('lr', lg)],

voting='hard',

n_jobs=-1)
voting_clf.fit(X, y)
ensemble_predict = voting_clf.predict(final_test)
plt.figure(figsize=(8, 4))

plt.plot([1]*10, svm_scores, ".")

plt.plot([2]*10, forest_scores, ".")

plt.boxplot([final_prediction, svm_scores, forest_scores, lg_scores], labels=("xgBoost", "SVM","Random Forest", "Logistic Reg"))

plt.ylabel("Accuracy", fontsize=14)

plt.show()
# train_data["AgeBucket"] = train_data["Age"] // 15 * 15

# train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
# train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]

# train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
submission = pd.DataFrame(columns = ['PassengerId', 'Survived']) 

print(submission.head())

submission.PassengerId = test_data.PassengerId

submission.Survived = ensemble_predict

print(len(submission))

print(submission.head())

submission.to_csv('ensemble_titanic_pred.csv', index=False)