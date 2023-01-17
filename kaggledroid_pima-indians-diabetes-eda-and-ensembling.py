import pandas as pd 

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

%matplotlib inline

import seaborn as sns
#Loading the dataset

X_path = '../input/pima-indians-diabetes-database/diabetes.csv'

X = pd.read_csv(X_path)

X.index.name = 'Id'
X.shape
X.head()
#Looking for missing values

X.isnull().any()
#Separating the target from the features

y = X.Outcome



#Creating a copy of the dataset for data visualization

X_copy = X.copy()



#Dropping the target from the original dataset

X.drop(['Outcome'], axis=1, inplace=True)
plt.figure(figsize=(14,6))

sns.boxplot(data=X_copy.drop(['Outcome'], axis=1))
fig = plt.figure()

fig.set_size_inches(14, 10)

fig.subplots_adjust(hspace=0.6, wspace=0.6)

list_of_y = [i for i in X.columns]

for i in range(1,9):

    ax = fig.add_subplot(2, 4, i)

    sns.swarmplot(x='Outcome', y=list_of_y[i-1], data=X_copy, ax=ax)

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(X_copy.corr())
g = sns.pairplot(X_copy.drop(['Outcome'], axis=1), kind='reg')

g.fig.set_size_inches(20,20)
sns.jointplot(x='Glucose', y='BMI', data=X_copy, kind='kde')
#Splitting the data into training and validation data

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
#Defining the models

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



xgb_model = XGBClassifier()

lgb_model = LGBMClassifier()

rf_model = RandomForestClassifier()

ada_model = AdaBoostClassifier()
#Training the models

xgb_model.fit(X_train, y_train)

lgb_model.fit(X_train, y_train)

rf_model.fit(X_train, y_train)

ada_model.fit(X_train, y_train)



#Getting the scores of the models

print("XGBoost Score           : ",xgb_model.score(X_valid, y_valid))

print("LightGBM Score          : ",lgb_model.score(X_valid, y_valid))

print("RandomForests Score     : ",rf_model.score(X_valid, y_valid))

print("AdaBoost Score          : ",ada_model.score(X_valid, y_valid))
from sklearn.ensemble import StackingClassifier

from sklearn.linear_model import LogisticRegression



estimators = [('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model), ('ada', ada_model)]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)