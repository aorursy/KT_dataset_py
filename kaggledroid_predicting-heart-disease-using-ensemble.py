import pandas as pd
X_path = '../input/heart-disease-uci/heart.csv'

X = pd.read_csv(X_path)

X.index.name = 'Id'
X.shape
X.head()
X.isnull().any()
import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(8,6))

sns.boxplot(data=X.drop(['target'], axis=1))
fig = plt.figure()

fig.set_size_inches(16,12)

fig.subplots_adjust(hspace=0.5, wspace=0.5)

list_of_y = [i for i in X.columns]

for i in range(1,13):

    ax = fig.add_subplot(7,2,i)

    sns.swarmplot(x='target', y=list_of_y[i-1], data=X, ax=ax)

plt.show()

plt.figure(figsize=(16,2))

sns.swarmplot(x='target', y='thal', data=X)
plt.figure(figsize=(12,8))

sns.heatmap(X.corr())
g = sns.pairplot(X, kind='reg')

g.fig.set_size_inches(20,20)
sns.jointplot(x=X['age'], y=X['chol'], kind='hex', color='#4CB391')
fig = plt.figure()

fig.set_size_inches(16,4)

fig.subplots_adjust(hspace=0.5, wspace=0.5)



ax = fig.add_subplot(1,3,1)

sns.violinplot(x='sex', y='chol', data=X)



ax = fig.add_subplot(1,3,2)

sns.violinplot(x='ca', y='age', data=X)



ax = fig.add_subplot(1,3,3)

sns.violinplot(x='fbs', y='age', data=X)
#Separating target from features

y = X.target

X.drop(['target'], axis=1, inplace=True)



from sklearn.model_selection import train_test_split



#Splitting the data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



#Defining the models

xgb_model = XGBClassifier(n_estimator=1000)

lgb_model = LGBMClassifier(n_estimators=1000, learning_rate=0.01)

rf_model = RandomForestClassifier(n_estimators=1000)

mlp_model = MLPClassifier(hidden_layer_sizes=(80,80,80), max_iter=10000)
#Training the models

xgb_model.fit(X_train, y_train)

lgb_model.fit(X_train, y_train)

rf_model.fit(X_train, y_train)

mlp_model.fit(X_train, y_train)



#Printing the scores of the models

print("XGBoost Score     : ",xgb_model.score(X_valid, y_valid))

print("LightGBM Score    : ",lgb_model.score(X_valid, y_valid))

print("RandomForest Score: ",rf_model.score(X_valid, y_valid))

print("MLP Score         : ",mlp_model.score(X_valid, y_valid))
from sklearn.ensemble import StackingClassifier



estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model), ('mlp', mlp_model)]



clf = StackingClassifier(estimators=estimators)

clf.fit(X_train, y_train)
print("Ensemble Score: ",clf.score(X_valid, y_valid))