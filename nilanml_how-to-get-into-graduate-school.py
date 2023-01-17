import pandas as pd

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('../input/Admission_Predict.csv')

df.head()
df = df.drop(['Serial No.'], axis=1)

df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns





fig = sns.distplot(df['GRE Score'], kde=False)

plt.title("Distribution of GRE Scores")

plt.show()



fig = sns.distplot(df['TOEFL Score'], kde=False)

plt.title("Distribution of TOEFL Scores")

plt.show()



fig = sns.distplot(df['University Rating'], kde=False)

plt.title("Distribution of University Rating")

plt.show()



fig = sns.distplot(df['SOP'], kde=False)

plt.title("Distribution of SOP Ratings")

plt.show()



fig = sns.distplot(df['CGPA'], kde=False)

plt.title("Distribution of CGPA")

plt.show()



plt.show()
fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)

plt.title("GRE Score vs TOEFL Score")

plt.show()
fig = sns.regplot(x="GRE Score", y="CGPA", data=df)

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.lmplot(x="CGPA", y="LOR ", data=df, hue="Research")

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.lmplot(x="GRE Score", y="LOR ", data=df, hue="Research")

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.regplot(x="CGPA", y="SOP", data=df)

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.regplot(x="GRE Score", y="SOP", data=df)

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.regplot(x="TOEFL Score", y="SOP", data=df)

plt.title("GRE Score vs CGPA")

plt.show()
import numpy as np

corr = df.corr()

fig, ax = plt.subplots(figsize=(8, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.show()
from sklearn.model_selection import train_test_split



X = df.drop(['Chance of Admit '], axis=1)

y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error



models = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM :', SVR()],

           ['AdaBoostClassifier :', AdaBoostRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

           ['Xgboost: ', XGBRegressor()],

           ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()],

           ['HuberRegressor: ', HuberRegressor()]]



print("Results...")





for name,model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
classifier = RandomForestRegressor()

classifier.fit(X,y)

feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()