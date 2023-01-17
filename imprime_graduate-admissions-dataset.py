import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import LassoCV
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')



df.head()
df.tail()
df = df.drop(['Serial No.'], axis=1)

df.head(n=10)
df.isnull().sum()
df.sample(frac=.01)
df.describe()
df.info()
df.columns
df.columns= df.columns.str.strip().str.replace(" ","_")
df.dtypes
df.shape
df.head()
sns.pairplot(df)
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(), cmap ='magma', square=True)
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(), cmap ='viridis',annot=True, square=True)
# the number of data is performed.

plt.figure(figsize=(12,6))

research_predict=df.groupby('Chance_of_Admit')['TOEFL_Score'].count()

sns.barplot(x=research_predict.index,y=research_predict.values,palette='viridis')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.show()
fig = sns.distplot(df['GRE_Score'], kde=False)

plt.title("Distribution of GRE Scores")



fig = sns.distplot(df['TOEFL_Score'], kde=False)

plt.title("Distribution of TOEFL Scores")





fig = sns.distplot(df['University_Rating'], kde=False)

plt.title("Distribution of University Rating")





fig = sns.distplot(df['SOP'], kde=False)

plt.title("Distribution of SOP Ratings")
fig = sns.distplot(df['CGPA'], kde=False)

plt.title("Distribution of CGPA")
fig = sns.regplot(data=df, x="GRE_Score", y="TOEFL_Score",)

plt.title("GRE Score vs TOEFL Score")
fig = sns.regplot(data=df, x="GRE_Score", y="CGPA")

plt.title("GRE Score vs CGPA")
fig = sns.lmplot(data=df, x="CGPA", y="LOR" , hue="Research",palette='coolwarm')

plt.title("CGPA Score vs CGPA")
fig = sns.lmplot(data=df, x="GRE_Score", y="LOR" , hue="Research",palette='coolwarm')

plt.title("CGPA Score vs CGPA")
fig = sns.countplot(hue="Research", x="LOR", data=df,palette="coolwarm")

plt.title("GRE Score vs LOR")
fig = sns.regplot( data=df, x="CGPA", y="SOP")

plt.title("CGPA vs SOP")

plt.show()
fig = sns.regplot( data=df, x="GRE_Score", y="SOP")

plt.title("GRE Score vs SOP")

plt.show()

fig = sns.regplot(data=df, x="TOEFL_Score", y="SOP")

plt.title("TOEFL Score vs CGPA")

X = df.drop(['Chance_of_Admit'], axis=1)

y = df['Chance_of_Admit']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False,)
print(X_train.shape,  X_test.shape,  y_train.shape,  y_test.shape) 
from sklearn.linear_model import Lasso, Ridge, BayesianRidge, ElasticNet, LinearRegression, LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
models = [['Linear Regression :', LinearRegression()],

           ['Lasso: ', LassoCV(cv=5, random_state=0)],

           ['Ridge: ', RidgeCV(alphas=[0.01, 0.1, 0.001, 1])],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()]]
print("Results...", '\n\n')

for name, model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

   

    print(name)

    print('MSE', mean_squared_error(y_test, predictions))

    print('MAE', mean_absolute_error(y_test, predictions))

    print("RMSE", np.sqrt(mean_squared_error(y_test, predictions)))

    print("Linear Score", model.score(X_test,y_test))

    print("R Squred", r2_score(y_test, predictions))

    print('explained_variance_score', explained_variance_score(y_test, predictions), "\n")