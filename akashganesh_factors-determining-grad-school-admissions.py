import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
df=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df= df.drop('Serial No.',axis=1)
df.head()
df.isnull().sum()
df.head()
df['GRE Score']=pd.to_numeric(df['GRE Score'])
df['TOEFL Score']=pd.to_numeric(df['TOEFL Score'])
df['University Rating']=pd.to_numeric(df['University Rating'])
df['SOP']=pd.to_numeric(df['SOP'])

df['CGPA']=pd.to_numeric(df['CGPA'])
df['Research']=pd.to_numeric(df['Research'])

df['GRE Score'].describe()

fig=sns.distplot(df['GRE Score'])
plt.show()
df['TOEFL Score'].describe()
fig=sns.distplot(df['TOEFL Score'])
plt.show()
df['CGPA'].describe()
fig=sns.distplot(df['CGPA'])
plt.show()
fig=sns.distplot(df['SOP'])
plt.show()
df['University Rating'].describe()
fig=sns.distplot(df['University Rating'])
plt.show()
fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)
plt.title("GRE Score vs TOEFL Score")
plt.show()
fig = sns.regplot(x="GRE Score", y="CGPA", data=df)
plt.title("GRE Score vs CGPA")
plt.show()

fig = sns.regplot(x="GRE Score", y="Chance of Admit ", data=df)
plt.title("GRE Score vs Chance of Admit")
plt.show()

fig = sns.regplot(x="GRE Score", y="LOR ", data=df)
plt.title("GRE Score vs LOR")
plt.show()
fig = sns.regplot(x="CGPA", y="SOP", data=df)
plt.title("CGPA vs SOP")
plt.show()
import numpy as np
corr=df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
from sklearn.model_selection import train_test_split

X = df.drop(['Chance of Admit '], axis=1)
y= df['Chance of Admit ']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,shuffle=False)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

models=[["DecisionTree:",DecisionTreeRegressor()],
        ["RandomForest:",RandomForestRegressor()],
        ["KNeighbors:",KNeighborsRegressor(n_neighbors=2)],
        ["LinearReg:",LinearRegression()],
          ]

for name,model in models:
    model=model
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
classifier = LinearRegression()
classifier.fit(X,y)
feature_names = X.columns
importance_frame = pd.DataFrame()
importance_frame['Features'] = X.columns
importance_frame['Importance'] = classifier.coef_
importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)
plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)
plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()
classifier = DecisionTreeRegressor()
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