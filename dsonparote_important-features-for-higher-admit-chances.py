import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model 

from sklearn.svm import SVR

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest,f_regression





%matplotlib inline
add_df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv") 

add_df.head(10)
add_df.info()
add_df.describe()
# removing extra index column

add_df.drop(columns=['Serial No.'],inplace=True)
# renaming columns forbetter clarity

add_df.rename(columns={'Chance of Admit ':'Chance of Admit','LOR ':'LOR'},inplace=True)

add_df.columns
add_df.hist(bins=10,figsize=(10,8))
plt.figure(figsize = (16,10))

sns.heatmap(add_df.corr(),annot=True,center=0, cmap="YlGnBu")
add_df.columns
sns.pairplot(add_df, x_vars=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',

       'Research'],y_vars=['Chance of Admit'],height=5, aspect=0.7)
add_df[['Research','Chance of Admit']].groupby(by='Research').mean().sort_values(by='Chance of Admit',ascending=False).plot(kind='bar')

add_df[['Research','Chance of Admit']].groupby(by='Research').mean().sort_values(by='Chance of Admit',ascending=False)
add_df[['University Rating','Chance of Admit']].groupby(by='University Rating').mean().sort_values(by='Chance of Admit',ascending=False).plot(kind='bar')

add_df[['University Rating','Chance of Admit']].groupby(by='University Rating').mean().sort_values(by='Chance of Admit',ascending=False)
add_df[['SOP','Chance of Admit']].groupby(by='SOP').mean().sort_values(by='Chance of Admit',ascending=False).plot(kind='bar')

add_df[['SOP','Chance of Admit']].groupby(by='SOP').mean().sort_values(by='Chance of Admit',ascending=False)
add_df[['LOR','Chance of Admit']].groupby(by='LOR').mean().sort_values(by='Chance of Admit',ascending=False).plot(kind='bar')

add_df[['LOR','Chance of Admit']].groupby(by='LOR').mean().sort_values(by='Chance of Admit',ascending=False)
#scaling the data

add_df_norm = add_df.copy()

scaler = MinMaxScaler()

column_names_to_normalize = ['CGPA', 'GRE Score', 'LOR', 'SOP','TOEFL Score']

x = add_df_norm[column_names_to_normalize].values

x_scaled = scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = add_df_norm.index)

add_df_norm[column_names_to_normalize] = df_temp
add_df_norm.hist(bins=10,figsize=(10,8))
add_df1 = add_df_norm.copy()

y = add_df1['Chance of Admit']

add_df1.drop(columns=['Chance of Admit'],inplace=True)
X_train, X_test, y_train, y_test = train_test_split(add_df1, y, test_size=0.2,random_state=1)
model1 = Pipeline([('scaler',MinMaxScaler()),('clf',linear_model.LinearRegression())])



model1.fit(X_train, y_train)

pred = model1.predict(X_test)



MSE = metrics.mean_squared_error(y_test, pred)

RMSE = np.sqrt(MSE)

Rsq = metrics.r2_score(y_test, pred)



print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))

coef_df1 = pd.DataFrame(zip(X_train.columns,model1.named_steps['clf'].coef_),columns=['Feature','Coefficient'])

coef_df1.sort_values(by='Coefficient',inplace=True, ascending=False)



plt.figure(figsize = (10,6))

sns.barplot(x='Feature',y='Coefficient',data=coef_df1)

coef_df1
#model without SOP considering score

add_df2 = add_df_norm.copy()

y = add_df2['Chance of Admit']

add_df2.drop(columns=['Chance of Admit','SOP'],inplace=True)



X_train1, X_test1, y_train1, y_test1 = train_test_split(add_df2, y, test_size=0.2,random_state=1)





# we can use the same model1 from above



model1.fit(X_train1, y_train1)

pred = model1.predict(X_test1)



MSE = metrics.mean_squared_error(y_test1, pred)

RMSE = np.sqrt(MSE)

Rsq = metrics.r2_score(y_test1, pred)



print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))
model2 = Pipeline([('scaler',MinMaxScaler()),('clf',SVR(kernel='linear'))])



model2.fit(X_train1, y_train1)

pred = model2.predict(X_test1)



MSE = metrics.mean_squared_error(y_test1, pred)

RMSE = np.sqrt(MSE)

Rsq = metrics.r2_score(y_test1, pred)



print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))

coef_df2 = pd.DataFrame(zip(X_train.columns,model2.named_steps['clf'].coef_[0]),columns=['Feature','Coefficient'])

coef_df2.sort_values(by='Coefficient',inplace=True, ascending=False)



plt.figure(figsize = (10,6))

sns.barplot(x='Feature',y='Coefficient',data=coef_df2)

coef_df2
model3 = make_pipeline(DecisionTreeRegressor())



model3.fit(X_train1, y_train1)

pred = model4.predict(X_test1)



MSE = metrics.mean_squared_error(y_test1, pred)

RMSE = np.sqrt(metrics.mean_squared_error(y_test1, pred))



Rsq = metrics.r2_score(y_test1, pred)



print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))

model4 = make_pipeline(RandomForestRegressor(random_state=1,n_estimators= 200))



model4.fit(X_train1, y_train1)

pred = model4.predict(X_test1)



MSE = metrics.mean_squared_error(y_test1, pred)

RMSE = np.sqrt(metrics.mean_squared_error(y_test1, pred))



Rsq = metrics.r2_score(y_test1, pred)



print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))

model5 = make_pipeline(ExtraTreesRegressor())



model5.fit(X_train, y_train)

pred = model5.predict(X_test)



MSE = metrics.mean_squared_error(y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))

Rsq = metrics.r2_score(y_test, pred)





print("MSE:{}  RMSE:{}  R-Squared:{}".format(MSE,RMSE,Rsq))
