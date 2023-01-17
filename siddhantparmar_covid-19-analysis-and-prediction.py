import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
df1=pd.read_csv("../input/coviddata/owid-covid-data.csv")
df1.head()
df1.info()
print("There are",(df1['location']=='India').sum(),"entries of India in the location column in the dataset.")
print("There are",df1['location'].isnull().sum(),"null values in the location column.")
print(((df1['location']=='India').sum())/(df1['location'].notnull().sum())*100,"% of rows contain India in the location column.")
df=df1.copy()
df.drop(df[df['location']!='India'].index,inplace=True)
df.head()
ProfileReport(df)
var_num=df.select_dtypes(exclude=['object']).columns.tolist()
var_num
dict1={}
dict2={}
list1=[]
for i in var_num:
    dict2['mean']=df[i].mean()
    dict2['median']=df[i].median()
    dict2['mode']=df[i].value_counts().index[0]
    dict2['min']=df[i].min()
    dict2['max']=df[i].max()
    range=(df[i].max())-(df[i].min())
    if range!=0:
        list1.append(i)
    dict1[i]=dict2
    dict2={}
import json
json_object = json.dumps(dict1)
df2=pd.read_json(json_object)
df2.head()
print("The list of numeric features without constant value is :\n",list1)
for z in list1:
    print(z)
    plt.figure(z)
    plt.hist(df[z],bins=10)
    plt.show()
for i in list1:
    for j in list1:
        if i!=j:
            plt.figure(i)
            sns.scatterplot(df[i],df[j])
for i in list1:
    for j in list1:
        if i!=j:
            plt.figure(i)
            sns.lineplot(df[i],df[j])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
feature_with_na=[feature for feature in df.columns if df[feature].isnull().sum()>=1]
feature_with_na
for i in feature_with_na:
    print(i,":",np.round(df[i].isnull().mean(),4)*100,"% missing values")
numerical_with_na=[]
for j in feature_with_na:
    if df[j].dtypes!='O':
        numerical_with_na.append(j)
numerical_with_na
for i in numerical_with_na:
    df[i]=df[i].fillna(df[i].mean())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
categorical_with_na=[]
for j in feature_with_na:
    if df[j].dtypes=='O':
        categorical_with_na.append(j)
print("Categorical features with null values are :\n",categorical_with_na)
for i in categorical_with_na:
    df[i]=df[i].fillna(df[i].value_counts().index[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
import datetime as dt
df['date']=pd.to_datetime(df["date"]) 
df["date"]=df["date"].map(dt.datetime.toordinal)
df.head()
var_cat=df.select_dtypes(include=['object']).columns.tolist()
var_cat
df3=df.drop(var_cat,axis=1)
df3.head()
y=df3['total_cases'].values
df4=df3.drop('total_cases',axis=1)
X=df4.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500,random_state=0)
model.fit(X_train, y_train) 
y_pred_rf=model.predict(X_test)
print("Accuracy for Linear Regression :\n")
from sklearn.metrics import classification_report,accuracy_score
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#visualize comparison result as a bar graph
df5 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
dfpred = df5.head(50)
dfpred.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print("Accuracy for Random Forest Regression :\n")
from sklearn.metrics import classification_report,accuracy_score
from sklearn import metrics
print('Root Mean Squared Error:', metrics.r2_score(y_test,y_pred_rf))
#visualize comparison
df6 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_rf.flatten()})
dfpred = df6.head(50)
dfpred.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()