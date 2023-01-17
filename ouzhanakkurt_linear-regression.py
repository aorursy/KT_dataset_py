import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv("../input/sydney-house-prices/SydneyHousePrices.csv")
df.head()
df.info()
df["Date"]=pd.to_datetime(df["Date"])

df["Year"]=df["Date"].dt.year

df["Month"]=df["Date"].dt.month

df["Day"]=df["Date"].dt.day
df=df.drop(["Id","Date"],axis=1)
df.head()
df.describe().T
df["new_column"]=df["sellPrice"]

df.drop(["sellPrice"],axis=1,inplace=True)

df.rename(columns={"new_column":"sellPrice"},inplace=True)
list_name=[]

list_type=[]

list_total_value=[]

list_missing_value=[]

list_unique_value=[]



for i in df.columns:

    list_name.append(i)

    list_type.append(str(df[i].dtype))

    list_total_value.append(df[i].notnull().sum())

    list_missing_value.append(df[i].isnull().sum())

    list_unique_value.append(len(df[i].unique()))



    df_info=pd.DataFrame(data={"Total_Value":list_total_value,"Missing_Value":list_missing_value,"Unique_Value":list_unique_value,"Type":list_type},index=list_name)
df_info
sns.set_style("whitegrid")
plt.figure(figsize=(15,6))

df["suburb"].value_counts()[:15].plot.barh()
plt.figure(figsize=(15,6))

df["propType"].value_counts().plot.barh()
data_num=df.select_dtypes(["float64","int64"]).columns
fig,ax=plt.subplots(nrows=4,ncols=2,figsize=(15,15))

count=0

for i in range(4):

    for j in range(2):

        sns.kdeplot(df[data_num[count]],ax=ax[i][j],shade=True,color="#008080")

        count+=1
plt.figure(figsize=(15,6))

sns.countplot(df["propType"],saturation=1,palette="pastel")
plt.figure(figsize=(15,6))

sns.countplot(df["Month"],palette="Set3")
plt.figure(figsize=(15,6))

sns.countplot(df["Year"],palette="hot_r")
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(15,12))

sns.barplot(data=df,x="Year",y="sellPrice",color="#8E44AD",ax=ax[0])

sns.barplot(data=df,x="Month",y="sellPrice",color="#2ECC71",ax=ax[1])
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(15,12))

sns.pointplot(data=df,x="Year",y="sellPrice",color="#8E44AD",ax=ax[0])

sns.pointplot(data=df,x="Month",y="sellPrice",color="#2ECC71",ax=ax[1])
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(15,12))

sns.pointplot(data=df,x="Year",y="sellPrice",hue="propType",ax=ax[0],ci=None)

sns.pointplot(data=df,x="Month",y="sellPrice",hue="propType",ax=ax[1],ci=None)
heat = pd.pivot_table(data = df,

                    index = 'Month',

                    values = 'sellPrice',

                    columns = 'Year')

heat.fillna(0, inplace = True)
heat
plt.figure(figsize=(15,10))

plt.title('Heat map about Years and Mounths means ')

sns.heatmap(heat)
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),xticklabels=df.corr().columns,yticklabels=df.corr().columns,annot=True)
plt.figure(figsize=(15,12))

sns.pairplot(df,vars=["bed","bath","car","sellPrice"])
list_value_counts=(df.groupby("suburb")["sellPrice"].mean().sort_values())[::-1].astype("int")
list_value_counts[:5]
suburb_grup1=list(list_value_counts[:137].index)

suburb_grup2=list(list_value_counts[137:274].index)

suburb_grup3=list(list_value_counts[274:411].index)

suburb_grup4=list(list_value_counts[411:548].index)

suburb_grup5=list(list_value_counts[548:685].index)
df.replace(suburb_grup1,"Group1",inplace=True)

df.replace(suburb_grup2,"Group2",inplace=True)

df.replace(suburb_grup3,"Group3",inplace=True)

df.replace(suburb_grup4,"Group4",inplace=True)

df.replace(suburb_grup5,"Group5",inplace=True)
df.head()
dff=df.copy()
#propType_data=pd.get_dummies(df["propType"],prefix="propType")

#suburp_data=pd.get_dummies(df["suburb"],prefix="suburb")

#df.drop("propType",axis=1,inplace=True)

#df=pd.concat([df,suburp_data,propType_data],axis=1)

df = pd.get_dummies(df,columns= ["suburb","propType"], prefix= ["suburb","propType"])
df.head()
data_num=list(df.select_dtypes(["int64","float64"]).columns)

data_num.remove("Year")

data_num.remove("Day")

data_num.remove("Month")
from warnings import filterwarnings



filterwarnings("ignore")
fig, ax =plt.subplots(nrows=5,ncols=1,figsize=(18,16))

for i in range(5):

    sns.boxplot(x = df[data_num[i]],ax=ax[i])

    count = count+1
lower_and_upper = {}



for col in data_num:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = 1.5*(q3-q1)

    

    lower_bound = q1-iqr

    upper_bound = q3+iqr

    

    lower_and_upper[col] = (lower_bound, upper_bound)

    df.loc[(df.loc[:,col]<lower_bound),col]=lower_bound*0.75

    df.loc[(df.loc[:,col]>upper_bound),col]=upper_bound*1.25

    

    

lower_and_upper
fig, ax =plt.subplots(nrows=5,ncols=1,figsize=(18,16))

for i in range(5):

    sns.boxplot(x = df[data_num[i]],ax=ax[i])

    count = count+1
df.corr()["sellPrice"]
import missingno as msno
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)
from sklearn.impute import KNNImputer

knn_imputer=KNNImputer()
df["bed"]=knn_imputer.fit_transform(df[["bed"]])

df["car"]=knn_imputer.fit_transform(df[["car"]])
msno.bar(df)
df = df.drop(["postalCode"],axis=1)



from statsmodels.stats.outliers_influence import variance_inflation_factor



df_x = df.select_dtypes(include=["float64","int64"])

df_x.dropna(inplace=True)

X = df_x.drop("sellPrice",axis=1)

vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X.values,i)for i in range(X.shape[1])]

vif["features"] = X.columns
vif.head()
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X=df.drop(["sellPrice","propType_acreage","propType_warehouse","Day"],axis=1)

Y=df["sellPrice"]

X=sm.add_constant(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
stats_model=sm.OLS(Y_train,X_train).fit()

print(stats_model.summary())
X=df.drop(["sellPrice","propType_acreage","propType_warehouse","Day"],axis=1)

Y=df["sellPrice"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
sklearn_model=LinearRegression().fit(X_train,Y_train)
cross_val_score(sklearn_model,X_train,Y_train,cv=10,scoring="r2").mean()
from sklearn.metrics import mean_squared_error
print("Train RMSE : ",np.sqrt(mean_squared_error(Y_train,sklearn_model.predict(X_train))))

print("Test RMSE : ",np.sqrt(mean_squared_error(Y_test,sklearn_model.predict(X_test))))
dff = pd.get_dummies(dff,columns= ["suburb","propType"], prefix= ["suburb","propType"])
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0,1))

array = list(dff.select_dtypes(include=["float64","int64"]))

dff.loc[:,array] = scaler.fit_transform(dff.loc[:,array])









dff["bed"]=knn_imputer.fit_transform(dff[["bed"]])

dff["car"]=knn_imputer.fit_transform(dff[["car"]])
X=dff.drop(["sellPrice"],axis=1)

Y=dff["sellPrice"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
sklearn_model=LinearRegression().fit(X,Y)

print("R2 Score :",r2_score(Y_train,sklearn_model.predict(X_train)))

print("Train RMSE : ",np.sqrt(mean_squared_error(Y_train,sklearn_model.predict(X_train))))

print("Test RMSE: ",np.sqrt(mean_squared_error(Y_test,sklearn_model.predict(X_test))));