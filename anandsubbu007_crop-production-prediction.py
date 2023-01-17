import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv("../input/crop-production-in-india/crop_production.csv")

df[:5]
df.isnull().sum()
# Droping Nan Values

data = df.dropna()

print(data.shape)

test = df[~df["Production"].notna()].drop("Production",axis=1)

print(test.shape)
for i in data.columns:

    print("column name :",i)

    print("No. of column :",len(data[i].unique()))

    print(data[i].unique())
sum_maxp = data["Production"].sum()

data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)
data[:5]
sns.lineplot(data["Crop_Year"],data["Production"])
plt.figure(figsize=(25,10))

sns.barplot(data["State_Name"],data["Production"])

plt.xticks(rotation=90)
sns.jointplot(data["Area"],data["Production"],kind='reg')
sns.barplot(data["Season"],data["Production"])
data.groupby("Season",axis=0).agg({"Production":np.sum})
data["Crop"].value_counts()[:5]
top_crop_pro = data.groupby("Crop")["Production"].sum().reset_index().sort_values(by='Production',ascending=False)

top_crop_pro[:5]
rice_df = data[data["Crop"]=="Rice"]

print(rice_df.shape)

rice_df[:3]
sns.barplot("Season","Production",data=rice_df)
plt.figure(figsize=(13,10))

sns.barplot("State_Name","Production",data=rice_df)

plt.xticks(rotation=90)

plt.show()
top_rice_pro_dis = rice_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(

    by='Production',ascending=False)

top_rice_pro_dis[:5]

sum_max = top_rice_pro_dis["Production"].sum()

top_rice_pro_dis["precent_of_pro"] = top_rice_pro_dis["Production"].map(lambda x:(x/sum_max)*100)

top_rice_pro_dis[:5]
plt.figure(figsize=(18,12))

sns.barplot("District_Name","Production",data=top_rice_pro_dis)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot("Crop_Year","Production",data=rice_df)

plt.xticks(rotation=45)

#plt.legend(rice_df['State_Name'].unique())

plt.show()
sns.jointplot("Area","Production",data=rice_df,kind="reg")
coc_df = data[data["Crop"]=="Coconut "]

print(coc_df.shape)

coc_df[:3]
sns.barplot("Season","Production",data=coc_df)
plt.figure(figsize=(13,10))

sns.barplot("State_Name","Production",data=coc_df)

plt.xticks(rotation=90)

plt.show()
top_coc_pro_dis = coc_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(

    by='Production',ascending=False)

top_coc_pro_dis[:5]

sum_max = top_coc_pro_dis["Production"].sum()

top_coc_pro_dis["precent_of_pro"] = top_coc_pro_dis["Production"].map(lambda x:(x/sum_max)*100)

top_coc_pro_dis[:5]
plt.figure(figsize=(18,12))

sns.barplot("District_Name","Production",data=top_coc_pro_dis)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot("Crop_Year","Production",data=coc_df)

plt.xticks(rotation=45)

#plt.legend(rice_df['State_Name'].unique())

plt.show()
sns.jointplot("Area","Production",data=coc_df,kind="reg")
sug_df = data[data["Crop"]=="Sugarcane"]

print(sug_df.shape)

sug_df[:3]
sns.barplot("Season","Production",data=sug_df)
plt.figure(figsize=(13,8))

sns.barplot("State_Name","Production",data=sug_df)

plt.xticks(rotation=90)

plt.show()
top_sug_pro_dis = sug_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(

    by='Production',ascending=False)

top_sug_pro_dis[:5]

sum_max = top_sug_pro_dis["Production"].sum()

top_sug_pro_dis["precent_of_pro"] = top_sug_pro_dis["Production"].map(lambda x:(x/sum_max)*100)

top_sug_pro_dis[:5]
plt.figure(figsize=(18,8))

sns.barplot("District_Name","Production",data=top_sug_pro_dis)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot("Crop_Year","Production",data=sug_df)

plt.xticks(rotation=45)

#plt.legend(rice_df['State_Name'].unique())

plt.show()
sns.jointplot("Area","Production",data=sug_df,kind="reg")
data1 = data.drop(["District_Name","Crop_Year"],axis=1)
data_dum = pd.get_dummies(data1)

data_dum[:5]
x = data_dum.drop("Production",axis=1)

y = data_dum[["Production"]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)

print("x_train :",x_train.shape)

print("x_test :",x_test.shape)

print("y_train :",y_train.shape)

print("y_test :",y_test.shape)

x_train[:5]
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(x_train,y_train)

preds = model.predict(x_test)
from sklearn.metrics import r2_score

r = r2_score(y_test,preds)

print("R2score when we predict using Randomn forest is ",r)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)
preds = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y_test,preds)

r2_score(y_test,preds)
tst = test.drop(["District_Name","Crop_Year"],axis=1)

tst_dum = pd.get_dummies(tst)

tst_dum[:5]
y_test = tst_dum.copy()

print(x_train.shape)

print(y_test.shape)
def common_member(x_train,x_test): 

    a_set =  set(x_train.columns.tolist())

    b_set =  set(x_test.columns.tolist())

    if (a_set & b_set): 

        return list(a_set & b_set) 
com_fea = common_member(x_train,tst_dum)

len(com_fea)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(x_train[com_fea],y_train)

preds = model.predict(y_test[com_fea])
preds
test["production"] = preds
test[:10]
test.to_csv('Prediction.csv')