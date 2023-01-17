import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

from collections import Counter

import os

print(os.listdir("../input"))
df=pd.read_csv("../input/BlackFriday.csv")
df.columns
#Firstly, I want to convert column name to lowercase.

df.columns = df.columns.str.lower()

df.columns
df.shape

# In this dataset, we have 537577 rows and 12 columns
df.head()

# In the "Product_Category_2" and "Product_Category_2", we already see there are a null values.
df.tail()
df.info()

# we see that occupation and Marital_Status are integer but they must be string so we need to change their types.

# Moreover, we can change Product_Category_2 and Product_Category_3 to integer.
df.isna().any()

# we have "na" value in Product_Category_2 and Product_Category_3
df.isnull().sum()

# we have null values in Product_Category_2 (166986) and Product_Category_3 (373299)
df["product_category_2"].value_counts(dropna=False)

# In "Product_Category_2" we have 166986 null values.
df["product_category_3"].value_counts(dropna=False)

# In "Product_Category_3" we have 373299 null values.
# We replace "0" to null values.

df.fillna(0,inplace=True)

df.isnull().sum()
df["occupation"]=df["occupation"].astype("object")

df.marital_status=df.marital_status.astype("object")

df.product_category_2=df.product_category_2.astype(int)

df.product_category_3=df.product_category_3.astype(int)

df.info()
df.dtypes

# We have 7 strings and 5 integer.
df.loc[:,["product_category_1","product_category_2","product_category_3","purchase"]].describe()
df.columns
f,ax=plt.subplots(figsize=(10,6))

sb.countplot(df.gender)

plt.gca().invert_xaxis()

plt.show()

# We saw that males are much more bought the product in black friday.
plt.figure(figsize=(10,6))

plt.pie(df.gender.value_counts().values,explode=(0,0),labels=df.gender.value_counts().index,colors=("blue","green"),autopct="%1.1f%%")

plt.title("Pie Chart of Gender")

plt.show()

# We also see that male percentage is 75.4 and female percentage is 24.6.
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.gender,y=df.product_category_1,data=df)

plt.title("Product_Category_1 Bar Graph")

plt.show()

# we see that females are buy more products_category_1.
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.gender,y=df.product_category_2,data=df)

plt.title("Product_Category_2 Bar Graph")

plt.show()
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.gender,y=df.product_category_3,data=df)

plt.title("Product_Category_3 Bar Graph")

plt.show()

# we see that males are mostly prefer to buy more products_category_3.
f,ax=plt.subplots(figsize=(10,6))

ax=sb.barplot(x="gender",y="purchase",data=df,ax=ax)

plt.title("Bar Plot of Gender Base Purchase")

plt.gca().invert_xaxis()

plt.show()
df.columns
f,ax=plt.subplots(figsize=(10,6))

df_age_sorted=df.age.value_counts()

sb.barplot(x=df_age_sorted.index,y=df_age_sorted.values)

plt.title("Number of People by Age Category")

plt.xlabel("Age",size=12)

plt.ylabel("Number of People",size=12)

plt.show()

# we see that mostly 26-35 age group buy products in black friday.
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.age,y=df.product_category_1)

f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.age,y=df.product_category_2)

f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.age,y=df.product_category_3)

plt.show()
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.age,y=df.purchase,hue=df.gender)

plt.show()
f,ax=plt.subplots(figsize=(10,6))

sb.barplot(x=df.gender,y=df.purchase,hue=df.city_category)

plt.title("Product_Category_1 Bar Graph")

plt.show()

# we see that females are buy more products_category_1.
df.occupation.value_counts().plot(kind="bar",color="b",alpha=.8,figsize=(15,6))

plt.xlabel("Occupation")

plt.show()

# We see that occupations number 4 and 0 are the ones that bought the most on Friday.
f,ax=plt.subplots(figsize=(10,6))

sb.countplot(df.city_category)

plt.title("Bar Graph of Cities")

plt.gca().invert_xaxis()

plt.show()

# In city B, customers bought more products than other cities.
f,ax=plt.subplots(figsize=(10,6))

sb.countplot(x=df.age,hue=df.marital_status)

plt.show()

# Maried couple spend less money in the black friday compared to single people.
f,ax=plt.subplots(figsize=(10,6))

sb.countplot(x=df.age,hue=df.city_category)

plt.show()
plt.rcParams["axes.facecolor"]="w"

df["product_id"].value_counts()[:10].plot(kind="barh",color="r",alpha=.5,figsize=(15,6))

plt.title("Bar Graph of Top Ten Products")

plt.xlabel("Number of Solds")

plt.gca().invert_yaxis()

plt.show()
df.loc[:,["product_category_1","product_category_2","product_category_3","purchase"]].corr()
f,ax=plt.subplots(figsize=(10,6))

sb.heatmap(df.loc[:,["product_category_1","product_category_2","product_category_3","purchase"]].corr(),annot=True,linewidth=.5,fmt=".3f",ax=ax)

plt.title("Correlations of Variables")

plt.show()
sb.pairplot(df.loc[:,["product_category_1","product_category_2","product_category_3","purchase"]])

plt.show()
df.boxplot(column="purchase",by="occupation",figsize=(10,6))

plt.show()

# There are outliers almost in each occupations group.
df.boxplot(column="purchase",by="city_category",figsize=(15,8))

plt.show()

# There are outliers in city of A and B.
df.boxplot(column="purchase",by="marital_status",figsize=(15,8))

plt.show()

# There are outliers in each group.
df.boxplot(column="purchase",by="age",figsize=(15,8))

plt.show()
df_product_category_1_normalize=(df.product_category_1-df.product_category_1.mean())/df.product_category_1.std()

df_product_category_2_normalize=(df.product_category_2-df.product_category_2.mean())/df.product_category_2.std()

df_product_category_3_normalize=(df.product_category_3-df.product_category_3.mean())/df.product_category_3.std()

df_purchase_normalize=(df.purchase-df.purchase.mean())/df.purchase.std()

# Now,we normalize our values.



normalize_data=pd.concat([df_product_category_1_normalize,df_product_category_2_normalize,df_product_category_3_normalize,df_purchase_normalize]

                         ,axis=1)

normalize_data.head()
# Now, we try to make linear model. Firstly,we need to upload sklearn.

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
y=normalize_data.purchase.values.reshape(-1,1)

x=normalize_data.iloc[:,0:3].values

print(y.shape)

print(x.shape)
lm.fit(x,y)

y_head=lm.predict(x)

print(y_head)

y_head.shape

y.shape

print("")

from sklearn.metrics import r2_score

print("R^2 value:",r2_score(y,y_head))

# this r score is realy too low that means these variable cannot explain the purchase properly.

# Before we do that we know these variables cannot explain purchase because there are less correlation between them.
# Now, we make a decision tree.

from sklearn.tree import DecisionTreeRegressor

t_reg=DecisionTreeRegressor(random_state=42)

t_reg.fit(x,y)

y_head=t_reg.predict(x)

y_head
print("R^2 value:",r2_score(y,y_head))

# this R square can tell us this values can be more explain with decision tree.
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)

y_head=rf.predict(x)

r2_score(y,y_head)

# In the random forest we use 100 decision tree to predict values. When we look at the r square this random forest cannot explain better than one decision tree.
x.shape
# Now, we make the logistic regression and try classify the gender.

x=normalize_data

y=df.gender

y=[1 if i=="M" else 0 for i in y]

y=pd.DataFrame(np.array(y).reshape(537577,1))

y.shape

# Now we have a normalize x values and dependent y variable.
# Firstly, we need to split train and test data by %80 and %20.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# Now ,we make our logistic regression.

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

a=lr.predict(x_train).reshape(-1,1)

b=np.concatenate((a,y_train.values),axis=1)

b=pd.DataFrame(b,columns=["predict_value","y_train"])

b.head(20)

# We see that there are some values are not correctly classify and predict. But still it is make corret prediction some of them.
# Lets find our R square.

print("R^2 value:",lr.score(x_test,y_test))

# Our R square score is not bad. Therefore, we can say that our model works not bad.