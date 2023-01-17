df = pd.read_csv("../input/insurance.csv")

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df.plot(kind="hist")
df.shape
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df.columns
df.info()
df.head(4)
df.isna().sum()
df.expenses.duplicated().sum()
df_num=df.select_dtypes(include=np.number)

df_cat=df.select_dtypes(exclude=np.number) 
encode_cat_data=pd.get_dummies(df_cat)

encode_cat_data.head(5).T

fin_df=[df_num,encode_cat_data]

df1=pd.concat(fin_df,axis=1)

df1.describe().head(5)
all_columns         = list(df)

numeric_columns     = ['age', 'bmi', 'children', 'expenses']

categorical_columns = [x for x in all_columns if x not in numeric_columns ]

cols_to_transform = categorical_columns

print('\nNumeric columns')

print(numeric_columns)

print('\nCategorical columns')

print(categorical_columns)

print(cols_to_transform)
df2=df1.drop(columns="expenses")

#df2

ax = sns.boxplot(data = df2, orient = "v", color = "Violet",)

plt.show()
#df1=df



#pd.get_dummies()
y=df1[["expenses"]]

x=df1.drop(columns=["expenses"])
x.head(5)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=100)
model=LinearRegression()

model.fit(train_x,train_y)

print(model.intercept_)
print(model.coef_)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("predicting the train data")

train_predict = model.predict(train_x)

print(train_predict)
import numpy as np

print("RMSE")

print("train : ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("test : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("predictin the test data")

test_predict = model.predict(test_x)

print("MAE")

print("train : ",mean_absolute_error(train_y,train_predict))

print("test : ",mean_absolute_error(test_y,test_predict))



print("MSE")

print("train : ",mean_squared_error(train_y,train_predict))

print("test : ",mean_squared_error(test_y,test_predict))
print("Rsquare")

print("train : ",r2_score(train_y,train_predict))

print("test : ",r2_score(test_y,test_predict))
plt.figure(figsize = (12,8))

g = sns.countplot(x="smoker",data=df1,palette='hls')

g.set_title("Expenses of Smokers and Non-Smokers", fontsize=20)

g.set_xlabel("smoker", fontsize=15)

g.set_ylabel("expenses", fontsize=20)
plt.figure(figsize = (12,8))

g = sns.countplot(x="age",data=df1,palette='hls')

g.set_title("Smoker in different ages", fontsize=20)

g.set_xlabel("age", fontsize=15)

g.set_ylabel("smoker", fontsize=20)
plt.figure(figsize=(12,6))

g = sns.distplot(df1["bmi"])

g.set_xlabel("bmi", fontsize=12)

g.set_ylabel("Frequency", fontsize=12)

g.set_title("Frequency Distribuition- bmi", fontsize=20)
#BOX PLOT Using SeaBorn

plt.figure(figsize = (20,8))

ax = sns.boxplot(x="expenses" ,y= "smoker", data=df1, linewidth=2.5)

plt.show()
plt.figure(figsize = (20,8))

ax = sns.boxplot(x="age" ,y= "expenses", data=df1, linewidth=2.5)

plt.show()
correlation_m = df1.corr()

correlation_m["expenses"].sort_values(ascending=False)
df1.region.value_counts().plot(kind="pie")
df1.groupby("smoker").expenses.agg(["mean","median","count"])
# Set the width and height of the figure

plt.figure(figsize=(20,10))



corr = df1.corr()

ax = sns.heatmap(corr,vmin=-1,vmax=1,center=0,annot=True)

import seaborn as sns

g = sns.pairplot(df1)

g.set(xticklabels=[])