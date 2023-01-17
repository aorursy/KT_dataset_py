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
ins_df = pd.read_csv("../input/insurance.csv")
ins_df
df = ins_df
df
df.isna().sum()
df.index
df.head(10)
df.tail(15)
import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns
df.describe().T
ins_corr=df.corr()

ins_corr
ins_cov=df.cov()

ins_cov
sns.heatmap(ins_corr,vmin=-1,vmax=1,center=0,annot=True)
sns.pairplot(data=df,hue='children')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['children'],size=df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['children'],size=df['age'])
sns.pairplot(data=df,hue='region')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['region'],size=df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['region'],size=df['age'])
sns.pairplot(data=df,hue='smoker')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['smoker'],size=df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['smoker'],size=df['age'])
sns.pairplot(data=df,hue='sex')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['sex'],size=df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['sex'],size=df['age'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
cat_col=['smoker','region','sex']

num_col=[i for i in df.columns if i not in cat_col]

num_col
# one-hot encoding

one_hot=pd.get_dummies(df[cat_col])

ins_procsd_df=pd.concat([df[num_col],one_hot],axis=1)

ins_procsd_df.head(10)
#label encoding

ins_procsd_df_label=df

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for i in cat_col:

    ins_procsd_df_label[i] = label_encoder.fit_transform(ins_procsd_df_label[i])

ins_procsd_df_label.head(10)
#using one hot encoding

x=ins_procsd_df.drop(columns='expenses')

y=df[['expenses']]
x
y
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1234)
model = LinearRegression()



model.fit(train_x,train_y)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)
cdf = pd.DataFrame(data=model.coef_.T, index=x.columns, columns=["Coefficients"])

cdf
# Print various metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_x)

print("Predicting the test data")

test_predict = model.predict(test_x)

print("MAE")

print("Train : ",mean_absolute_error(train_y,train_predict))

print("Test  : ",mean_absolute_error(test_y,test_predict))

print("====================================")

print("MSE")

print("Train : ",mean_squared_error(train_y,train_predict))

print("Test  : ",mean_squared_error(test_y,test_predict))

print("====================================")

import numpy as np

print("RMSE")

print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))

print("====================================")

print("R^2")

print("Train : ",r2_score(train_y,train_predict))

print("Test  : ",r2_score(test_y,test_predict))

print("MAPE")

print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)
#Plot actual vs predicted value

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted expenses",fontsize=25)

plt.xlabel("Actual expenses",fontsize=18)

plt.ylabel("Predicted expenses", fontsize=18)

plt.scatter(x=test_y,y=test_predict)
#using label encoding

x1=ins_procsd_df.drop(columns='expenses')

y1=ins_procsd_df_label[['expenses']]
x1
y1
# split data into train and test

train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y1,test_size=0.3,random_state=1234)
# Create Linear regression model with train and test data

model = LinearRegression()



model.fit(train_x1,train_y1)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)
cdf1 = pd.DataFrame(data=model.coef_.T, index=x1.columns, columns=["Coefficients"])

cdf1
# Print various metrics



from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_x1)

print("Predicting the test data")

test_predict = model.predict(test_x1)

print("MAE")

print("Train : ",mean_absolute_error(train_y1,train_predict))

print("Test  : ",mean_absolute_error(test_y1,test_predict))

print("====================================")

print("MSE")

print("Train : ",mean_squared_error(train_y1,train_predict))

print("Test  : ",mean_squared_error(test_y1,test_predict))

print("====================================")

import numpy as np

print("RMSE")

print("Train : ",np.sqrt(mean_squared_error(train_y1,train_predict)))

print("Test  : ",np.sqrt(mean_squared_error(test_y1,test_predict)))

print("====================================")

print("R^2")

print("Train : ",r2_score(train_y1,train_predict))

print("Test  : ",r2_score(test_y1,test_predict))

print("MAPE")

print("Train : ",np.mean(np.abs((train_y1 - train_predict) / train_y1)) * 100)

print("Test  : ",np.mean(np.abs((test_y1 - test_predict) / test_y1)) * 100)
#Plot actual vs predicted value

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted expenses",fontsize=25)

plt.xlabel("Actual expenses",fontsize=18)

plt.ylabel("Predicted expenses", fontsize=18)

plt.scatter(x=test_y1,y=test_predict)
print("MAPE")

print("Train : ",np.mean(np.abs((train_y1 - train_predict) / train_y1)) * 100)

print("Test  : ",np.mean(np.abs((test_y1 - test_predict) / test_y1)) * 100)