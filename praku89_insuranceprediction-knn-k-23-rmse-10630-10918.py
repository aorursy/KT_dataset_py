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
insurance_df=pd.read_csv("../input/insurance.csv")
insurance_df.shape
insurance_df.isna().sum()
insurance_df.index
insurance_df.head(10)
import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns
insurance_df.describe()
insur_corr=insurance_df.corr()

insur_corr
insur_cov=insurance_df.cov()

insur_cov
sns.heatmap(insur_corr,vmin=-1,vmax=1,center=0,annot=True)
#sns.pairplot(data=insurance_df,hue='children')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['age'], y=insurance_df['expenses'],hue=insurance_df['children'],size=insurance_df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['bmi'], y=insurance_df['expenses'],hue=insurance_df['children'],size=insurance_df['age'])
sns.pairplot(data=insurance_df,hue='region')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['age'], y=insurance_df['expenses'],hue=insurance_df['region'],size=insurance_df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['bmi'], y=insurance_df['expenses'],hue=insurance_df['region'],size=insurance_df['age'])
sns.pairplot(data=insurance_df,hue='smoker')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['age'], y=insurance_df['expenses'],hue=insurance_df['smoker'],size=insurance_df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['bmi'], y=insurance_df['expenses'],hue=insurance_df['smoker'],size=insurance_df['age'])
sns.pairplot(data=insurance_df,hue='sex')
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['age'], y=insurance_df['expenses'],hue=insurance_df['sex'],size=insurance_df['bmi'])
plt.figure(figsize=(14, 7))

sns.scatterplot(x=insurance_df['bmi'], y=insurance_df['expenses'],hue=insurance_df['sex'],size=insurance_df['age'])
# Importing necessary package for creating model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

cat_col=['smoker','region','sex']

num_col=[i for i in insurance_df.columns if i not in cat_col]

num_col
# one-hot encoding

one_hot=pd.get_dummies(insurance_df[cat_col])

insur_procsd_df=pd.concat([insurance_df[num_col],one_hot],axis=1)

insur_procsd_df.head(10)
#label encoding

insr_procsd_df_label=insurance_df

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for i in cat_col:

    insr_procsd_df_label[i] = label_encoder.fit_transform(insr_procsd_df_label[i])

insr_procsd_df_label.head(10)
#using one hot encoding

X=insur_procsd_df.drop(columns='expenses')

y=insurance_df[['expenses']]
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1234)
model = LinearRegression()



model.fit(train_X,train_y)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)
cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])

cdf
# Print various metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_X)

print("Predicting the test data")

test_predict = model.predict(test_X)

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

X=insur_procsd_df.drop(columns='expenses')

y=insr_procsd_df_label[['expenses']]
# split data into train and test

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=2000)
# Create Linear regression model with train and test data

model = LinearRegression()



model.fit(train_X,train_y)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)
cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])

cdf
# Print various metrics



from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_X)

print("Predicting the test data")

test_predict = model.predict(test_X)

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
print("MAPE")

print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size = 0.3, random_state = 100)

y_train=np.ravel(y_train)

y_test=np.ravel(y_test)
rmse_train_dict={}

rmse_test_dict={}

df_len=round(sqrt(len(insur_procsd_df)))

#Train Model and Predict  

for k in range(3,df_len):

    neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)

    yhat_train = neigh.predict(X_train)

    yhat = neigh.predict(X_test)

    test_rmse=sqrt(mean_squared_error(y_test,yhat))

    train_rmse=sqrt(mean_squared_error(y_train,yhat_train))

    rmse_train_dict.update(({k:train_rmse}))

    rmse_test_dict.update(({k:test_rmse}))

    print("RMSE for train : ",train_rmse," test : ",test_rmse," difference between train and test :",abs(train_rmse-test_rmse)," with k =",k)
elbow_curve_train = pd.Series(rmse_train_dict,index=rmse_train_dict.keys())

elbow_curve_test = pd.Series(rmse_test_dict,index=rmse_test_dict.keys())

elbow_curve_train.head(10)
ax=elbow_curve_train.plot(title="RMSE of train VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of train")

ax=elbow_curve_test.plot(title="RMSE of test VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of test")

#for K in range(25):

    #K_value = K+1

    #neigh = KNeighborsRegressor(n_neighbors = K_value, weights='uniform', algorithm='auto')

    #neigh.fit(X_train, y_train) 

    #y_pred = neigh.predict(X_test)

    #print("MAPE")

    #print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

    #print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)