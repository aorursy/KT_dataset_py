import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/black-friday/train.csv')
df_backup = pd.read_csv('../input/black-friday/train.csv')
df.head()
df.info()
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df["Gender"])
dftemp = df[['Gender', 'Purchase']].groupby(['Gender'], as_index=False)
dfG = dftemp.sum()
dfG
dfG.plot(kind='bar')
plt.xlabel = ('Gender')
plt.pyplot.hist(df["Age"])
df['Occupation'].plot(kind='hist', figsize=(20, 5), bins=20)
dftemp = df[['Occupation', 'Purchase']].groupby(['Occupation'], as_index=False)
dfO = dftemp.sum()
dfO
dfO.plot(kind='bar')
plt.pyplot.hist(df["City_Category"])
dftemp = df[['City_Category', 'Purchase']].groupby(['City_Category'], as_index=False)
dfCity = dftemp.sum()
dfCity
dfCity.plot(kind='bar')
plt.pyplot.hist(df["Stay_In_Current_City_Years"])
dftemp = df[['Stay_In_Current_City_Years', 'Purchase']].groupby(['Stay_In_Current_City_Years'], as_index=False)
dfStay = dftemp.sum()
dfStay
dfStay.plot(kind='bar')
plt.pyplot.hist(df["Marital_Status"], bins = 20)
dftemp = df[['Marital_Status', 'Purchase']].groupby(['Marital_Status'], as_index=False)
dfMarried = dftemp.sum()
dfMarried
plt.pyplot.hist(df["Product_Category_1"], bins = 20)
df['Product_Category_2'].unique()
plt.pyplot.hist(df["Product_Category_2"], bins = 20)
df["Product_Category_2"].value_counts()
df['Product_Category_2'].isnull().sum()
# here we filled null values with most frequently occuring values
cnt=0
for i,j in df.iterrows():
    if pd.isnull(j['Product_Category_2']):
        if cnt <= 70000:
            df['Product_Category_2'][i] = '8.0'
            cnt+=1
        elif cnt <=130000:
            df['Product_Category_2'][i] = '14.0'
            cnt+=1
        else :
            df['Product_Category_2'][i] = '2.0'
            cnt+=1
        print(cnt)
                
            
df['Product_Category_2']
df['Product_Category_2'].isnull().sum()
df['Product_Category_2'] = df['Product_Category_2'].astype(int)
df['Product_Category_2'].dtype
df['Product_Category_3'].unique()
df["Product_Category_3"].value_counts()
df['Product_Category_3'].isnull().sum()
cnt=0
for i,j in df.iterrows():
    if pd.isnull(j['Product_Category_3']):
        if cnt <= 125000:
            df['Product_Category_3'][i] = '16.0'
            cnt+=1
        elif cnt <=240000:
            df['Product_Category_3'][i] = '15.0'
            cnt+=1
        elif cnt <= 300000 :
            df['Product_Category_3'][i] = '14.0'
            cnt+=1
        elif cnt <= 345000 :
            df['Product_Category_3'][i] = '17.0'
            cnt+=1
        else :
            df['Product_Category_3'][i] = '5.0'
            cnt+=1
        print(cnt)
df['Product_Category_3'].isnull().sum()
df['Product_Category_3'] = df['Product_Category_3'].astype(int)
df['Product_Category_3'].dtype
df.info()
y_data = df['Purchase'].copy()
x_data = df.copy()
x_data.info()
x_data.drop(['Purchase', 'User_ID', 'Product_ID'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
categorical_column = ['Gender','Age','City_Category','Stay_In_Current_City_Years']
le = LabelEncoder()
for i in categorical_column:
    x_data[i] = le.fit_transform(x_data[i])
x_data.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=1)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

test_y_hat = lm.predict(x_test)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(test_y_hat , y_test))
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_train, y_train)

test_y_hat = model.predict(x_test)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(test_y_hat , y_test))
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, objective ='reg:linear')
my_model.fit(x_train,y_train)
predictions = my_model.predict(x_test)

from sklearn.metrics import mean_absolute_error
print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((predictions - y_test) ** 2))
print("Accuracy of train dataset is : ",my_model.score(x_train,y_train))
print("Accuracy of test dataset is : ",my_model.score(x_test,y_test))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(predictions, y_test))
x_data_new = x_data.copy()
x_data_new.drop(['Stay_In_Current_City_Years', 'Marital_Status', 'Occupation'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(x_data_new, y_data, test_size=0.20, random_state=1)
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_train_n, y_train_n)

test_y_hat = model.predict(x_test_n)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test_n)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test_n) ** 2))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(test_y_hat , y_test_n))
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, objective ='reg:linear')
my_model.fit(x_train_n,y_train_n)
predictions = my_model.predict(x_test_n)

from sklearn.metrics import mean_absolute_error
print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - y_test_n)))
print("Residual sum of squares (MSE): %.2f" % np.mean((predictions - y_test_n) ** 2))
print("Accuracy of train dataset is : ",my_model.score(x_train_n,y_train_n))
print("Accuracy of test dataset is : ",my_model.score(x_test_n,y_test_n))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(predictions, y_test_n))
x_train_final = x_data.copy()
y_train_final = y_data.copy()
x_test_final = pd.read_csv('../input/black-friday/test.csv')
x_test_final.head()
x_test_final.info()
x_test_final["Product_Category_2"].value_counts()
x_test_final["Product_Category_2"].isnull().sum()
# here we filled null values with most frequently occuring values
cnt=0
for i,j in x_test_final.iterrows():
    if pd.isnull(j['Product_Category_2']):
        if cnt <= 35000:
            x_test_final['Product_Category_2'][i] = '8.0'
            cnt+=1
        elif cnt <=60000:
            x_test_final['Product_Category_2'][i] = '14.0'
            cnt+=1
        else :
            x_test_final['Product_Category_2'][i] = '2.0'
            cnt+=1
        print(cnt)
                
x_test_final['Product_Category_2'] = x_test_final['Product_Category_2'].astype(int)
from sklearn.preprocessing import LabelEncoder
categorical_column = ['Gender','Age','City_Category','Stay_In_Current_City_Years']
le = LabelEncoder()
for i in categorical_column:
    x_test_final[i] = le.fit_transform(x_test_final[i])
x_test_final.head()
x_test_final.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)
x_train_final.drop('Product_Category_3', axis=1, inplace=True)
x_test_final.drop('Product_Category_3', axis=1, inplace=True)
x_test_final.info()
x_train_final.info()
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(x_train_final,y_train_final)
predictions = my_model.predict(x_test_final)
print(predictions)
col_list = ['User_ID', 'Product_ID']
df_submission = pd.read_csv('../input/black-friday/test.csv',usecols=col_list)
df_submission.head()
df_submission['Purchase'] = predictions
df_submission.head()
df_submission.set_index('Purchase', inplace=True)
df_submission.head()
df_submission.to_csv("submission.csv")