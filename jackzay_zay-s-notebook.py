# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline
df_test = pd.read_csv("/kaggle/input/dasprodatathon/test.csv") 

df_train = pd.read_csv("/kaggle/input/dasprodatathon/train.csv") 

df_sample = pd.read_csv("/kaggle/input/dasprodatathon/sample_submission.csv")
print("data test")

display(df_test)



print("data train")

display(df_train)



print("data sample")

display(df_sample)
df_train.isnull().any()
print(df_train.dtypes)
# df_train.Price = df_train.Price.astype(int)

# df_train.Bathrooms = df_train.Bathrooms.astype(int)

# df_train.Floors = df_train.Floors.astype(int)

df_train.head(5)
df_train.describe()
corr = df_train.corr()

display(corr)
corr["Price"].sort_values(ascending = False)
fig, ax = plt.subplots(figsize=(20,20))

k = 19

  

cols = corr.nlargest(k, 'Price')['Price'].index

cm = np.corrcoef(df_train[cols].values.T)

ax = sns.heatmap(

    cm, 

    vmin=-1, vmax=1, center=0,

    cmap="cubehelix",

    square=True,

    annot=True,

    yticklabels = cols.values,

    xticklabels = cols.values

)

# sns.boxplot(df_train["Living Area"] )
# df_train["Price"]= np.log(df_train["Price"])

# df_train["Living Area"] = np.log(df_train["Living Area"])
coloumn = ["Living Area","Grade","Above the Ground Area","Neighbors Living Area","Bathrooms","View","Basement Area","Latitude","Bedrooms","Waterfront","Floors",

           "Year Renovated","Total Area","Neighbors Total Area","Year Built","Condition","Longitude"]

x_train = df_train[coloumn]

y_train = df_train["Price"]

x_test = df_test[coloumn]

id_test = df_test["ID"]

xs_train,xs_test,ys_train,ys_test = train_test_split(x_train, y_train, test_size = 0.2,random_state = 0)
xgbr = xgb.XGBRegressor(colsample_bytree = 0.5,

                       learning_rate = 0.1,

                       n_estimator = 1000,

                       max_depth = 5,

                       gamma = 0)

xgbr.fit(xs_train,ys_train)
cek_xgbr = round(xgbr.score(xs_test, ys_test) * 100,2)

cek_xgbr
linear = LinearRegression()

linear.fit(xs_train,ys_train)
coeff_df = pd.DataFrame(linear.coef_,x_train.columns, columns = ["Coefficient"])

coeff_df.sort_values("Coefficient",axis = 0, ascending = False, 

                 inplace = True, na_position ='first')

coeff_df
cek_linear = round(linear.score(xs_test, ys_test) * 100,2)

cek_linear
y_predict = xgbr.predict(xs_test)

display(y_predict)
df = pd.DataFrame({"Actual" : ys_test, "Predicted": y_predict})

df1 = df.head(25)

df1
df1.plot(kind="bar",figsize=(10,8))

plt.grid(which="major",linestyle="-",linewidth="0.5",color="green")

plt.grid(which="minor",linestyle=":",linewidth="0.5",color="black")

plt.show()
print("MAE :",(metrics.mean_absolute_error(ys_test, y_predict)))

print("MSE :",(metrics.mean_squared_error(ys_test, y_predict)))

print("RSME :",np.sqrt(metrics.mean_squared_error(ys_test, y_predict)))

print("R squared :",metrics.r2_score(ys_test,y_predict)*100)
xgbr.fit(x_train,y_train)

predict_result = xgbr.predict(x_test)
submission_data = pd.DataFrame()

submission_data['ID'] = id_test

submission_data['Price'] = predict_result

display(submission_data.head())
submission_data.to_csv('ZAY_Submit.csv', index=False)