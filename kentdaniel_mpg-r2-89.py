# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

df.head()
df.info()
df["horsepower"].unique()
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') #If ‘coerce’, then invalid parsing will be set as NaN.
df.describe().style.background_gradient()
def show_unique(df,col):

    df_info=pd.DataFrame(index=df.columns,data=[df[i].nunique() for i in col])

    return df_info.T.style.background_gradient(axis=1)

show_unique(df,df.columns)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

target="mpg"

sns.distplot(df[target])
sns.boxplot(x=df[target])
disc_ft=["origin"]

cont_ft=[i for i in df.columns if i not in [target]+disc_ft+["car name"]]
cont_ft
def plot_cont(df,con_ft,size):

    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)

    plt.subplots_adjust(right=2)

    plt.subplots_adjust(top=2)

    for i, feature in enumerate(list(df[con_ft]),1):

        plt.subplot(len(list(con_ft)), 3, i)

        sns.distplot(df[feature],kde=False)



        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

        plt.title('skewness : {:.2f} kurtosis : {:.2f}'.format(df[feature].skew(),df[feature].kurtosis()),fontsize=12)



        for j in range(2):

            plt.tick_params(axis='x', labelsize=12)

            plt.tick_params(axis='y', labelsize=12)



        plt.legend(loc='best', prop={'size': 10})

    plt.show()

plot_cont(df,cont_ft,(10,15))
sns.boxplot(x="origin", y=target, data=df)
sns.boxplot(x="cylinders", y=target, data=df)
type(df[["model year",target]].corr().iloc[1,:1][0])
def plot_cont_target(con_ft,df,target,size):

    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)

    plt.subplots_adjust(right=2)

    plt.subplots_adjust(top=2)

    for i, feature in enumerate(list(df[con_ft]),1):

        plt.subplot(len(list(con_ft)), 3, i)

        

        sns.scatterplot(x=feature, y=target, data=df)

        plt.title("Correlation: {:.2f}".format(df[[feature,target]].corr().iloc[1,:1][0]))

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

    



        for j in range(2):

            plt.tick_params(axis='x', labelsize=12)

            plt.tick_params(axis='y', labelsize=12)



        plt.legend(loc='best', prop={'size': 10})

    plt.show()

plot_cont_target(cont_ft,df,target,(10,20))
df[cont_ft+[target]].corr().style.background_gradient("Oranges")
df[df["horsepower"].isnull()]
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
def impute_hp():

    model=LinearRegression()

    data=df.dropna().copy()

    X=data[["displacement","weight"]]

    y=data["horsepower"]

    model.fit(X,y)

    return model

model_hp_imputer=impute_hp()
df_hp_cleaned=df[df["horsepower"].isnull()]

df_hp_cleaned.loc[:,('horsepower')]=model_hp_imputer.predict(df[df["horsepower"].isnull()][["displacement","weight"]])
df_hp_cleaned
df.dropna(axis=0,inplace=True)

df=df.append(df_hp_cleaned,ignore_index=True)
df.drop(columns=["car name"],inplace=True)
def baseline():

    model=LinearRegression()

    data=df.copy()

    X=data.drop(columns=[target])

    y=data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=69420504)

    model.fit(X_train,y_train)

    y_true=y_test

    y_pred=model.predict(X_test)

    plt.figure(figsize=(24,6))

    plt.plot(list(range(len(X_test))),y_true)

    plt.scatter(list(range(len(X_test))),y_pred,color='red',marker='o')

    print("mean squared error: {:.2f} miles per gallon".format(mean_squared_error(y_true,y_pred)))

    print("mean abs error: {:.2f} miles per gallon".format(mean_absolute_error(y_true,y_pred)))

    print("r2: {}".format(r2_score(y_true,y_pred)))

baseline()
np.log1p(df).corr().style.background_gradient("Oranges")
df_log=np.log1p(df)
data=df_log

X=data.drop(columns=[target])

y=data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=69420504)
model=LinearRegression()

model.fit(X_train,y_train)

y_true=y_test

y_pred=model.predict(X_test)
plt.figure(figsize=(24,6))

plt.plot(list(range(len(X_test))),np.expm1(y_true))

plt.scatter(list(range(len(X_test))),np.expm1(y_pred),color='red',marker='o')

print("mean squared error: {:.2f} miles per gallon".format(mean_squared_error(np.expm1(y_true),np.expm1(y_pred)))) # use exp(x)-1 as an inverse of log(x+1)

print("mean abs error: {:.2f} miles per gallon".format(mean_absolute_error(np.expm1(y_true),np.expm1(y_pred))))

print("r2: {}".format(r2_score(y_true,y_pred)))