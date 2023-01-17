# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split

from ipywidgets import interact

from sklearn.metrics import accuracy_score, r2_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/insurance.csv") #read csv file

df
# df.smoker.replace(('yes', 'no'), (1, 0), inplace=True)

# df.sex.replace

df["sex"] = df["sex"].astype('category')

df["smoker"] = df["smoker"].astype('category')

df["region"] = df["region"].astype('category')

df.dtypes

df["sex"] = df["sex"].cat.codes

df["smoker"] = df["smoker"].cat.codes

df["region"] = df["region"].cat.codes

df

x=df.iloc[:,0:6].values

y=df.iloc[:,-1].values
def Train_Test_Split(x,y,test_size=0.5,random_state=None):

    n=len(y)

    if len(x) ==len(y):

        if random_state:

            np.random.seed(random_state)

        shuffle_index = np.random.permutation(n)

        x=x[shuffle_index]

        y=y[shuffle_index]

        test_data = round(n*test_size)

#         train_data =n-test_data

        x_train, x_test = x[test_data:], x[:test_data]

        y_train, y_test = y[test_data:], y[:test_data]

        return x_train, x_test, y_train, y_test

    else:

        print("Data not Equal")
xtrain, xtest, ytrain, ytest =Train_Test_Split(x ,y, test_size=0.3, random_state=8)

xtrain.shape
model = LinearRegression()

model.fit(xtrain,ytrain)
m = model.coef_

c = model.intercept_
x=[10,0,38.665,2,1,3]

y_pred = model.predict([x])

y_pred
y_pred = model.predict(xtest)
def Insurance_Predict(age,sex,bmi,children,smoker,region):

    y_pred = model.predict([[age, sex, bmi, children, smoker, region]])

    print("Insurance cost is: ", y_pred[0])
age_min =df.iloc[:, 0].min()

age_max =df.iloc[:, 0].max()

sex_min =df.iloc[:, 1].min()

sex_max =df.iloc[:, 1].max()

bmi_min =df.iloc[:, 2].min()

bmi_max =df.iloc[:, 2].max()

children_min =df.iloc[:,3].min()

children_max =df.iloc[:,3].max()

smoker_min =df.iloc[:, 4].min()

smoker_max =df.iloc[:, 4].max()

region_min =df.iloc[:, 5].min()

region_max =df.iloc[:, 5].max()
interact(Insurance_Predict, age=(age_min, age_max) , 

        sex=(sex_min, sex_max),

        bmi =(bmi_min, bmi_max), 

        children = (children_min, children_max),

        smoker=(smoker_min, smoker_max),

        region=(region_min, region_max))