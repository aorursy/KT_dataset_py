# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")

data.head(20)
data.info()
data.fillna(0, inplace = True)

data.info()
data.describe()
data = data.drop(["User_ID","Product_ID"], axis = 1)

data.head(20)
ax = sns.countplot(data.Gender, label = "Count")



M, F = data.Gender.value_counts()



print("Number of male customer : {}".format(M))

print("Number of female customer : {}".format(F))





ax =sns.countplot(data.Age)



for each in data.Age.value_counts().index:

    new_data = data[data.Age == each]

    print(" Number of : {} ".format(new_data.Age.value_counts()))
ax = sns.countplot(data.Marital_Status)



Single, Married = data.Marital_Status.value_counts()



print("Number of Single : {}".format(Single))

print("Number of Married : {}".format(Married))
data = data.loc[:1000]





for each in data.Age:

    def age_func(each):

        if each == "0-17":

            return 0

        elif each == "18-25":

            return 1

        elif each == "26-35":

            return 2

        elif each == "36-45":

            return 3

        elif each == "46-50":

            return 4

        elif each == "51-55":

            return 5

        else:

            return 6



data.Age = data.Age.apply(age_func)





for each in data.Gender:

    def func_gender(each):

        if each == "F":

            return 0

        elif each == "M":

            return 1





data.Gender = data.Gender.apply(func_gender)





for each in data.City_Category:

    def func_city(each):

        if each == "A":

            return 0

        elif each == "B":

            return 1

        else:

            return 2



        

data.City_Category = data.City_Category.apply(func_city)







for each in data.Stay_In_Current_City_Years:

    def func_years(each):

        if "+" in each:

            return int(each.replace("+",""))

        else:

            return int(each)

        

data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.apply(func_years)



data
y = data.Purchase

x = data.drop(["Purchase"],axis =1)



x_norm = (x-x.min())/(x.max()- x.min())

x_norm
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.linear_model import LinearRegression



reg = LinearRegression()



reg.fit(x_norm,y)



print("b0 value is {}".format(reg.intercept_))

print("b1,b2 values are {}".format(reg.coef_))



y_pred = reg.predict(x_norm)

y_pred



reg.score(x_norm,y)