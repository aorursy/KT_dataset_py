import numpy as np

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

import math

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge



%matplotlib inline



import os

print(os.listdir("../input"))

## Read the input file

xy_df = pd.read_csv(r"../input/xy.csv")
sns.lmplot(x='x',y='y',data=xy_df,fit_reg=False)
#get the predicted value of y, using the database

#We will try this at various degrees so let's add the degree here

def get_lreg(data,deg=1,show=True):

    X = data.iloc[:,:deg]

    X = sm.add_constant(X)

    Y = data['y']

    print(X.shape)

    #print(X,Y)

    model = None

    model = sm.OLS(Y,X).fit()

    Y_pred = model.predict(X)

    if(True == show):

        plt.plot(data.x,data.y,"k.",Markersize=10)

        plt.plot(data.x,Y_pred,"r.",Markersize=20,label = "degree" + str(deg) + "fit")

        plt.legend(loc='upper right')



    return model,Y_pred



def get_lreg_alpha(data,deg=1,alpha = 0.01,show=True):

    X = data.iloc[:,:deg]

    X = sm.add_constant(X)

    Y = data['y']

    lreg = Ridge(alpha)

    model = lreg.fit(X,Y)

    Y_pred = model.predict(X)

    if(True == show):

        plt.plot(data.x,data.y,"k.",Markersize=10)

        plt.plot(data.x,Y_pred,"r.",Markersize=20,label = "degree: " + str(deg) + "alpha :" + str(alpha))

        plt.legend(loc='upper right')

    return model,Y_pred,X



def get_lreg_skt(data,deg=1,show=True):

    X = data.iloc[:,:deg]

    print(X.shape)

    Y = data['y']

    lreg = LinearRegression()

    model = lreg.fit(X,Y)

    Y_pred = model.predict(X)

    if(True == show):

        plt.plot(data.x,data.y,"k.",Markersize=10)

        plt.plot(data.x,Y_pred,"r.",Markersize=20,label = "degree" + str(deg) + "fit")

        plt.legend(loc='upper right')

    return model,Y_pred

xy_df = sm.add_constant(xy_df)

xy_df.head()
#Now Add more variables to database, each representing x.pow()

for i in range(2,6):

    xy_df['x'+str(i)] = xy_df['x'].pow(i)

xy_df.head()

y = xy_df['y']

xy_df.drop('y',axis=1,inplace=True)

xy_df['y'] = y

print(xy_df.head())
output = []

alpha_vals = [10,1,0.1,0.01,0.001,0.0001]

degree = [2,3,4,5,6]

for alphaVal in alpha_vals:

    for degVal in degree:

        model,Ypred,X = get_lreg_alpha(xy_df,deg=degVal,alpha = alphaVal,show=False)

        error = xy_df['y'] - Ypred

        rmse = np.sqrt(np.mean(error**2))

        score = model.score(X,xy_df.y)

        output.append((degVal,alphaVal,rmse,score))

output_df = pd.DataFrame(output,columns=['Degree','alpha','rmse','score'])

grp_deg = output_df.groupby('Degree')

#output_df.to_csv(r"..\input\sample_summary.csv")
for degree, temp_df in grp_deg:

    print(degree)

    print(temp_df)
#From the above, it looks like we can go with degree as 3, and alpha as 0.0010  

degree=3

alphaVal = 0.001

model,Ypred,X = get_lreg_alpha(xy_df,deg=degVal,alpha = alphaVal,show=False)

test_df = pd.read_csv(r"../input/test.csv")

test_df.head()
test_df = sm.add_constant(test_df)



#Now Add more variables to database, each representing x.pow()

for i in range(2,6):

    test_df['x'+str(i)] = xy_df['x'].pow(i)

test_df.head()
Ypred_test = model.predict(test_df)

Ypred_test_df = pd.DataFrame({'Id':test_df.index,'y':Ypred_test})

#Ypred_test_df.to_csv(r"sample_submission.csv")

#Try various graphs at various degrees

model,Y_pred = get_lreg(xy_df,1)

model.pvalues
model,Y_pred = get_lreg(xy_df,2)

model.pvalues
model,Y_pred = get_lreg(xy_df,3)

model.pvalues