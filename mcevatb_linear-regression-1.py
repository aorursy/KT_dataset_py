# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
# read csv (comma separated value) into data

data = pd.read_csv('../input/column_2C_weka.csv')

data.head()
data.info()
data.describe()
#First I visualize the seaborn heatmap to see  the correlation between the features to choose two with max. corr.

#and these are pelvic_incidence and sacral_slope

fig,ax = plt.subplots(figsize=(7,5))



ax = sns.heatmap(

data.corr(), 

annot=True, annot_kws={'size':8},

linewidths=.3,linecolor="blue", fmt= '.2f',square=True,cmap="YlGnBu_r",cbar=False)

plt.show()
data.loc[:,'class'].value_counts()
# For my kernel I used "pelvic_incidence" and "sacral_slope" for class "Normal"

data1 = data.loc[data['class'] =='Normal']

data1.head()
#numpy array for x_axis

#numpy array for y_axis

x = data1.pelvic_incidence.values

y = data1.sacral_slope.values

# Scatter

plt.figure(figsize=[6,6])

plt.scatter(x,y,color="magenta")

plt.xlabel('pelvic_incidence',fontsize = 25,color='blue')

plt.ylabel('sacral_slope',fontsize = 25,color='blue')

plt.show()

#Linear Regression

denominator= x.dot(x)-x.mean()*x.sum()

a= (x.dot(y)-y.mean()*x.sum())/denominator

b= (y.mean()*x.dot(x)-x.mean()*x.dot(y))/denominator

    

Yhat= a*x + b  # best fitting line



plt.figure(figsize=[6,6])

plt.scatter(x,y,color="magenta")

plt.plot(x,Yhat)

plt.xlabel('pelvic_incidence',fontsize = 25,color='blue')

plt.ylabel('sacral_slope',fontsize = 25,color='blue')

plt.show()
#accuracy

d1= y-Yhat

d2= y-y.mean()

r2= 1-(d1.dot(d1)/d2.dot(d2))

print("the r-squared is:",r2)
# Linear Regression with sklearn

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

X = x.reshape(-1,1)

Y = y.reshape(-1,1)

lr.fit(X,Y)

y_pred = lr.predict(X)  



# plot scatter

plt.figure(figsize=[6,6])

plt.scatter(X,Y,color="magenta")

plt.xlabel('pelvic_incidence',fontsize = 25,color='blue')

plt.ylabel('sacral_slope',fontsize = 25,color='blue')



#plot regression line

plt.plot(X, y_pred)

plt.show()



#accuracy

print("R^2 score: ", lr.score(X,Y))

#or

from sklearn.metrics import r2_score

print("r_square score: ",r2_score(Y,y_pred))
#Multiple Linear Regression

#load data (I took lumbar_lordosis_angle as x2)

x = data1.iloc[:, [0,2]].values

# I added a column of ones (noise)

X = np.insert(x,0,1,axis=1)

Y = data1.sacral_slope.values

print(X.shape)

print(Y.shape)
trace1 = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=Y,

    mode='markers',

    marker=dict(

        size=8,

        line=dict(

            color='rgba(217, 217, 217, 0.14)',

            width=0.5

        ),

        opacity=1

    ),

    

)



data_ = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data_, layout=layout)



iplot(fig)

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

Yhat = np.dot(X, w)
#accuracy

d1 = Y - Yhat

d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("the r-squared is:", r2)
#Decision Tree Regression

# I took first 15 samples of X and Y



x = data1.iloc[:15,[0]].values.reshape(-1,1)

y = data1.iloc[:15,[2]].values.reshape(-1,1)



from sklearn.tree import DecisionTreeRegressor

des_tree_reg = DecisionTreeRegressor()



des_tree_reg.fit(x,y)





x_new = np.linspace(min(x),max(x)).reshape(-1,1)

x_new1 = np.arange(min(x),max(x),0.01).reshape(-1,1)

yhat_new = des_tree_reg.predict(x_new)

yhat_new1 = des_tree_reg.predict(x_new1)

plt.subplot(111)

plt.scatter(x,y, c="m")

plt.plot(x_new,yhat_new)

plt.show()

plt.scatter(x,y, c="m")

plt.plot(x_new1,yhat_new1)

plt.show()
#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

random_for = RandomForestRegressor(n_estimators =100, random_state = 42)



random_for.fit(x,y)





x_new = np.linspace(min(x),max(x)).reshape(-1,1)

x_new1 = np.arange(min(x),max(x),0.01).reshape(-1,1)

yhat_new = random_for.predict(x_new)

yhat_new1 = random_for.predict(x_new1)

plt.subplot(111)

plt.scatter(x,y, c="m")

plt.plot(x_new,yhat_new)

plt.show()

plt.scatter(x,y, c="m")

plt.plot(x_new1,yhat_new1)

plt.show()
#accuracy

from sklearn.metrics import r2_score

yhat_rf = random_for.predict(x)



print("r_square score: ", r2_score(y,yhat_rf))

# or:

print("r2: ", random_for.score(x,y))

# linear regression with gradient descent

# I took first 15 samples of X and Y and multiply them with 0.1

x = data1.iloc[:15, [0,2]].values

x = x * 0.1

# I added a column of ones (noise)

X = np.insert(x,0,1,axis=1)

y = data1.sacral_slope.values

Y = y[:15] * 0.1





print(X.shape)

print(Y.shape)

D = X.shape[1]

N = X.shape[0]
#plot the data

fig = plt.figure()

ax = fig.add_subplot(111, projection ="3d")

ax.scatter(X[:,0],X[:,1],Y,c="r",marker="o")

plt.show()
cost = []

w = np.random.randn(D)/np.sqrt(D)

learning_rate = 0.001



for i in range(300):

    Yhat = X.dot(w)

    interval = Yhat - Y

    w = w - learning_rate * X.T.dot(interval)

    # mean squared error

    # 1/N *(Yhat -Y)        for N ...samples

    mse = interval.dot(interval) / N

    cost.append(mse)



plt.plot(cost)

plt.show()
print(w)
plt.plot(Yhat, label="prediction")

plt.plot(Y ,label="target")



plt.legend()

plt.show()
#accuracy

d1 = Y - Yhat

d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("the r-squared is:", r2)
# linear regression with l2 regularization(Ridge)

# I took the same data as with gradient descent

x = data1.iloc[:15, [0,2]].values

x = x * 0.1

# I added a column of ones (noise)

X = np.insert(x,0,1,axis=1)

y = data1.sacral_slope.values

Y = y[:15] * 0.1

# I added +20 to the last 2 samples of Y

Y[-1] += 20

Y[-2] += 20

print(X.shape)

print(Y.shape)

D = X.shape[1]

N = X.shape[0]
plt.scatter(X[:,1],Y)

plt.show()

# maximum likelihood

w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))

Yhat_ml = X.dot(w_ml)



plt.scatter(X[:,1],Y)

plt.plot(sorted(X[:,1]),sorted(Yhat_ml))

plt.show()
# l2 regularization (Ridge)

l2 = 1000.0

w_ridge = np.linalg.solve(l2*np.eye(3) + X.T.dot(X),X.T.dot(Y))

Yhat_ridge = X.dot(w_ridge)



plt.scatter(X[:,1],Y)

plt.plot(sorted(X[:,1]),sorted(Yhat_ml), label = "maximum likelihood")

plt.plot(sorted(X[:,1]),sorted(Yhat_ridge), label = "l2_reg/ridge")

plt.legend()

plt.show()
#Logistic Regression

data["class"] = ["1" if each =="Normal" else "0" for each in data["class"]]

data.head()
data.loc[:,'class'].value_counts()
y = data["class"].values

x_ = data.drop(["class"],axis=1)

print(y.shape,x_.shape)

x_.head()
#Normalisation

x = (x_ - np.min(x_))/(np.max(x_) - np.min(x_)).values

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state=42)

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

test_acc = lr.score(x_test,y_test)

print("accuracy:",test_acc)