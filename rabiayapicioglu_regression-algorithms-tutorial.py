from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img1.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img2.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img3.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img4.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img5.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs/reg_img6.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Sİmple Linear Regression



#Y ≈ β 0 + β 1 X.



#this is our equation of simple linear regression model, here b0 is called 

#bias or intercept and b1 is coefficient



#First we'll import the required libraries matplotlib for data visualization

#pandas for dataframe,list,series data structures of python



#Second we'll import the csv files we need,to do that:

#  1.Choose file explorer and make sure that your csv file is in the correct directory

#  2.Use the read_csv command to import the required csv file



#  or choose variable explorer and click on the import data icon,find the file

#  that you want to import and click next,choose the data type dataframe and done.

#  if you import it as array then you should make some arrangement after that 

#  if we import the dataset as array,skiprows=1.

#  then convert it by using df=pd.DataFrame( nameofvariable(array),columns=['col1','col2'])



#if we import it as dataframe we can directly use it,and in spyder datasets have names

#you can directly use it like following

#plt.scatter(jobcsv.experience,jobcsv.income)

#if it is separated by ; then use sep=';' before go on in read_csv()



dataFrame=pd.read_csv('../input/regression/job2.csv')

dataFrame.head()



car_dataFrame=pd.read_csv('../input/regression/car.csv')



decision_dataFrame=pd.read_csv('../input/regression/car.csv',sep=',')



randForest_dataFrame=pd.read_csv('../input/regression/car.csv',sep=',')
#after getting this csv file visualize this data by using scatter plot

#let us choose one of the features of age and experience in simple linear regression

plt.scatter( dataFrame.experience,dataFrame.income )

plt.xlabel('Experince')

plt.ylabel('Income')

plt.show()



#here we saw a normal scatter plot here x is a feature that affects the values of y

#experience is a feature that affects the income
#now we'll go on with learning how to fit  line into this graph

#we'll be working with sklearn library,so import it



from sklearn.linear_model import LinearRegression



#initialize the linear regression model

linear_reg_model=LinearRegression()



#here why don't we use just x=dataFrame.experince ,cause this is a series

#or why don't we use x=dataFrame.experince.values,let's show the size of it



x=dataFrame.experience.values.shape

print(x)

#see it gives (12,) that means 12 rows and 1 column but it is

#not preferable by sklearn library so reshape it to (14,1) by using

#reshape(-1,1) method as below

x=dataFrame.experience.values.reshape(-1,1)

y=dataFrame.income.values.reshape(-1,1 )

#print( type(x)) x is now an array has the shape (14,1)



#fitting the line into our scatter plot



linear_reg_model.fit( x,y )
#after creating the equation now we find b0 and b1

#there are two ways of finding the bo,bo is the point where graph intersects

#the yaxis.So,give 0 to x.First way of finding b0 is,



b0=linear_reg_model.predict([[0]])

print('bo: ',b0)

#we must use 0 in 2D type array,cause we can predict more than one values

#Second way of finding b0 is using intercept_ method

b0=linear_reg_model.intercept_ 

print('b0: ',b0)

#these two result must be same



#and lets find b1 by using coefficient_ method

b1=linear_reg_model.coef_

print('b1: ',b1)
#testing manually



#and now we can use our simple AI,we'll show you two ways

#first manually predict 'How much money does a worker that has 30 years of

#experience gain ?'



experience_=30



new_income=b0+b1*experience_

print('new income: ', new_income )



#another way to use it -automatically- is,

print( 'new income with predict method:',linear_reg_model.predict([[experience_]]))

#so they are same as it can be seen
y_head=linear_reg_model.predict(x)

plt.scatter( x , y )

#if we plot this line in red we see the following output

plt.plot( x, y_head , color='red')

plt.show()



print('r_square score: ',r2_score(y,y_head))



#evaluation of regression models



#residual=y-y_head

#here y_head is the result of real prediction results

#square_residual=residual^2

#so Squared Sum Residual( SSR ) is: sum( (y-y_head)^2 )

#Assume that we've found y_head_avg

#so Squared Sum Total is sum( (y-y_head_avg)^2 )

#r_squared=1-( SSR-SST ) ,the cloesest value of R2 is the better prediction results
#it is time to see that already fitted line,let's plot it

#but first we must predict all the x values in our graph

#we'll call that prediction results as y_head

import numpy as np



#to see better how the fitted line changes we are scaling our array into a broader array

array=np.arange(min(x),max(x),0.01) #fix the (1500,)

array=array.reshape(-1,1)



#y_head includes the prediction results according to our predicted model

y_head=linear_reg_model.predict(array)





plt.scatter( x , y )

#if we plot this line in red we see the following output

plt.plot( array , y_head , color='red')

plt.show()





#Multiple Linear Regression

#we'll use same data frame to show multiple linear regression

#and the previously imported libraries are enough to create a multiple linear regression model



dataFrame.head()



#our equation will be like

#y=b0+b1*x1+b2*x2+...+bn*xn
#all columns and 0. and 2.column as a feature,independent variables

x=dataFrame.iloc[:,[0,1]].values

#we'll be separating dataFrame as we're taking all rows and just the 0. and 1. columns

#why we are taking just 0. and 1. columns? Cause They are the predictors...

print(type(x))

print(x.shape) #Check the shape,it is usable

#y is the dependent variable

y=dataFrame.income.values.reshape(-1,1)
#initializing our multiple linear regression model

multiple_linear_regression_model=LinearRegression()

multiple_linear_regression_model.fit(x,y)



#Let's find the intercept and coefficients...



print( 'b0: ',multiple_linear_regression_model.intercept_ )

print( 'b1: ',multiple_linear_regression_model.coef_ )#we'll see two values at there

#now let's predict that how much many does a man that has

# 5 years of experience and is 35 years old and another man that

#has 10 years of experience and 35 years old gain money ?

multiple_linear_regression_model.predict([[35,5],[35,10]])





y_head=multiple_linear_regression_model.predict(x)



#both age and experience affects the income,plot both of them in scatter mode

plt.scatter( dataFrame.experience,dataFrame.income,color='violet',label='experience' )

plt.scatter( dataFrame.age,dataFrame.income,color='green',label='age' )



plt.xlabel('Experince-Age')

plt.ylabel('Income')



#draw the fitting line

plt.plot( x , y_head , color='red')

plt.show()
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs2/ou.png")
#now we'll change the dataset that we've been working on



#our equation will be like

#y=b0+b1*x+b2*x^2+...+bn*x^n



import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



car_dataFrame.head()
#x is independent max_speed and y is dependent price variables don't forget the reshape them

x=car_dataFrame.max_speed.values.reshape(-1,1)

y=car_dataFrame.price.values.reshape(-1,1)

x[:5] # see what's inside x again
#First I'll show you the simple linear regression with a different dataset

linear_model=LinearRegression()

linear_model.fit(x,y)



y_head=linear_model.predict(x)

#it is enough to draw a simple linear regression line,and again our reai prediction result in y_head
#go on with polynomial regression model



#as the degree increases we get better prediction results in PolynomialFeatures(degree=10)

polynomial_regression_model=PolynomialFeatures(degree=10)

x_polynomial=polynomial_regression_model.fit_transform(x)



linear_model_poly=LinearRegression()

linear_model_poly.fit(x_polynomial,y)



y_head_poly=linear_model_poly.predict(x_polynomial)

#result of the prediction results of x_polynomial is in y_head_poly

y_head_poly[:5]

plt.scatter(x,y)

plt.plot(x,y_head,color='red',label='linear')

plt.plot(x,y_head_poly,color='green',label='polynomial')

plt.legend()

plt.show()

#so we see polynomial regression algorithm gives us better results
from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs3/dec.png")
decision_dataFrame.loc[:5,::]

#first five rows and all columns in the dataset
x=decision_dataFrame.iloc[:,0].values.reshape(-1,1)

y=decision_dataFrame.iloc[:,1].values.reshape(-1,1)



print('x shape: ',x.shape,'\ny shape: ',y.shape)
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()

tree_reg.fit(x,y)



print('The prediction result of 123 km/h: ',tree_reg.predict([[900]]))

print('The prediction result of 128 km/h: ',tree_reg.predict([[900]]))

      
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=tree_reg.predict(x_)



print('x shape: ',x_.shape,'\ny_head shape: ',y_head.shape)
plt.scatter(x,y,color='red')

plt.plot(x_,y_head,color='green')

plt.xlabel('max_speed km/h')

plt.ylabel('price')

plt.show()

from IPython.display import Image

import os

!ls ../input/



Image("../input/reg-imgs3/rand.png")
randForest_dataFrame.head()
x=randForest_dataFrame.iloc[:,0].values.reshape(-1,1)

y=randForest_dataFrame.iloc[:,1].values



print('x shape: ',x.shape,'\ny_head shape:',y.shape)


from sklearn.ensemble import RandomForestRegressor



rf=RandomForestRegressor(n_estimators=100, random_state=42 )

rf.fit(x,y)



y_head=rf.predict(x)



print('385 km/h price: ',rf.predict([[385]]))



plt.scatter(x,y,color='red' )

plt.plot(x,y_head,color='green')

plt.xlabel('max_speed')

plt.ylabel('price')

plt.show()



print('r2_score: ',r2_score(y,y_head))
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=rf.predict(x_)



plt.scatter(x,y,color='red' )

plt.plot(x_,y_head,color='green')

plt.xlabel('max_speed')

plt.ylabel('price')

plt.show()
