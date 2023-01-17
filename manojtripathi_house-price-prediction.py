# Importing Libraries like numpy for mathematical calculation, pandas for dataframes, matplotlib for drawing data graps etc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import metrics

from pandas.plotting import scatter_matrix
# Reading the Real Estate.csv data into notebook.
real=pd.read_csv("../input/real-estate/Real Estate.csv")

real
real.shape # total no of rows and cols.
real.size # total no of elements in data set
real.head() # printing top 5 rows from dataset.
real.tail() # printing last 5 rows from dataset.
real.dtypes
# Using Distribution Plot to get a sense how the variables are distributed.
sns.distplot(real['total_rooms']) # by default the distribution plot uses histogram and fit a KDE(kernal density estimate)on it.

real['total_rooms'].mean()# the middle of the bell curve of distplot will give mean of the col here we can see mena is 2635. when we compare with the graph value it'll be the same.
sns.distplot(real['housing_median_age']) # as we can see in the graph the data of the 'housing_median_age' is distributed soo much as I understand that the data is cahging form 18 to 35 as median age.

real['housing_median_age'].mean()
sns.distplot(real['median_income']) # as we can see in the graph the data is uniformaly distributed and we can see the clear bell curve is formaing in the data

print("mean value: ",real['median_income'].mean())

print("maximum value: ",real['median_income'].max()) # as their may be some outliers in the 'median_income' col because with the 

# max() it is showing maximum value is 15.0001 and when we look at the graph the maximum value is more than 25.

print("minimum value",real['median_income'].min())
sns.distplot(real['median_house_value'])

real['median_house_value'].mean()
# As we know we have one categorical col in our Data Set which is " ocean_proximity " so,for categorical col 

# we can't use Distribution Plot. For this we will use some other plols.
real.head(0)
real['ocean_proximity'].value_counts() # Here we can see that ocean_proximity has 5 categories
figsize=(25,25)

sns.barplot(x='ocean_proximity',y='housing_median_age',data=real )# so, from the bar we can understand that housing age is more 

# in ISLAND(which is 43 )  as compared to others categories.

figsize=(25,25)

sns.barplot(x='ocean_proximity',y='median_income',data=real)
real.hist(figsize=(25,25),bins=50)

# A histogram is used when we have to check the  distribution of a continuous variable in a dataset

# here from this histogram we analysed that their is lot of data change rate in longitude,population,total bedrooms,total rooms.

#and in case of housing_median_age the data is not varing so much in it'starting phase but at the middle it is slightly 

# chnaging and when it comes to end the graph is incresing too much it menas their may be some outliers in the col.

# for checking outliers we will use different method.
real.plot.scatter(x = 'total_rooms', y = 'total_bedrooms')
sns.jointplot(x='total_rooms',y='total_bedrooms',data=real,kind='reg')

#so, here we are finding he best fit line by using jointplot and kind ='reg', what this graph is showing is that: the best fit 

# line passs through those data points which are having minimum distance from that line. the line can be in any direction.
# sns.pairplot(real,hue='ocean_proximity',palette='rainbow')
# sns.pairplot(real,hue='median_house_value',palette='rainbow')
scatter_matrix(real,figsize=(25,25),alpha=0.9,diagonal="kde",marker="o") #understanding how one col is related to other col using scatter matrix.....

# Scatter plots shows how much one variable is affected by another or the relationship between them

# with the help of dots in two dimensions. 

# Scatter plots are very much like line graphs in the concept that they use horizontal and vertical axes to plot data points.
# so, from the above scatter matrix plot we can say that some columns are highly related to each other like: (median_income and housing_median_age),

# (median_house_value and housing_median_age) and many more col... we can see which box has higher density of datapoints their 

# respective coumns are highely related to each others.....
# Displaying the above data in the form of heatmap
sns.set(context='paper',font='monospace')

real_cor_matrix=real.corr()

fig,axe=plt.subplots(figsize=(12,8))

cmap=sns.diverging_palette(220,10,center='light',as_cmap=True)

sns.heatmap(real_cor_matrix,vmax=1,square=True,cmap=cmap,annot=True)
# from above heatmap we understand that col total_rooms and total_bedrooms ,total_rooms and households , similarly total_bedrooms

# and populations are highly related because we can see that right side color identifier line that 0.8 and above is highly related 

# and similarly less than 0.4 are not soo much related to each other.
plt.boxplot(real["total_rooms"],showmeans=True)

plt.show()
# box plot is mainly use to detect outliers in any particular column. the red line in the box shows the mean vlaue of the column

# and here in this column box plot detected outliers form 23,000 because from the range of (20,000 - 40,000) the datapoints are 

# scattering too much hence they are outliers in the dataset.
print("mean value : ",real['total_rooms'].mean())

print("minimum value  : ",real['total_rooms'].min())

print("maximum value   : ",real['total_rooms'].max())

# as we can see here also the minimum value is 2 and respective to it maximum value is 39320 and mean is 2635 , hence if we take 

# difference datapoints from mean value to maximum vlaue theri will be soo much of differences in values so, those values are called

# outliers in the dataset.
def getOutliers(dataframe,column):

    column = "total_rooms" 

    des = dataframe[column].describe()

    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}

    Q1 = des[desPairs['25']]

    Q3 = des[desPairs['75']]

    IQR = Q3-Q1

    lowerBound = Q1-1.5*IQR #finding lower bound

    upperBound = Q3+1.5*IQR #finding upper bound

    print("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))

    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

    print("Outliers out of total = {} are \n {}".format(real[column].size,len(data[column])))

    outlierRemoved = real[~real[column].isin(data[column])] #remove the outliers from the dataframe

    return outlierRemoved
df_outliersRemoved = getOutliers(real,"total_rooms")
real.isnull().sum() # isnull() displyes null values respective to it's columns

# here their ar 207 null values in total_bedrooms columns
real['total_bedrooms'].mean()

# real['total_bedrooms'].mode()

# real['total_bedrooms'].max()

# real['total_bedrooms'].min()
real['total_bedrooms'].fillna(value=np.mean(real['total_bedrooms']),inplace=True)

real.isnull().sum()
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

real["ocean_proximity"] = labelEncoder.fit_transform(real["ocean_proximity"])

real["ocean_proximity"].value_counts()

# real["ocean_proximity"].head()

real.head()
x=real.iloc[:,[0,1,2,3,4,5,6,7,8]] # independent variables(cols)

y=real.iloc[:,[-1]] # dependent variables(cols)
x.head(0)# Independent columns in x
y.head(0)# Dependent column in y (Target variable)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head() #Independent variable training dataset (80%)

# print(x_train.shape)

# print(x_train.size)
x_test.head() #Independent variable testing dataset (20%)

print(x_test.shape)

print(x_test.size) # displaying shape and size of x_test(20%) data
y_train.head() #Dependent variable training dataset (80%)

# print(y_train.shape)

# print(y_train.size)
y_test.head() #Dependent variable testing dataset (20%)

print(y_test.shape)

print(y_test.size) # Displaying shape and size of y_test(20%) data
# Applying Linear Regression model without Standardizing the data and checking the accuracy.....
from sklearn.linear_model import LinearRegression #importing linear regression from sklearn...

lm=LinearRegression() # creating object of model..
lm.fit(x_train,y_train)# providing traing data to model.......
pred=lm.predict(x_test)

pred
from sklearn.metrics import r2_score

r2_score(y_test,pred)
# appling Linear regression model with Standardizing the data.
from sklearn.preprocessing import StandardScaler

independent_scaler = StandardScaler()

x_train = independent_scaler.fit_transform(x_train)

x_test = independent_scaler.transform(x_test)

x_train
lm.fit(x_train,y_train)# providing traing data to model.......
pred=lm.predict(x_test)# after fitting into the model priniting predicted values

pred

c=pd.DataFrame(pred)

c.head()
y_test.head() # printing original value and if we compare with predicted value in above cell some of the values are closer but in some case the difference of values is very high.

# hence, we can say that Linear Regression model gives the minimum accuracy in this case.
print(np.sqrt(metrics.mean_squared_error(y_test,pred))) # printing Root Mean Squared Error (RMSE)
from sklearn.metrics import r2_score # checking the accuracy with with Adjusted R2

r2_score(y_test,pred)
from sklearn.tree import DecisionTreeRegressor# importing decision tree form sklearn lib....
dtReg = DecisionTreeRegressor(max_depth=9) # creating obj of decision tree and providing max_depth =9

dtReg.fit(x_train,y_train)# fitting training data to the model.
dtReg_y_pred = dtReg.predict(x_test)# after fitting into the model priniting predicted values

dtReg_y_pred

e=pd.DataFrame(dtReg_y_pred)

e.head()
y_test.head() # printing original value and if we compare with predicted value in above cell their is a lot of difference in the predicted values and the original values,

# hence, we can say that Decision Tree Regression model is not fit in this case(dataset).
print(np.sqrt(metrics.mean_squared_error(y_test,dtReg_y_pred))) # printing Root Mean Squared Error (RMSE)
from sklearn.metrics import r2_score# checking the accuracy with with Adjusted R2

r2_score(y_test,dtReg_y_pred)
from sklearn.ensemble import RandomForestRegressor # importing Random forest from sklearn module.
rfReg = RandomForestRegressor(30)# creating object of random forest 

rfReg.fit(x_train,y_train)# fitting training data in random forest
rfReg_y_pred = rfReg.predict(x_test)# after fitting into the model priniting predicted values

rfReg_y_pred

d=pd.DataFrame(rfReg_y_pred) # converting the data into dataframe so that, we can get the data in right format.

d.head()
y_test.head() # printing original value and if we compare with predicted value in above cell the values are quite closer,

# hence, we can say that Random Forest Regressor model gives the maximum accuracy.
print(np.sqrt(metrics.mean_squared_error(y_test,rfReg_y_pred))) # printing Root Mean Squared Error (RMSE)
from sklearn.metrics import r2_score # checking the accuracy with with Adjusted R2

r2_score(y_test,rfReg_y_pred)