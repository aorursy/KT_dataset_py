import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
file="../input/auto.csv"

df = pd.read_csv(file, header=None)
# show the first 5 rows using dataframe.head() method

print("The first 5 rows of the dataframe") 

df.head(5)
# create headers list

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]

print("headers\n", headers)
df.columns = headers

df.head(5)
#Data Types

df.dtypes
df.describe(include="all")
# replace "?" to NaN

df.replace("?", np.nan, inplace = True)

df.head(5)
#The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data

#True indicates Missing Data

missing_data = df.isnull()

missing_data.head(10)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("") 
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)

print("Average of normalized-losses:", avg_norm_loss)
avg_bore=df['bore'].astype('float').mean(axis=0)

print("Average of bore:", avg_bore)
avg_stroke=df['stroke'].astype('float').mean(axis=0)

print("Average of stroke:", avg_stroke)
avg_horsepower=df['horsepower'].astype('float').mean(axis=0)

print("Average of horsepower:", avg_horsepower)
avg_peak_rpm=df['peak-rpm'].astype('float').mean(axis=0)

print("Average of peak-rpm:", avg_peak_rpm)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

df["bore"].replace(np.nan, avg_bore, inplace=True)

df['stroke'].replace(np.nan,avg_stroke, inplace= True) 

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

df['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace=True)

df.head()
#We can see here that "foor" is the most frequent value of the 

df['num-of-doors'].value_counts()
#We replace the missing 'num-of-doors' values by "foor"

df["num-of-doors"].replace(np.nan, "four", inplace=True)
# simply drop whole row with NaN in "price" column

df.dropna(subset=["price"], axis=0, inplace=True)



# reset index, because we droped two rows

df.reset_index(drop=True, inplace=True)
df.head()
#Lets list the data types for each column

df.dtypes
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

df[["price"]] = df[["price"]].astype("float")

df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

df["horsepower"]=df["horsepower"].astype(int, copy=True)
df.dtypes
# replace (original value) by (original value)/(maximum value)

df['length'] = df['length']/df['length'].max()

df['width'] = df['width']/df['width'].max()

df['height'] = df['height']/df['height'].max()

df[["length","width","height"]].head()
#Lets plot the histogram of horspower, to see what the distribution of horsepower looks like.

%matplotlib inline

import matplotlib as plt

from matplotlib import pyplot

plt.pyplot.hist(df["horsepower"])



# set x/y labels and plot title

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

bins
#We set group names:

group_names = ['Low', 'Medium', 'High']
# We apply the function "cut" to determine what each value of "df['horsepower']" belongs to

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )

df[['horsepower','horsepower-binned']].head(20)
# number of vehicles in each bin

df["horsepower-binned"].value_counts()
#distribution of each bin

%matplotlib inline

import matplotlib as plt

from matplotlib import pyplot

pyplot.bar(group_names, df["horsepower-binned"].value_counts())



# set x/y labels and plot title

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

dummy_variable_1.head()
#change column names for clarity

dummy_variable_1.rename(columns={'fuel-type-diesel':'diesel', 'fuel-type-diesel':'gas'}, inplace=True)

dummy_variable_1.head()
dummy_variable_2 = pd.get_dummies(df["aspiration"])

dummy_variable_2.head()
# merge data frame "df" and "dummy_variable_1" 

df = pd.concat([df, dummy_variable_1], axis=1)

df = pd.concat([df, dummy_variable_2], axis=1)

df.head()
df.corr()
#Let's find the scatterplot of "engine-size" and "price"

# Engine size as potential predictor variable of price

sns.regplot(x="engine-size", y="price", data=df)
sns.regplot(x="highway-mpg", y="price", data=df)
sns.regplot(x="peak-rpm", y="price", data=df)
#Let's look at the relationship between "body-style" and "price"

sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
sns.boxplot(x="drive-wheels", y="price", data=df)
df.describe()
df.describe(include=['object'])
from sklearn.linear_model import LinearRegression
#Create the linear regression object

lm = LinearRegression()

lm
#we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable

X = df[['highway-mpg']]

Y = df['price']
#Fit the linear model using highway-mpg.

lm.fit(X,Y)
#We can output a prediction

y_pred=lm.predict(X)

y_pred[0:5]
#the value of the intercept (a)

lm.intercept_
#the value of the Slope (b)

lm.coef_
#Let's develop a model using these variables as the predictor variables

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z, df['price'])
# the value of the intercept(a)

lm.intercept_

#the values of the coefficients (b1, b2, b3, b4)

lm.coef_
sns.regplot(x="highway-mpg", y="price", data=df)
#First lets make a prediction

Y_pred = lm.predict(Z)
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")

sns.distplot(Y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)





plt.title('Actual vs Fitted Values for Price')

plt.xlabel('Price (in dollars)')

plt.ylabel('Proportion of Cars')



plt.show()

plt.close()
#We will use the following function to plot the data:



def PlotPolly(model, independent_variable, dependent_variabble, Name):

    x_new = np.linspace(15, 55, 100)

    y_new = model(x_new)



    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')

    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')

    ax = plt.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()

    plt.xlabel(Name)

    plt.ylabel('Price of Cars')



    plt.show()

    plt.close()
x = df['highway-mpg']

y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 

f = np.polyfit(x, y, 3)

p = np.poly1d(f)

print(p)
PlotPolly(p, x, y, 'highway-mpg')
#highway_mpg_fit

lm.fit(X, Y)

# Find the R^2

print('The R-square is: ', lm.score(X, Y))
from sklearn.metrics import mean_squared_error

Yhat=lm.predict(X)

print('The output of the first four predicted value is: ', Yhat[0:4])

mse = mean_squared_error(df['price'], Yhat)

print('The mean square error of price and predicted value is: ', mse)
# fit the model 

lm.fit(Z, df['price'])

# Find the R^2

print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ', \

      mean_squared_error(df['price'], Y_predict_multifit))
from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))

print('The R-square value is: ', r_squared)
print('The mean squared error value is: ',mean_squared_error(df['price'], p(x)))
#First lets only use numeric data

df=df._get_numeric_data()

df.head()
y_data = df['price']

x_data=df.drop('price',axis=1)
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])

lr = LinearRegression()

lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
#Prediction using training data:

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_train[0:5]
#Prediction using test data:

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

yhat_test[0:5]
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):

    width = 12

    height = 10

    plt.figure(figsize=(width, height))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)



    plt.title(Title)

    plt.xlabel('Price (in dollars)')

    plt.ylabel('Proportion of Cars')



    plt.show()

    plt.close()
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
from sklearn.model_selection import GridSearchCV
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

parameters1
RR=Ridge()

RR
Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR=Grid1.best_estimator_

BestRR
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)