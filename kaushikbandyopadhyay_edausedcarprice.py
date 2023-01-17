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
# Import pandas library

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib as plt

from matplotlib import pyplot

import seaborn as sns

# Read the file path and load data to the dataframe, df_auto

df_auto = pd.read_csv('/kaggle/input/auto.csv',header=None)



# create headers list

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]

#assign headers

df_auto.columns = headers

df_auto.head(15)
# replace "?" to NaN

df_auto.replace('?',np.nan,inplace=True)

df_auto.head(15)
#find out the number of columns and the number of reocrds with null data

missing_data = df_auto.isnull().sum()

missing_data
#dealing with missing data, replacing NaN for 5 columns by their mean

avg_norm_loss = df_auto['normalized-losses'].astype('float').mean(axis=0)

avg_bore = df_auto['bore'].astype('float').mean(axis=0)

avg_stroke = df_auto['stroke'].astype('float').mean(axis=0)

avg_horsepower = df_auto['horsepower'].astype('float').mean(axis=0)

avg_peakrpm = df_auto['peak-rpm'].astype('float').mean(axis=0)



df_auto["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

df_auto["bore"].replace(np.nan, avg_norm_loss, inplace=True)

df_auto["stroke"].replace(np.nan, avg_norm_loss, inplace=True)

df_auto["horsepower"].replace(np.nan, avg_norm_loss, inplace=True)

df_auto["peak-rpm"].replace(np.nan, avg_norm_loss, inplace=True)

df_auto.head(15)
#check values are present in a num-of-doors column using value_counts() and check max value using idxmax()

df_auto['num-of-doors'].value_counts().idxmax()
#Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur. 

df_auto['num-of-doors'].replace(np.nan,'four',inplace=True)
# drop whole row with NaN in "price" column, since price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us

df_auto.dropna(subset=["price"], axis=0, inplace=True)



# reset index, because we droped two rows

df_auto.reset_index(drop=True, inplace=True)

df_auto.head(15)
df_auto.dtypes
#change data types to proper format

df_auto[["bore", "stroke"]] = df_auto[["bore", "stroke"]].astype("float")

df_auto[["normalized-losses"]] = df_auto[["normalized-losses"]].astype("int")

df_auto[["price"]] = df_auto[["price"]].astype("float")

df_auto[["peak-rpm"]] = df_auto[["peak-rpm"]].astype("float")

df_auto.dtypes
# standardise data

df_auto['city-L/100km'] = 235/df_auto["city-mpg"]

df_auto['highway-L/100km'] = 235/df_auto["highway-mpg"]

df_auto.head()
df_auto.columns.values
#values of several variables into a similar range

df_auto['height'] = df_auto['height']/df_auto['height'].max()

df_auto['width'] = df_auto['width']/df_auto['width'].max()

df_auto['length'] = df_auto['length']/df_auto['length'].max()
df_auto['horsepower'] = df_auto['horsepower'].astype(int,copy=True)
plt.pyplot.hist(df_auto["horsepower"])



# set x/y labels and plot title

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")
#We build a bin array, with a minimum value to a maximum value, with bandwidth calculated above. The bins will be values used to determine when one bin ends and another begins.

bins = np.linspace(min(df_auto['horsepower']),max(df_auto['horsepower']),4)

group_names = ['Low', 'Medium', 'High']

df_auto['horsepower-binned'] = pd.cut(df_auto['horsepower'], bins, labels=group_names, include_lowest=True )

df_auto[['horsepower','horsepower-binned']].head(20)
df_auto['horsepower-binned'].value_counts()
pyplot.bar(group_names, df_auto["horsepower-binned"].value_counts())

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")
plt.pyplot.hist(df_auto["horsepower"], bins = 3)



# set x/y labels and plot title

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")
dummy_variable_1=pd.get_dummies(df_auto['fuel-type'])

dummy_variable_1
df_auto = pd.concat([df_auto, dummy_variable_1], axis=1)



# drop original column "fuel-type" from "df"

df_auto.drop("fuel-type", axis = 1, inplace=True)
dummy_variable_2=pd.get_dummies(df_auto['aspiration'])

dummy_variable_2
df_auto = pd.concat([df_auto, dummy_variable_2] , axis=1)

df_auto.drop("aspiration", axis = 1, inplace=True)

df_auto.head(20)
df_auto[['bore','stroke' ,'compression-ratio','horsepower']].corr()
# Engine size as potential predictor variable of price

df_auto[["engine-size", "price"]].corr()

sns.regplot(x='engine-size',y='price',data=df_auto)

plt.pyplot.ylim(0,)

#As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. 

#Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.
sns.regplot(x='highway-mpg',y='price',data=df_auto)

plt.pyplot.ylim(0,)

df_auto[['highway-mpg','price']].corr()

#As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.
sns.regplot(x='peak-rpm',y='price',data=df_auto)

plt.pyplot.ylim(0,)

df_auto[['peak-rpm','price']].corr()

#Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore it's it is not a reliable variable.
sns.regplot(x='horsepower',y='price',data=df_auto)

plt.pyplot.ylim(0,)

df_auto[['horsepower','price']].corr()

#As the horsepower goes up, the price goes up: this indicates an positive relationship between these two variables. Horsepower could potentially be a predictor of price.
sns.boxplot(x='body-style',y='price',data=df_auto)

#We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price.
drive_wheels_count=df_auto['drive-wheels'].value_counts().to_frame()

drive_wheels_count.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

drive_wheels_count.index.name = 'drive-wheels'

df_auto['drive-wheels'].unique()
df_gptest = df_auto[['drive-wheels','body-style','price']]

grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()

grouped_test1
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0

grouped_pivot
fig, ax = plt.pyplot.subplots()

im = ax.pcolor(grouped_pivot, cmap='RdBu')



#label names

row_labels = grouped_pivot.columns.levels[1]

col_labels = grouped_pivot.index



#move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)

ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)



#insert labels

ax.set_xticklabels(row_labels, minor=False)

ax.set_yticklabels(col_labels, minor=False)



#rotate label if too long

plt.pyplot.xticks(rotation=90)



fig.colorbar(im)

plt.pyplot.show()

#The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis respectively. This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.
from scipy import stats
#Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)

pearson_coef, p_value = stats.pearsonr(df_auto['wheel-base'], df_auto['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(df_auto['width'], df_auto['price'])

print("Widht: The Pearson Correlation Coefficient for is", pearson_coef, " with a P-value of P = ", p_value)  

#Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).

pearson_coef, p_value = stats.pearsonr(df_auto['curb-weight'], df_auto['price'])

print("Curb Weight: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

#Since the p-value is  <  0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).

pearson_coef, p_value = stats.pearsonr(df_auto['engine-size'], df_auto['price'])

print("Engine-Size: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

#Since the p-value is  <  0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).

pearson_coef, p_value = stats.pearsonr(df_auto['bore'], df_auto['price'])

print("Price: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

#Since the p-value is  <  0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).

pearson_coef, p_value = stats.pearsonr(df_auto['highway-mpg'], df_auto['price'])

print("Hihghway MPG: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of ~ -0.705 shows that the relationship is negative and moderately strong.

pearson_coef, p_value = stats.pearsonr(df_auto['city-mpg'], df_auto['price'])

print("City MPG:The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

#Since the p-value is  <  0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of ~ -0.687 shows that the relationship is negative and moderately strong.
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test2.get_group('4wd')['price']

# ANOVA

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  

print( "ANOVA results: F=", f_val, ", P =", p_val)   

#This is a great result, with a large F test score showing a strong correlation and a P value of almost 0 implying almost certain statistical significance.
'''We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:



Continuous numerical variables:



Length

Width

Curb-weight

Engine-size

Horsepower

City-mpg

Highway-mpg

Wheel-base

Bore

Categorical variables:



Drive-wheels

As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.

'''
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X=df_auto[['highway-mpg']]

Y=df_auto['price']

#Fit the linear model using highway-mpg.

lm.fit(X,Y)
Yhat=lm.predict(X)

Yhat[0:5] 
print('Intercept: ',lm.intercept_)

print('Slope: ',lm.coef_)
price = lm.intercept_ + lm.coef_*X

price
Z = df_auto[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z,df_auto['price'])

print(lm.intercept_)

print(lm.coef_)
width = 12

height = 10

#plt.pyplot.figure(figsize=(width, height))

sns.regplot(x="highway-mpg", y="price", data=df_auto)

plt.pyplot.ylim(0,)

#We can see from this plot that price is negatively correlated to highway-mpg, since the regression slope is negative
plt.pyplot.figure(figsize=(width, height))

sns.regplot(x="peak-rpm", y="price", data=df_auto)

plt.pyplot.ylim(0,)

#not a suitable model for prediction
#Comparing the regression plot of "peak-rpm" and "highway-mpg" we see that the points for "highway-mpg" are much closer to the generated line and on the average decrease. The points for "peak-rpm" have more spread around the predicted line, and it is much harder to determine if the points are decreasing or increasing as the "highway-mpg" increases.
width = 12

height = 10

plt.pyplot.figure(figsize=(width, height))

sns.residplot(df_auto['highway-mpg'], df_auto['price'])

plt.pyplot.show()

#We can see from this residual plot that the residuals are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data.
Y_hat = lm.predict(Z)
plt.pyplot.figure(figsize=(width, height))





ax1 = sns.distplot(df_auto['price'], hist=False, color="r", label="Actual Value")

sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)





plt.pyplot.title('Actual vs Fitted Values for Price')

plt.pyplot.xlabel('Price (in dollars)')

plt.pyplot.ylabel('Proportion of Cars')



plt.pyplot.show()

plt.pyplot.close()

#We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.
def PlotPolly(model, independent_variable, dependent_variabble, Name):

    x_new = np.linspace(15, 55, 100)

    y_new = model(x_new)



    plt.pyplot.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')

    plt.pyplot.title('Polynomial Fit with Matplotlib for Price ~ Length')

    ax = plt.pyplot.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.pyplot.gcf()

    plt.pyplot.xlabel(Name)

    plt.pyplot.ylabel('Price of Cars')



    plt.pyplot.show()

    plt.pyplot.close()
x = df_auto['highway-mpg']

y = df_auto['price']

# Here we use a polynomial of the 3rd order (cubic) 

f = np.polyfit(x, y, 3)

p = np.poly1d(f)

print(p)
PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

#We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.
from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2)

pr

Z_pr=pr.fit_transform(Z)

Z.shape

Z_pr.shape
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe=Pipeline(Input)

pipe

pipe.fit(Z,y)

ypipe=pipe.predict(Z)

ypipe[0:4]
#Model 1: Simple Linear Regression

lm.fit(X, Y)

# Find the R^2

print('The R-square is: ', lm.score(X, Y))

#~ 49.659% of the variation of the price is explained by this simple linear model "horsepower_fit"

Yhat=lm.predict(X)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df_auto['price'], Yhat)

print('The mean square error of price and predicted value is: ', mse)
# Model 2: Multiple Linear Regression 

lm.fit(Z, df_auto['price'])

# Find the R^2

print('The R-square is: ', lm.score(Z, df_auto['price']))

Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ', \

      mean_squared_error(df_auto['price'], Y_predict_multifit))
#Model 3: Polynomial Fit

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))

print('The R-square value is: ', r_squared)

mean_squared_error(df_auto['price'], p(x))

#We can say that ~ 67.419 % of the variation of price is explained by this polynomial fit
#Prediction

new_input=np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X, Y)

lm

yhat=lm.predict(new_input)

yhat[0:5]

plt.pyplot.plot(new_input, yhat)

plt.pyplot.show()
'''Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)

Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and or even act as noise.

As a result, you should always check the MSE and R^2.



So to be able to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.



MSEThe MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.

R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).

This R-squared in combination with the MSE show that MLR seems like the better model fit in this case, compared to SLR.



Simple Linear Model (SLR) vs Polynomial Fit

MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.

R-squared: The R-squared for the Polyfit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.

Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting Price with Highway-mpg as a predictor variable.



Multiple Linear Regression (MLR) vs Polynomial Fit

MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.

R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.

Conclusion:

Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price.'''