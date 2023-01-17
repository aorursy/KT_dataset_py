# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing the necessary libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import functools 



%matplotlib inline
crime_rate_df = pd.read_csv ("../input/europe-datasets/crime_2016.csv")

gdp_df = pd.read_csv ('../input/europe-datasets/gdp_2016.csv')

job_satisfaction_df = pd.read_csv ('../input/europe-datasets/job_satisfaction_2013.csv')

leisure_satisfaction_df = pd.read_csv ('../input/europe-datasets/leisure_satisfaction_2013.csv')

life_expectancy_df = pd.read_csv ('../input/europe-datasets/life_expectancy_2016.csv')

life_satisfaction_df = pd.read_csv ('../input/europe-datasets/life_satisfaction_2013.csv')

median_income_df = pd.read_csv ('../input/europe-datasets/median_income_2016.csv')

population_df = pd.read_csv ('../input/europe-datasets/population_2011.csv')

trust_in_politics_df = pd.read_csv ('../input/europe-datasets/trust_in_politics_2013.csv')

unemployment_df = pd.read_csv ('../input/europe-datasets/unemployment_2016.csv')

weather_df = pd.read_csv ('../input/europe-datasets/weather.csv')

work_hours_df = pd.read_csv ('../input/europe-datasets/work_hours_2016.csv')
#dropping the null columns and checking the content of the weather dataframe



filtered_weather = weather_df.dropna(axis='columns', how='all')

filtered_weather.head()
#creating a dataframe from the downloaded files joining with respect to each country and checking its content



data_dfs=[crime_rate_df,gdp_df,job_satisfaction_df,leisure_satisfaction_df, life_expectancy_df, life_satisfaction_df, 

          median_income_df, population_df, trust_in_politics_df, unemployment_df, filtered_weather, work_hours_df] 

data_df = functools.reduce(lambda left,right: pd.merge(left,right,on='country'), data_dfs)

data_df.head()
stats_df = data_df.drop(columns=['prct_job_satis_med','prct_job_satis_low','prct_leisure_satis_med',

                                 'prct_leisure_satis_low', 'prct_life_satis_med','prct_life_satis_low','avg_high_temp',

                                'avg_low_temp'])

stats_df.head()
# Looking at the attributes of the data set and scatter plots across the dataframe. 

# Prior to predictions, scatter plots are used for comparison analysis. 



sns.pairplot(stats_df)
# Consider two features, average temperature and unemployment rate. 

# The scatter plot might suggest that there is a linear relationship between the two features, 

# as the average temperature rises, so does the unemployment rate. 



sns.jointplot(x='avg_temp',y='unemp_rate',data=stats_df)
# constructing the regression line of the two features

# at higher average temperature, the scatter plot observed a few outliers that appeared to have no such linear relationship.



sns.lmplot(x='avg_temp',y='unemp_rate',data=stats_df)
# finding the correlation coefficient of about 0.6

# this reveales that about 36% of data points are explained by the least square regression line. 



stats_df[['avg_temp','unemp_rate']].corr()
# looking at the countries with the highest average temperature



stats_df[stats_df['avg_temp']>58]
# creating a dataframe that disregards the warmest 5 countries



unemp_rate_df=stats_df[stats_df['avg_temp']<58]
# looking at the information of the dataframe



unemp_rate_df.info()
# taking out the 5 warmest countries does not improve the correlation of the remaining data points

# the correlation between the remaining data points is now approximately 0.52

# a better correlation might be revealed if only the 3 outliers will be removed, 

# but that would temper too much with statistical analysis



unemp_rate_df[['avg_temp','unemp_rate']].corr()
# Now considering the feature gdp. 

# correlation coefficient between gdp and the political trust rating is approximately 0, 

# as in, there is no apperent correlation between the two features

 

stats_df[['political_trust_rating','gdp']].corr()
# looking at the scatter plot between gdp and political trust rating reveals five outliers 



sns.lmplot(x='political_trust_rating',y='gdp',data=stats_df)
# looking at the scatter plot between gdp and the unemployment rate also reveals outliers



sns.lmplot(x='unemp_rate',y='gdp',data=stats_df)
# isolating the outliers with the highest gdp and ordering in descending order by gdp



stats_df[stats_df['gdp']>1000000].sort_values(by=['gdp'],ascending=False)
# isolating the outliers with the highest total population and ordering in descending by gdp

# the top five countries with the highest gdp are the same five countries with the highest total number of population, 

# in the same order. 



stats_df[stats_df['gdp']>1000000]

stats_df[stats_df['total_pop']>40000000].sort_values(by=['gdp'],ascending=False)
# dataframe is created to take a closer look at how life satisfaction compares to job satisfaction, 

# leisure satisfaction, median income and life expectancy at birth. 



happy_df=stats_df[['prct_job_satis_high','prct_leisure_satis_high','prct_life_satis_high','median_income','life_expect']]
# scatter plots are created across the dataframe



sns.pairplot(happy_df)
happy_df.corr()
# identifying the countries that have the highest percent of people with an opinion of high life satisfaction



stats_df[stats_df['prct_life_satis_high']>38]
# identifying the countries with greater life expectancy sorted by highest life satisfaction

# the four countries with high life satisfaction are also on the list of long life.



stats_df[stats_df['life_expect']>80].sort_values(by=['prct_life_satis_high'],ascending=False).head(4)
# using machine learning to fit the dataframe 'stats_df' to a linear regression model



X=stats_df.drop(['country','prct_life_satis_high'],axis=1)

y=stats_df['prct_life_satis_high']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)



from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train, y_train)
#coefficients and the intercept of the regresion line are found by the regression model



print (lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,columns=['Coefficients'])

coeff_df
# scatter plot of the points used to test the model

# linear relationship should be observed



predictions=lm.predict(X_test)

plt.scatter(y_test, predictions)
# distribution of the residuals is observed to be approximately normal



sns.distplot((y_test-predictions))
# statistical errors of the linear regression model



from sklearn import metrics

print ('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))

print ('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))

print ('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# comparing the actual percentage to the predicted



comp_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions, 'Residuals': (y_test-predictions)})

comp_df
# Running the linear regression model using less features proved less effective with a much higher statistical error



X=stats_df[['prct_job_satis_high','prct_leisure_satis_high','median_income','life_expect']]

y=stats_df['prct_life_satis_high']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)



from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train, y_train)
print (lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,columns=['Coefficient'])

coeff_df
sns.distplot((y_test-predictions))
from sklearn import metrics

print ('MAE:', metrics.mean_absolute_error(y_test, predictions))

print ('MSE:', metrics.mean_squared_error(y_test, predictions))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# constructing a linear regression between high life satisfaction and high job satisfaction 



from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



X=stats_df[['prct_leisure_satis_high']]

y=stats_df['prct_job_satis_high']



lr = linear_model.LinearRegression()



lr.fit(X,y)



print('Coefficient: \n', lr.coef_)

print ('Intercept: \n', lr.intercept_)

lr.coef_*20.4+lr.intercept_