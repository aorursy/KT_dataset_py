# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt

import matplotlib.pyplot as plt 

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/2017.csv')

df.head()
#List of columns in the dataset

df.columns
print(df.isnull().any())
#First checking the correlation between the two values

print('Correlation between Happiness Score and GDP:',df['Happiness.Score'].corr(df['Economy..GDP.per.Capita.']))
happiness_score = df['Happiness.Score']

Economy_GDP = df['Economy..GDP.per.Capita.']



plt.scatter(happiness_score,Economy_GDP)

plt.title('Correlation between Happiness and GDP')

plt.xlabel('Happiness Score')

plt.ylabel('Economy GDP')

plt.show()
freedom = df['Freedom']

Life_expectacny = df['Health..Life.Expectancy.']

print('Correlation between Freedom and Life Expectancy:',freedom.corr(Life_expectacny))
corruption_data = df['Trust..Government.Corruption.']

happiness_score = df['Happiness.Score']

ax1 = plt.subplot2grid((1,1),(0,0))

ax1.scatter(corruption_data,happiness_score,label='Corruption VS Happiness')

ax1.plot(corruption_data,happiness_score,label='Corruption VS Happiness')

plt.xlabel('Corruption')

plt.ylabel('HAppiness')

plt.plot()

plt.show()
df_2016 = pd.read_csv('../input/2016.csv')

#df_2016.head()

df_2016.columns


# USING GDP,Life Expectancy,Corruption as the features and Happiness SCore as Label we train linear Regression Model

features = ['Trust (Government Corruption)','Health (Life Expectancy)','Economy (GDP per Capita)','Country']

temp_dataframe = df_2016[features]

temp_dataframe = temp_dataframe.sort_values('Country')

#temp_dataframe



del temp_dataframe['Country']

X = temp_dataframe

X

# Label for Regression MOdel TRaining

label = 'Happiness Score'

y = df_2016[label]
regressor = LinearRegression()

model = regressor.fit(X,y)

print(model) #this shows model is an object of type LinearRegression 
#DataFrame of 2017 doesn't have Region Column so we drop it from DataFrame of 2016 also

del df_2016['Region']

df.columns = df_2016.columns

df
temp_dFrame = df.sort_values('Country')

temp_dFrame
new_features = ['Trust (Government Corruption)','Health (Life Expectancy)','Economy (GDP per Capita)']

train_x = temp_dFrame[new_features]

train_y = temp_dFrame[label]

prediction = regressor.predict(train_x)
temp_df = pd.DataFrame()

temp_df['Original Happiness Score'] = train_y

temp_df['Predicted Happiness Score'] = prediction 

temp_df



temp_df['Country'] = df_2016['Country']
mean_squared_error_value = sqrt(mean_squared_error(y_true=train_y,y_pred=prediction))

mean_squared_error_value