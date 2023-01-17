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
happy = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
# Columns and the main countries

happy.head()
# Data information per column

happy.info()
# Describing data

happy.describe()
# Which is the country with the highest -Healthy life expectancy-?

happy[happy['Healthy life expectancy']==happy['Healthy life expectancy'].max()]['Country or region']
# Which is the country with the highest -Freedom to make life choices-?

happy[happy['Freedom to make life choices']==happy['Freedom to make life choices'].max()]['Country or region']
# Which country has the biggest -GDP per capita-?

happy[happy['GDP per capita']==happy['GDP per capita'].max()]['Country or region']
# Top 10 countries with bigger GDP per capita

happy[['Country or region', 'GDP per capita']].sort_values(by = 'GDP per capita',

                                                ascending = False).head(10)
# Which country has the smaller GDP per capita

happy[happy['GDP per capita']==happy['GDP per capita'].min()]['Country or region']
# Top 10 Most Generous Countries

happy[['Country or region', 'Generosity']].sort_values(by = 'Generosity',

                                                ascending = False).head(10)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Relation between GDP and Social support

sns.jointplot(x='GDP per capita', y='Social support', kind='reg', data=happy)
# Relation between GDP and Generosity

sns.jointplot(x='GDP per capita', y='Generosity', kind='reg', data=happy)
# Relation between GDP and Perception of corruption

sns.jointplot(x='GDP per capita', y='Perceptions of corruption', kind='reg', data=happy)
# Relation between GDP and Healthy life expectancy

sns.jointplot(x='GDP per capita', y='Healthy life expectancy', kind='reg', data=happy)
# Relation between GDP and Social support (In blue Argentina)

g = sns.jointplot(x='Social support', y='Healthy life expectancy', data=happy, color='grey')

df = happy.loc[happy['Country or region'] == 'Argentina']

g.ax_joint.scatter(x = df['Social support'], y=df['Healthy life expectancy'], color = 'blue', s=100)
# A fast view of some special data.

ppf = happy[['GDP per capita', 'Generosity', 'Healthy life expectancy', 'Perceptions of corruption']]

sns.pairplot(ppf, kind='reg')
# In the following plot we can see the corralation between all columns

sns.heatmap(happy.corr(), annot=True, cmap="YlGnBu")
# Score value distribution

sns.distplot(happy['Score'], bins=20)
happy.columns
# Assigning "y" to score value

y = happy['Score']
# Chossing some columns for the model

X = happy[['GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']]
# Create a test data

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Use random_state for reproducibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
# Linear regression object

lm = LinearRegression()
# Fit the training data

lm.fit(X_train,y_train)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])

cdf.head()
# We give to the model data that never saw (in our case X_text)

# We train our model with X_train

predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# Metrics

from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# Distributions of residuals

sns.distplot((y_test-predictions),bins=30)