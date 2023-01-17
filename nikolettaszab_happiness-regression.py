import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



fif = pd.read_csv('../input/world-happiness/2015.csv')

six = pd.read_csv('../input/world-happiness/2016.csv')

sev = pd.read_csv('../input/world-happiness/2017.csv')

eig = pd.read_csv('../input/world-happiness/2018.csv')

nin = pd.read_csv('../input/world-happiness/2019.csv')
nin.rename(columns={'Country or region': 'Country',

                    'Overall rank': 'Overall rank 2019',

                    'Score': 'Score 2019', 

                    'GDP per capita': 'GDP per capita 2019',

                    'Social support': 'Social support 2019', 

                    'Healthy life expectancy': 'Healthy life expectancy 2019', 

                    'Freedom to make life choices': 'Freedom to make life choices 2019', 

                    'Generosity': 'Generosity 2019',

                    'Perceptions of corruption': 'Perceptions of corruption 2019'}, inplace=True)
eig.rename(columns={'Country or region': 'Country',

                    'Overall rank': 'Overall rank 2018',

                    'Score': 'Score 2018',

                    'GDP per capita': 'GDP per capita 2018',

                    'Social support': 'Social support 2018',

                    'Healthy life expectancy': 'Healthy life expectancy 2018',

                    'Freedom to make life choices': 'Freedom to make life choices 2018',

                    'Generosity': 'Generosity 2018',

                    'Perceptions of corruption': 'Perceptions of corruption 2018'}, inplace=True)
sev.rename(columns={'Happiness.Rank': 'Overall Rank 2017',

                    'Happiness.Score': 'Score 2017', 

                    'Whisker.high': 'Whisker high 2017', 

                    'Whisker.low':'Whisker low 2017', 

                    'Economy..GDP.per.Capita.': 'GDP per capita 2017', 

                    'Family':'Family 2017', 

                    'Health..Life.Expectancy.':'Healthy life expectancy 2017',

                    'Freedom':'Freedom 2017', 

                    'Generosity':'Generosity 2017',

                    'Trust..Government.Corruption.': 'Perceptions of corruption 2017', 

                    'Dystopia.Residual':'Dystopia Residual 2017'}, inplace=True)
six.rename(columns={'Happiness Rank':'Happiness Rank 2016',

                    'Happiness Score ': 'Score 2016',

                    'Lower Confidence Interval ': 'Lower Confidence Interval 2016',

                    'Upper Confidence Interval' : 'Upper Confidence Interval 2016',

                    'Economy (GDP per Capita)' : 'GDP per Capita 2016',

                    'Family':'Family 2016',

                    'Health (Life Expectancy)': 'Healthy life expectancy 2016',

                    'Freedom' : 'Freedom 2016',

                    'Trust (Government Corruption)' : 'Perceptions of corruption 2016',

                    'Generosity' : 'Generosity 2016',

                    'Dystopia Residual': 'Dystopia Residual 2016'}, inplace=True)
fif.rename(columns={'Happiness Rank': 'Happiness Rank 2015',

                    'Happiness Score': 'Score 2015',

                    'Standard Error': 'Standard Error 2015',

                    'Economy (GDP per Capita)': 'GDP per Capita 2015',

                    'Family' : 'Family 2015',

                    'Health (Life Expectancy)': 'Healthy life expectancy 2015',

                    'Freedom': 'Freedom 2015',

                    'Trust (Government Corruption)': 'Perceptions of government corruption 2015',

                    'Generosity ':'Generosity 2015',

                    'Dystopia Residual': 'Dystopia Residual 2015'}, inplace=True)
one = pd.merge(fif, six, on='Country')
one.columns
two = pd.merge(sev, eig, on='Country')

three = pd.merge(one, two, on='Country')

data = pd.merge(three, nin, on='Country')
data.set_index('Country')
data.info()
data['Average Score'] = data[['Score 2015', 

                              'Happiness Score', 

                              'Score 2017', 

                              'Score 2018', 

                              'Score 2019']].mean(numeric_only=True, axis=1)
data['Average GDP'] = data[['GDP per capita 2019', 

                              'GDP per capita 2018',

                              'GDP per capita 2017',

                              'GDP per Capita 2016',

                              'GDP per Capita 2015']].mean(numeric_only=True, axis=1)
data['Average life expextancy'] = data[['Healthy life expectancy 2015',

                                       'Healthy life expectancy 2016',

                                       'Healthy life expectancy 2017',

                                       'Healthy life expectancy 2018',

                                       'Healthy life expectancy 2019']].mean(numeric_only=True, axis=1)
data['Average social support'] = data[['Social support 2019',

                                      'Social support 2018',

                                      'Family 2017',

                                      'Family 2016',

                                      'Freedom 2015']].mean(numeric_only=True, axis=1)
data.head()
gdp_df = data[['Country','Average Score','GDP per capita 2019','GDP per capita 2018',

               'GDP per capita 2017','GDP per Capita 2016','GDP per Capita 2015','Average GDP']]
gdp_df.head()
from scipy import stats

pearson_coef, p_value = stats.pearsonr(gdp_df['Average GDP'], gdp_df['Average Score'])

print(pearson_coef, p_value)
import seaborn as sns

corr = gdp_df.corr()

sns.heatmap(corr, cmap='BuPu', square=True)
sns.regplot(x='Average Score', y = 'Average GDP', data = gdp_df, scatter_kws = {'color': 'purple', 'alpha': 0.3}, line_kws = {'color': 'teal', 'alpha': 0.3, 'lw':6})

sns.set_style('whitegrid')

plt.ylim(0,)
sns.residplot(gdp_df['Average Score'], gdp_df['Average GDP'], scatter_kws = {'color': 'Maroon', 'alpha': 0.3})
sns.jointplot(x ='Average Score', y='Average GDP', data = gdp_df, color='LightCoral')
gdp_df.set_index('Country')

y_data = gdp_df['Average Score']

x_data = gdp_df.drop(['Average Score','Country'], axis=1)
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train[['Average GDP']], y_train)

lm.score(x_train[['Average GDP']], y_train) #r2 score for training dataset
lm.score(x_test[['Average GDP']], y_test) #r2 score for test dataset will need to do mse as well
from sklearn.model_selection import cross_val_score

rcross = cross_val_score(lm, x_data[['Average GDP']], y_data, cv=10)

rcross
print("The mean of the folds are", rcross.mean(), "and the standard deviation is" , rcross.std())
from sklearn.model_selection import cross_val_predict

yhat = cross_val_predict(lm, x_data[['Average GDP']], y_data, cv=10)

yhat[0:5]
import seaborn as sns

ax1 = sns.distplot(y_data, hist=False, color="Sienna", label="Actual Avg. Happiness")

sns.distplot(yhat, hist=False, color="DarkCyan", label="Fitted Happiness" , ax=ax1)



plt.show()
inpGDP = [[1.94862]]

yhat2 = lm.predict(inpGDP)

print('The happiness score for a coutry with this GDP is predicted to be:', yhat2)