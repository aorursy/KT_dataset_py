import numpy as np

import pandas as pd



# List all files under the current input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load the data for 2020, leaving some unwanted columns out

wh_2020 = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv', usecols = range(12))
# Preview the data

wh_2020.head()
print("Our data has {} rows (observations/countries) and {} columns.".format(wh_2020.shape[0], wh_2020.shape[1]))
col_names_dict = {'Country name':'Country', 'Regional indicator':'Region', 'Ladder score': 'Ladder',

                  'Standard error of ladder score':'Standard Error', 'Logged GDP per capita':'Logged GDPPC',

                  'Social support':'Social Support', 'Healthy life expectancy':'Life Expectancy',

                  'Freedom to make life choices':'Freedom', 'Perceptions of corruption': 'Corruption'}



wh_2020.rename(columns = col_names_dict, inplace = True)
# Check for any missing values in the data

wh_2020.isnull().sum()
# Add a 'Rank' column to our data (luckily for us, the rows are already ordered from happiest to unhappiest)

wh_2020['Rank'] = range(1, 154)
quartile_index = np.percentile(wh_2020['Rank'], [25, 50, 75])

quartiles = pd.Series(wh_2020['Rank'].map(lambda x:(np.searchsorted(quartile_index, x) + 1)), name = 'Quartile')

wh_2020 = pd.concat([wh_2020, quartiles], axis = 1)    
# Check our updated data with the new 'Rank' and 'Quartile' columns

wh_2020.head()
# Set font sizes for all of our plots

plt.rc('font', size = 14)

plt.rc('axes', labelsize = 16)

plt.rc('legend', fontsize = 18)

plt.rc('axes', titlesize = 24)

plt.rc('figure', titlesize = 24)
# Set style

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize = (18, 14))

ax = plt.axes()



countplot = sns.countplot('Region', data = wh_2020, saturation = 0.8, palette = 'tab10')

countplot.set_xticklabels(countplot.get_xticklabels(), rotation = 90)

countplot.set_title("Countplot by Region", y = 1.05);
fig = plt.figure(figsize = (18, 14))

ax = plt.axes()



stacked_countplot = sns.countplot('Region', data = wh_2020, hue = 'Quartile')

stacked_countplot.set_xticklabels(countplot.get_xticklabels(), rotation = 90)

stacked_countplot.set_title("Countplot of Quartiles for Each Region", y = 1.05);

ax.legend(loc = "upper left", title = 'Quartile', title_fontsize = 18);
print("Table of Average Rank for Each Region:\n")

print(wh_2020.groupby('Region')['Rank'].agg('mean'))
# Gather columns corresponding to the six measured values (Logged GDP per capita, social support, etc.)

feature_cols = ['Logged GDPPC', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
df = pd.concat([wh_2020['Ladder'], wh_2020[feature_cols]], axis = 1)



fig = plt.figure(figsize = (13, 10))

plt.style.use('seaborn-white')



plt.matshow(df.corr(), fignum = fig.number, cmap = 'viridis')

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)



cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)



plt.title('Correlation Matrix', fontsize = 24, y = 1.2);
pairplot = sns.pairplot(wh_2020, hue = 'Quartile', vars = feature_cols, corner = False)

pairplot.fig.suptitle("Pairplot of the 6 Happiness Metrics", fontsize = 24, y = 1.05);
fig, axes = plt.subplots(2, 3, figsize = (20, 12))



for i, ax in enumerate(axes.flat):

    ax.plot(wh_2020['Rank'], wh_2020[feature_cols[i]], color = 'red')

    ax.set_title(feature_cols[i] + ' by Rank', fontsize = 18)

    ax.set_xlim(153, 1)

    ax.axis('tight')
fig = plt.figure(figsize = (15, 12))

ax = plt.axes()



scatter = ax.scatter(wh_2020['Logged GDPPC'], wh_2020['Social Support'], alpha = 0.4, s = wh_2020['Life Expectancy']**1.5, c = wh_2020['Quartile'], cmap = 'viridis')

ax.set(xlabel = 'Logged GDPPC', ylabel = 'Social Support')

legend = ax.legend(*scatter.legend_elements(prop = 'colors', size = 16),

                    loc = "lower right", title = "Quartile", title_fontsize = 18)

ax.add_artist(legend);
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Designate features and target variable

y = wh_2020['Ladder']

X = wh_2020[feature_cols]
# Split data into training and validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)
# Fit a linear regression model to the data

lin_reg_model = LinearRegression()



lin_reg_model.fit(X_train, y_train)
# Use the fitted model to make predictions

preds = lin_reg_model.predict(X_test)
# Find the average error of our predictions for the validation data

mean_squared_error(preds, y_test)    
# Another metric for evaluating error

mean_absolute_error(preds, y_test)    