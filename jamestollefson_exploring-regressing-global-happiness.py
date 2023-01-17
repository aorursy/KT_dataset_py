# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data2015 = pd.read_csv('../input/2015.csv')

data2016 = pd.read_csv('../input/2016.csv')



data2015['Year'] = 2015

data2016['Year'] = 2016

data = pd.concat([data2015, data2016])

data = data.reset_index()

data.head()

data.tail()

data.info()
i = data['Lower Confidence Interval'][data['Year'] == 2015].isnull()

assert i.any() == True

print(len(i))

l = data['Lower Confidence Interval'][data['Year'] == 2016].isnull()

assert l.any() == False

print(len(l))

b = data['Standard Error'][data['Year'] == 2015].isnull()

assert b.any() == False

print(len(b))
midpoint = (data2016['Upper Confidence Interval'] - data2016['Lower Confidence Interval']) / 2

data2016['Sample Mean'] = data2016['Lower Confidence Interval'] + midpoint

data2016['Standard Error'] = midpoint / 1.96



del data2016['Sample Mean']

del data2016['Upper Confidence Interval']

del data2016['Lower Confidence Interval']



data = pd.concat([data2015, data2016])

data = data.reset_index()

data.info()
print(data.head())
_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Economy (GDP per Capita)'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Economy (GDP per Capita)'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Economy (GDP per Capita)')

_ = plt.title('Happiness vs. GDP')

plt.show()



_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Family'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Family'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Family')

_ = plt.title('Happiness vs. Family')

plt.show()



_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Freedom'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Freedom'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Freedom')

_ = plt.title('Happiness vs. Freedom')

plt.show()



_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Generosity'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Generosity'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Generosity')

_ = plt.title('Happiness vs. Generosity')

plt.show()



_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Health (Life Expectancy)'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Health (Life Expectancy)'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Health (Life Expectancy)')

_ = plt.title('Happiness vs. Health (Life Expectancy)')

plt.show()



_ = sns.kdeplot(data=data['Happiness Score'], data2=data['Trust (Government Corruption)'], shade=True)

_ = plt.scatter(x=data['Happiness Score'], y=data['Trust (Government Corruption)'], alpha=0.2, color='green')

_ = plt.xlabel('Happiness Score')

_ = plt.ylabel('Trust (Government Corruption)')

_ = plt.title('Happiness vs. Trust (Government Corruption)')

plt.show()
cols = ['Dystopia Residual', 'Economy (GDP per Capita)', 'Family', 'Freedom', 'Generosity', 'Happiness Score', 'Health (Life Expectancy)', 'Trust (Government Corruption)']



heatmap = data[cols]

corr = heatmap.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="YlGnBu", annot=True)

plt.show()
from pandas.tools.plotting import lag_plot



plt.figure(1)

lag_plot(data['Freedom'])

plt.title('Freedom')



plt.figure(2)

lag_plot(data['Family'])

_ = plt.title('Family')



plt.figure(3)

lag_plot(data['Dystopia Residual'])

_ = plt.title('Dystopia Residual')



plt.figure(4)

lag_plot(data['Economy (GDP per Capita)'])

_ = plt.title('Economy (GDP per Capita)')



plt.figure(5)

lag_plot(data['Generosity'])

_ = plt.title('Generosity')



plt.figure(6)

lag_plot(data['Happiness Score'])

_ = plt.title('Happiness Score')



plt.figure(7)

lag_plot(data['Health (Life Expectancy)'])

_ = plt.title('Health (Life Expectancy)')



plt.figure(8)

lag_plot(data['Trust (Government Corruption)'])

_ = plt.title('Trust (Government Corruption)')



plt.tight_layout()

plt.show()
from pandas.tools.plotting import radviz

pd.options.mode.chained_assignment = None



del heatmap['Happiness Score']

heatmap['Region'] = data['Region']



plt.figure()

radviz(heatmap, 'Region')

plt.legend(bbox_to_anchor=(1,1))

plt.show()
del heatmap['Region']

heatmap = heatmap ** 6

heatmap['Region'] = data['Region']



plt.figure()

radviz(heatmap, 'Region')

plt.legend(bbox_to_anchor=(1,1))

plt.show()
order =['Sub-Saharan Africa', 'Southern Asia', 'Southeastern Asia', 'Eastern Asia', 'Australia and New Zealand', 'Central and Eastern Europe', 'Western Europe', 'Latin America and Caribbean', 'North America']





_ = sns.barplot(x=data['Region'], y=data['Happiness Score'], order=order, hue=data['Year'], hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Average Happiness Score 2015-2016')

_ = plt.title('Happiness by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Economy (GDP per Capita)', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Average GDP per Capita 2015-2016')

_ = plt.title('GDP per Capita by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Freedom', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Freedom 2015-2016')

_ = plt.title('Freedom by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Family', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Family 2015-2016')

_ = plt.title('Family by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Health (Life Expectancy)', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Health (Life Expectancy) 2015-2016')

_ = plt.title('Health (Life Expectancy) by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Trust (Government Corruption)', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Trust (Government Corruption) 2015-2016')

_ = plt.title('Trust (Government Corruption) by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Generosity', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Generosity 2015-2016')

_ = plt.title('Generosity by Region 2015-2016')

plt.show()



_ = sns.barplot(x='Region', y='Dystopia Residual', data=data, hue='Year', order=order, hue_order=[2015, 2016])

_ = plt.xticks(rotation=75)

_ = plt.xlabel('Regions')

_ = plt.ylabel('Dystopia Residual 2015-2016')

_ = plt.title('Dystopia Residual by Region 2015-2016')

plt.show()











from sklearn.preprocessing import LabelEncoder



data.Region = LabelEncoder().fit_transform(data.Region)

data.Country = LabelEncoder().fit_transform(data.Country)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import sklearn.metrics as m



target = data['Happiness Score']

features = data[['Country', 'Dystopia Residual', 'Economy (GDP per Capita)', 'Family', 'Freedom', 'Generosity', 'Health (Life Expectancy)', 'Region', 'Trust (Government Corruption)']]



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)



model = LinearRegression().fit(X_train, y_train)

predictions = model.predict(X_test)

mae = m.mean_absolute_error(y_test, predictions)

mse = m.mean_squared_error(y_test, predictions)

print(mae)

print(mse)



_ = plt.hist(predictions, alpha=0.5, color='red', cumulative=True, normed=True, bins=len(predictions), histtype='stepfilled', stacked=True)

_ = plt.hist(y_test, alpha=0.5, color='blue', cumulative=True, normed=True, bins=len(predictions), histtype='stepfilled', stacked=True)

plt.show()
