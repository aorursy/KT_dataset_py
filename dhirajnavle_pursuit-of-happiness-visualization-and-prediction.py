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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
data_2015 = pd.read_csv("/kaggle/input/world-happiness-report/2015.csv", delimiter=',')

data_2015.head()
data_2015.describe()
data_2015.info()
heatmap = sns.heatmap(

    data_2015.corr(), 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

heatmap.set_xticklabels(

    heatmap.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
sns.pairplot(data_2015)

plt.show()
sns.jointplot("Economy (GDP per Capita)", "Happiness Score", data=data_2015,

                  kind="reg", truncate=False,

                  xlim=(0, 2), ylim=(0, 10),

                  color="m", height=7)
sns.jointplot("Health (Life Expectancy)", "Happiness Score", data=data_2015,

                  kind="reg", truncate=False,

                  xlim=(0, 2), ylim=(0, 10),

                  color="m", height=7)
sns.jointplot("Freedom", "Happiness Score", data=data_2015,

                  kind="reg", truncate=False,

                  xlim=(0, 1), ylim=(0, 10),

                  color="m", height=7)
sns.jointplot("Trust (Government Corruption)", "Happiness Score", data=data_2015,

                  kind="reg", truncate=False,

                  xlim=(0, 1), ylim=(0, 10),

                  color="m", height=7)
avg_happiness = data_2015.groupby("Region").mean()

print(avg_happiness)
avg_happiness[["Happiness Score"]].plot.bar()
sns.lmplot(x="Health (Life Expectancy)", y="Happiness Score", col="Region", data=data_2015, col_wrap=4)
sns.lmplot(x="Economy (GDP per Capita)", y="Happiness Score", col="Region", data=data_2015, col_wrap=4)
sns.lmplot(x="Family", y="Happiness Score", col="Region", data=data_2015, col_wrap=4)
sns.lmplot(x="Freedom", y="Happiness Score", col="Region", data=data_2015, col_wrap=4)
sns.lmplot(x="Trust (Government Corruption)", y="Happiness Score", col="Region", data=data_2015, col_wrap=4)
data_2015['Year'] = '2015'

data2015 = data_2015[['Country','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity','Year']]

data2015.info()
data_2016 = pd.read_csv("/kaggle/input/world-happiness-report/2016.csv", delimiter=',')

data_2016['Year'] = 2016

data2016 = data_2016[['Country','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity','Year']]

data2016.info()
data_2017 = pd.read_csv("/kaggle/input/world-happiness-report/2017.csv", delimiter=',')

data_2017.rename(columns={"Happiness.Score": "Happiness Score", "Economy..GDP.per.Capita.": "Economy (GDP per Capita)", 

                          "Health..Life.Expectancy.": "Health (Life Expectancy)","Trust..Government.Corruption.": "Trust (Government Corruption)"},

                    inplace = True)

data_2017['Year'] = 2017

data2017 = data_2017[['Country','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity','Year']]

data2017.info()
data_2018 = pd.read_csv("/kaggle/input/world-happiness-report/2018.csv", delimiter=',')

data_2018.rename(columns={"Country or region": "Country", "Score": "Happiness Score", "GDP per capita": "Economy (GDP per Capita)",

                         "Healthy life expectancy": "Health (Life Expectancy)","Freedom to make life choices": "Freedom",

                         "Perceptions of corruption": "Trust (Government Corruption)"}, inplace = True)

data_2018['Year'] = 2018

data2018 = data_2018[['Country','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity','Year']]

data2018.info()
is_NaN = data2018.isnull()

row_has_NaN = is_NaN. any(axis=1)

rows_with_NaN = data2018[row_has_NaN]

print(rows_with_NaN)
corruption_15 = data2015.loc[data2015['Country'] == 'United Arab Emirates']

corruption_15 = corruption_15['Trust (Government Corruption)'].values.item()

corruption_15
corruption_16 = data2016.loc[data2016['Country'] == 'United Arab Emirates']

corruption_16 = corruption_16['Trust (Government Corruption)'].values.item()

corruption_16
corruption_17 = data2017.loc[data2017['Country'] == 'United Arab Emirates']

corruption_17 = corruption_17['Trust (Government Corruption)'].values.item()

corruption_17
corruption_mean = np.mean([corruption_15,corruption_16,corruption_17])

corruption_mean
data2018.loc[data2018.Country == 'United Arab Emirates', 'Trust (Government Corruption)'] = corruption_mean

data2018.info()
data_2019 = pd.read_csv("/kaggle/input/world-happiness-report/2019.csv", delimiter=',')

data_2019['Year'] = 2019

data_2019.rename(columns={"Country or region": "Country","Score": "Happiness Score","GDP per capita": "Economy (GDP per Capita)",

                         "Healthy life expectancy": "Health (Life Expectancy)", "Freedom to make life choices": "Freedom",

                         "Perceptions of corruption": "Trust (Government Corruption)"}, inplace = True)

data2019 = data_2019[['Country','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity','Year']]

data2019.info()
merged_happiness  = pd.concat([data2015, data2016, data2017, data2018, data2019], ignore_index = True)

merged_happiness.head()
merged_happiness.info()
merged_happiness.describe()
sns.swarmplot(x="Year", y="Happiness Score",data=merged_happiness)
#sns.swarmplot(x="Year", y="Happiness Score", hue="Country", data=merged_happiness)
heatmap = sns.heatmap(

    merged_happiness.corr(), 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

heatmap.set_xticklabels(

    heatmap.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
X = merged_happiness[['Economy (GDP per Capita)','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity']]

y = merged_happiness[['Happiness Score']]

X.round(3)

y.round(3)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

linear_pred = lin_model.predict(X_test)

linear_error = mean_squared_error(y_test, linear_pred)
from sklearn.svm import LinearSVR

svr_model = LinearSVR()

svr_model.fit(X_train, y_train.values.ravel())

svr_pred = svr_model.predict(X_test)

svr_error = mean_squared_error(y_test, svr_pred)
from sklearn.ensemble import RandomForestRegressor

rm_model = RandomForestRegressor(n_estimators=500, max_leaf_nodes=20, n_jobs=-1)

rm_model.fit(X_train, y_train.values.ravel())

rm_pred = rm_model.predict(X_test)

rm_error = mean_squared_error(y_test, rm_pred)
print("The Mean Squared Error For Linear Regression is: {}".format(linear_error))
print("The Mean Squared Error For Linear SVR is: {}".format(svr_error))
print("The Mean Squared Error For RandomForestRegressor is: {}".format(rm_error))
data_2020 = pd.read_csv("/kaggle/input/world-happiness-report/2020.csv", delimiter=',')

data_2020.head()
data_2020.drop(columns=['Generosity'] , inplace = True)

data_2020.head()
data_2020.rename(columns={'Ladder score':'Happiness Score','Explained by: Log GDP per capita': 'Economy (GDP per Capita)',

                         'Explained by: Healthy life expectancy':'Health (Life Expectancy)',

                          'Explained by: Freedom to make life choices':'Freedom',

                         'Explained by: Perceptions of corruption':'Trust (Government Corruption)',

                          'Explained by: Generosity': 'Generosity'}, inplace=True)

data2020 = data_2020[['Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity']]

data2020.head()
X2020 = data2020[['Economy (GDP per Capita)','Health (Life Expectancy)',

                      'Freedom','Trust (Government Corruption)','Generosity']]

y2020 = data2020['Happiness Score']
X2020.head()
pred2020 = rm_model.predict(X2020)
maen_2020 = mean_squared_error(y2020, pred2020)
print("The Mean Squared Error is: {}".format(maen_2020))
plt.scatter(y2020, pred2020)

plt.plot([pred2020.min(), pred2020.max()], [pred2020.min(), pred2020.max()], 'k--', lw=4)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('RandomForestRegressor')

plt.show()