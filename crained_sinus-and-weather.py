# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pylab as plt

import os

import seaborn as sns

sns.set_style('whitegrid')



# For interactive plots

from plotly import offline

import plotly.graph_objs as go
# Read the input data

cle = pd.read_csv('../input/clevelandweathernan/ClevelandWeatherRemoveNAN - Sheet1.csv')
cle.head(5)
cle.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
# plot correlated values

plt.rcParams['figure.figsize'] = [16, 6]



fig, ax = plt.subplots(nrows=1, ncols=3)



ax=ax.flatten()



cols = ['Temp_dif', 'Pressure_dif', 'Precip_Avg']

colors=['#415952', '#f35134', '#243AB5', '#243AB5']

j=0



for i in ax:

    if j==0:

        i.set_ylabel('GoogleData')

    i.scatter(cle[cols[j]], cle['GoogleData'],  alpha=0.5, color=colors[j])

    i.set_xlabel(cols[j])

    i.set_title('Pearson: %s'%cle.corr().loc[cols[j]]['GoogleData'].round(2)+' Spearman: %s'%cle.corr(method='spearman').loc[cols[j]]['GoogleData'].round(2))

    j+=1



plt.show()
# Define a function for a histogram

def histogram(data, x_label, y_label, title):

    _, ax = plt.subplots()

    ax.hist(data, color = '#539caf')

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    ax.set_title(title)



# Call the function to create plot

histogram(data = cle['GoogleData']

           , x_label = 'Pressure_dif'

           , y_label = 'GoogleData'

           , title = 'GoogleData vs. Barometric Pressure')
%matplotlib inline

import matplotlib.pyplot as plt

cle.hist(bins=50, figsize=(20,15))

plt.savefig("attribute_histogram_plots")

plt.show()
corr_matrix = cle.corr()

corr_matrix["GoogleData"].sort_values(ascending=False)
cle = cle.dropna()
X = cle[['Temp_Max', 'Temp_Min', 'Temp_dif', 'DewPoint_Max', 'DewPoint_Min', 'DewPoint_dif', 

        'Humidity_Max', 'Humidity_Min', 'Humidity_dif', 'WindSpeed_Max', 'WindSpeed_Min', 'WindSpeed_dif', 'Pressure_Max', 'Pressure_Min', 'Pressure_dif', 'Precip_Avg']]

Y = cle['GoogleData']



#n = pd.get_dummies(cle.group)

#X = pd.concat([X, n], axis=1)

#m = pd.get_dummies(cle.usecode)

#X = pd.concat([X, m], axis=1)

#drops = ['group', 'usecode']

#X.drop(drops, inplace=True, axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Linear Regression R squared": %.4f' % regressor.score(X_test, y_test))
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_pred, y_test)

lin_rmse = np.sqrt(lin_mse)

print('Linear Regression RMSE: %.4f' % lin_rmse)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)

forest_reg.fit(X_train, y_train)
print('Random Forest R squared": %.4f' % forest_reg.score(X_test, y_test))