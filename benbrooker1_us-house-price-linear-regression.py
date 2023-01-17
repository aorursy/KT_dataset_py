import numpy as np

import pandas as pd
import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
df = pd.read_csv('../input/USA_Housing.csv', engine='python')
df.head()
df.info()
df.describe()
def get_state(x):

    words = [i for i in x.split(' ')]

    return words[::-1][1]



regions = {'New England':['CT','MA','ME','NH','RI','VT'],'Mid Atlantic':['PA','NY','DE','NJ','MD'],

              'South':['AL','AR','KY','GA','MS','LA','SC','NC','VA','WV','TN'],'Texas':['TX'],'Florida':['FL'],

               'Midwest':['IA','IL','IN','MI','MN','WI','OH','MO'],

              'Great Plains':['SD','ND','KS','NE','OK'],'Rocky Mountains':['CO','MT','ID','WY'],

              'South West':['NV','AZ','NM','UT'],'California':['CA'],'Pacific Northwest':['WA','OR'],'Alaska':['AK'],

              'Hawaii':['HI'],'Commonwealth':['DC','MH'],'Military':['AE','AA','AP']}

   

def get_region(x):

    for i in list(regions.keys()):

        if x in regions[i]:

            return i
list(regions.keys())
df['State'] = df['Address'].apply(lambda x:get_state(x))

df['Region'] = df['State'].apply(lambda x:get_region(x))
df.head()
dfreg = df[['Region','State','Price']]



regionmean = dfreg.groupby('Region').mean()



AvPerRegion = pd.DataFrame(regionmean).sort_values('Price')



AvPerRegion.plot(kind='bar',figsize=(16,7),fontsize=20)

plt.xlabel('Region', fontsize=20)

plt.ylabel('Avg. house price', fontsize=20)

plt.suptitle('Avg. House Price by Region', fontsize=30)
stateprice = dfreg.groupby('State').mean()

stateprice.columns = ['Average house price']



statecodes = [i for i in stateprice.index]



px.choropleth(data_frame = stateprice,

               locations = statecodes,

              locationmode = 'USA-states',

              scope='usa',

             color = 'Average house price',

             color_continuous_scale = 'Blues')
df.head()
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]

y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100)
lm = LinearRegression()
lm.fit(X_train,y_train)
coef = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])

coef.index.name = 'Independent variables'

coef
predictions = lm.predict(X_test)

predictions
sns.set_style('darkgrid')

sns.scatterplot(y_test,predictions).set(xlabel='Price',ylabel='Predicted Price')
sns.distplot(y_test-predictions,bins=30).set(xlabel='Price',ylabel='Error')
mae = metrics.mean_absolute_error(y_true=y_test,y_pred=predictions)

mse = metrics.mean_squared_error(y_true=y_test,y_pred=predictions)

rmse = np.sqrt(mse)
errors = pd.DataFrame([mae,mse,rmse],columns=['error measure'],index=['mae','mse','rmse'])

errors