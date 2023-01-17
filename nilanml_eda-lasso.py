# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/avocado.csv')
df = df.drop('Unnamed: 0',1)
df = df[df.region!='TotalUS']
df = df[df.region!='West']
df = df[df.region!='California']
df = df[df.region!='GreatLakes']
df = df[df.region!='Northeast']
df = df[df.region!='Midsouth']
df = df[df.region!='SouthCentral']
df = df[df.region!='Southeast']
df = df[df.region!='Plains']
df.head(3)
df['Date']=pd.to_datetime(df['Date'], format="%Y/%m/%d")
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


groupBy_whole = df.groupby('Date').mean()
scatter = go.Scatter(x=groupBy_whole.AveragePrice.index, y=groupBy_whole.AveragePrice)
data = [scatter]
layout=go.Layout(title="Time Series Plot for Mean Daily Price across all regions", xaxis={'title':'Date'}, yaxis={'title':'Prices'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)
conventional = df[df.type=="conventional"]
organic = df[df.type=="organic"]

groupBy1_price = conventional.groupby('Date').mean()
scatter1 = go.Scatter(x=groupBy1_price.AveragePrice.index, y=groupBy1_price.AveragePrice, name="Conventional")

groupBy2_price = organic.groupby('Date').mean()
scatter2 = go.Scatter(x=groupBy2_price.AveragePrice.index, y=groupBy2_price.AveragePrice, name="Organic")

data = [scatter1, scatter2]
layout=go.Layout(title="Time Series Plot for Mean Daily Price of Conventional and Organic Avocados", xaxis={'title':'Date'}, yaxis={'title':'Prices'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupby_region_conventional = conventional.groupby('region').median()
groupby_region_conventional = groupby_region_conventional.sort_values('AveragePrice')

bar_data_conventional = go.Bar(
            x=groupby_region_conventional.AveragePrice,
            y=groupby_region_conventional.AveragePrice.index,
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [bar_data_conventional]
layout=go.Layout(title="Median Price of Conventional Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'},height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupby_region_organic = organic.groupby('region').median()
groupby_region_organic = groupby_region_organic.sort_values('AveragePrice')

bar_data_organic = go.Bar(
            x=groupby_region_organic.AveragePrice,
            y=groupby_region_organic.AveragePrice.index,
            orientation = 'h',
            marker=dict(
                color='rgb(248,146,146)',
                line=dict(
                    color='rgb(249,52,52)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [bar_data_organic]
layout=go.Layout(title="Median Price of Organic Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'}, height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupBy1_volume = conventional.groupby('Date').mean()
groupBy2_volume = organic.groupby('Date').mean()

scatter1 = go.Scatter(x=groupBy1_volume['Total Volume'].index, y=groupBy1_volume['Total Volume'], name="Conventional")

scatter2 = go.Scatter(x=groupBy2_volume['Total Volume'].index, y=groupBy2_volume['Total Volume'], name="Organic")

data = [scatter1, scatter2]
layout=go.Layout(title="Time Series Plot for Volume of Conventional and Organic Avocados Sold", xaxis={'title':'Date'}, yaxis={'title':'Volume'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupby_region_conventional = conventional.groupby('region').median()
groupby_region_conventional = groupby_region_conventional.sort_values('Total Volume')

bar_data_organic = go.Bar(
            x=groupby_region_conventional['Total Volume'],
            y=groupby_region_conventional['Total Volume'].index,
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [bar_data_organic]
layout=go.Layout(title="Volume of Conventional Avocado Sold in each Region", xaxis={'title':'Volume'}, yaxis={'title':'Region'}, height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupby_region_organic = organic.groupby('region').median()
groupby_region_organic = groupby_region_organic.sort_values('Total Volume')

bar_data_organic = go.Bar(
            x=groupby_region_organic['Total Volume'],
            y=groupby_region_organic['Total Volume'].index,
            orientation = 'h',
            marker=dict(
                color='rgb(248,146,146)',
                line=dict(
                    color='rgb(249,52,52)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [bar_data_organic]
layout=go.Layout(title="Volume of Organice Avocado Sold in each Region", xaxis={'title':'Volume'}, yaxis={'title':'Region'}, height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
import seaborn as sns
correlation = conventional.groupby('Date').median()
correlation = groupby_region_conventional.corr()

sns.heatmap(correlation,square=False,robust=True,annot=True,fmt=".1f",annot_kws={"size": 12})
ml_df = df.copy()
X = ml_df.drop('AveragePrice',1)
y = ml_df.AveragePrice
X = X[:-1].reset_index(drop=True)
y = y[1:].reset_index(drop=True)


ml_df = pd.concat([X,y], 1)
ml_df =  ml_df.groupby('Date').mean()

ml_df.dtypes
X = ml_df.drop('AveragePrice',1)
y = ml_df.AveragePrice

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10, shuffle=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# print X_train.shape, X_test.shape

classifiers = [['DecisionTree :',DecisionTreeRegressor()],
               ['RandomForest :',RandomForestRegressor()],
               ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
               ['SVM :', SVR()],
               ['AdaBoostClassifier :', AdaBoostRegressor()],
               ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
               ['Xgboost: ', XGBRegressor()],
               ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
               ['Lasso: ', Lasso()],
               ['Ridge: ', Ridge()],
               ['BayesianRidge: ', BayesianRidge()],
               ['ElasticNet: ', ElasticNet()],
               ['HuberRegressor: ', HuberRegressor()]]

print("Accuracy Results...")


for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
classifier = Lasso()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

new_frame = y_test.copy()
new_frame['Predictions'] = predictions

scatter1 = go.Scatter(x=ml_df.index, y=ml_df['AveragePrice'], name="Actual")

scatter2 = go.Scatter(x=new_frame.index, y=new_frame['Predictions'], name="Predictions")

data = [scatter1, scatter2]
layout=go.Layout(title="Prediction vs Actual Test Points", xaxis={'title':'Date'}, yaxis={'title':'Volume'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)