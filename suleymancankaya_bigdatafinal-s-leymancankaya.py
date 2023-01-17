import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

train  = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

test = test.dropna()

d_satis = train.groupby('date').item_cnt_day.sum().reset_index()

d_satis.head()
import plotly.offline as pyoff

import plotly.graph_objs as go

plot_data = [

    go.Scatter(

        x=d_satis['date'],

        y=d_satis['item_cnt_day'],

    )

]

plot_layout = go.Layout(

        title=' Satışlar'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
d_satis.isnull().values.any() # boş değer kontrolü
d_satis.hist()
import numpy as np

X = pd.DataFrame(train['item_id'])

y = pd.DataFrame(train['item_price'])

model = LinearRegression()

scores = []

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X, y)):

 model.fit(X.iloc[train,:], y.iloc[train,:])

 score = model.score(X.iloc[test,:], y.iloc[test,:])

 scores.append(score)

print(scores)

plt.plot(X, model.predict(X), color='red')

plt.show()