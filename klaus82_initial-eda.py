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
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
train_df.head()
train_df.info()
train_df.isna().sum()
train_df[train_df.isnull().any(axis=1)].head(10)
test_df.info()
test_df.isna().sum()
test_df[test_df.isnull().any(axis=1)].head(10)
train_df.groupby("City").count()
import plotly.express as px
train_intersections_count=train_df.groupby(['City','Latitude','Longitude']).IntersectionId.count().reset_index()

train_intersections_count.columns=['City','Latitude','Longitude','Count_Obs']
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Atlanta'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Boston'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Chicago'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Philadelphia'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
#Creating Dummies for train Data

dfen = pd.get_dummies(train_df["EntryHeading"],prefix = 'en')

dfex = pd.get_dummies(train_df["ExitHeading"],prefix = 'ex')

#hours = pd.get_dummies(train_df["Hour"],prefix = 'hour')

city=pd.get_dummies(train_df["City"])

train_df = pd.concat([train_df,dfen],axis=1)

train_df = pd.concat([train_df,dfex],axis=1)

train_df = pd.concat([train_df,city],axis=1)

#train_df = pd.concat([train_df,hours],axis=1)



#Creating Dummies for test Data

dfent = pd.get_dummies(test_df["EntryHeading"],prefix = 'en')

dfext = pd.get_dummies(test_df["ExitHeading"],prefix = 'ex')

#hours = pd.get_dummies(test_df["Hour"],prefix = 'hour')

city=pd.get_dummies(test_df["City"])

test_df = pd.concat([test_df,dfent],axis=1)

test_df = pd.concat([test_df,dfext],axis=1)

test_df = pd.concat([test_df,city],axis=1)

#test_df = pd.concat([test_df,hours],axis=1)
train_df.head()
#Training Data

X = train_df[["IntersectionId","Latitude","Longitude","Hour","Weekend","Month",

              'Atlanta','Boston','Chicago','Philadelphia',

               'en_E','en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

               'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]

y1 = train_df["TotalTimeStopped_p20"]

y2 = train_df["TotalTimeStopped_p50"]

y3 = train_df["TotalTimeStopped_p80"]

y4 = train_df["DistanceToFirstStop_p20"]

y5 = train_df["DistanceToFirstStop_p50"]

y6 = train_df["DistanceToFirstStop_p80"]
X.head()
testX = test_df[["IntersectionId","Latitude","Longitude","Hour","Weekend","Month",

              'Atlanta','Boston','Chicago','Philadelphia',

               'en_E','en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

               'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]
import matplotlib.pyplot as plt
#Correlation for engireed dataset

f = plt.figure(figsize=(19, 15))

plt.matshow(train_df.corr(), fignum=f.number)

plt.xticks(range(train_df.shape[1]), train_df.columns, fontsize=14, rotation=45)

plt.yticks(range(train_df.shape[1]), train_df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
#Correlation for engireed dataset

f = plt.figure(figsize=(19, 15))

plt.matshow(X.corr(), fignum=f.number)

plt.xticks(range(X.shape[1]), X.columns, fontsize=14, rotation=45)

plt.yticks(range(X.shape[1]), X.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
from sklearn import linear_model

#lasso_reg = linear_model.Lasso(alpha=0.00001, random_state=1)

lasso_reg = linear_model.BayesianRidge()

lasso_reg.fit(X,y1)

predict_l1= lasso_reg.predict(testX)

lasso_reg.fit(X,y2)

predict_l2= lasso_reg.predict(testX)

lasso_reg.fit(X,y3)

predict_l3= lasso_reg.predict(testX)

lasso_reg.fit(X,y4)

predict_l4= lasso_reg.predict(testX)

lasso_reg.fit(X,y5)

predict_l5= lasso_reg.predict(testX)

lasso_reg.fit(X,y6)

predict_l6= lasso_reg.predict(testX)
# Appending all predictions

prediction_l = []

for i in range(len(predict_l1)):

    for j in [predict_l1,predict_l2,predict_l3,predict_l4,predict_l5,predict_l6]:

        prediction_l.append(j[i])

submission_l = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_l["Target"] = prediction_l

submission_l.to_csv("Submission_l.csv",index = False)     
submission_l