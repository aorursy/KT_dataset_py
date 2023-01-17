# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from re import sub
from decimal import Decimal
import matplotlib.pyplot
import pylab
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
# Any results you write to the current directory are saved as output.
#Importing the data

data=pd.read_csv("../input/data.csv")
#Plotting the overall ratings of the players

%matplotlib inline
data.Potential.hist(bins=50, figsize=(20,15))
plt.show()
#Deleting the columns which will not be used

t=["Name","Club Logo","Work Rate","Body Type","Real Face","Loaned From","Contract Valid Until","ID","Photo","Nationality","Flag","Special","Joined","Height","Weight","LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB"]
d1=data
d1.columns
d=d1.drop(t,axis=1)
del d["Unnamed: 0"]
#Covering the wages and values of players to float type variables

wage=[]
for money in d.Wage:
    value = Decimal(sub(r'[^\d.]', '', money))
    wage.append(value)
wage=[int(x) for x in wage]
d["Wage"]=wage
v=[]
d.Value[1]
for money in d.Value:
    m=str(money)
    value = Decimal(sub(r'[^\d.]', '', money))
    if (m.endswith('K')):
        value=value/1000
    v.append(value)
v=[int(x) for x in v]
d["Value"]=v
release=[]
#Plotting the wage and values of the players against the overall ratings of the players 

d.plot(kind="scatter",y="Value",x="Overall")
d.plot(kind="scatter",x="Overall",y="Wage")
#Plotting the clubs which have the highest wage bills

t1=d.groupby(by=["Club"])["Wage"].sum().sort_values(ascending=False)
t2=d.groupby(by=["Club"])["Overall"].mean().sort_values(ascending=False)
t1[:10].plot.bar()
t2[:10].plot.bar()
#Removing the information on goalkeepers

dat1=(d[d["Position"]!="GK"])
Club=dat1["Club"].copy()
delete=["GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes","Jersey Number","Release Clause","Potential","Club"]
dat1=dat1.drop(delete,axis=1)
#Removing the observations for which missing values are present

dat1.isnull().sum()
dat1.Position
dat1.dropna(subset=["Position"],inplace=True)
#Splitting the data into train and test set

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dat1, test_size=0.2, random_state=4)
Value=train_set["Value"]
train=train_set.drop("Value",axis=1)
#Defining classes which we use to select columns and binarize categorical data

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
#Selecting categorical and numerical columns

cat_attribs = ["Preferred Foot","Position"]
num_attribs=[]
for i in train.columns:
    if i not in cat_attribs:
        num_attribs.append(i)
#Defining a pipeline to pre-process the data

num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('std_scaler', StandardScaler()),
])
cat_pipeline1 = Pipeline([
('selector', DataFrameSelector(cat_attribs[0])),
('label_binarizer', MyLabelBinarizer()),
])
cat_pipeline2 = Pipeline([
('selector', DataFrameSelector(cat_attribs[1])),
('label_binarizer', MyLabelBinarizer()),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline1", cat_pipeline1),
("cat_pipeline2", cat_pipeline2)
])

train_prepared=full_pipeline.fit_transform(train)
#DataFrameSelector(cat_attribs)

#Fitting a linear regression model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
model=lin_reg.fit(train_prepared, Value)
#Computing the r2 metric on the train data set

from sklearn.metrics import r2_score
predictions = lin_reg.predict(train_prepared)
lin_score = r2_score(predictions,Value)
lin_score
#Computing the r2 metric on the test dataset

X_test = test_set.drop("Value", axis=1)
y_test = test_set["Value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
r2_score(y_test,final_predictions)