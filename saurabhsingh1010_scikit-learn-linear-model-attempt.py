# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
df = pd.read_csv("../input/train.csv")
dg = pd.read_csv("../input/test.csv")
#print(df.dtypes)
#df.head()
cols2Keep = dg.columns[dg.isnull().sum() == 0]
df.isnull().sum().sort_values(ascending = False)
df.BsmtExposure.value_counts()
df.PoolQC.value_counts()
df.MiscFeature.value_counts()
df.Alley.value_counts()
df.shape


y = df.SalePrice
X = df[cols2Keep]
newX = dg[cols2Keep]
print(X.isnull().sum())
from sklearn.preprocessing import Imputer

my_imputer = Imputer() 
df.columns.values
varnames = df.columns.values

for varname in varnames:
    if df[varname].dtype == 'object':
        lst = df[varname].unique()
        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))
df.describe()
myVars = ["LotArea","OverallQual","OverallCond","BedroomAbvGr","KitchenAbvGr","FullBath","TotRmsAbvGrd"]
df.plot.scatter(x = "LotFrontage", y ='SalePrice')
def plotVar(name):
    df.plot.scatter(x = name, y ='SalePrice')
    plt.title(name + " vs price")
for n in myVars:
    plotVar(n)
from sklearn import linear_model 
X_train = df[myVars]
y_train = df['SalePrice']
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
dg = pd.read_csv("../input/test.csv")
print(dg.dtypes)
dg.head()
X_test = dg[myVars]
y_test = model.predict(X_test)
sum(y_test < 0)
dg.head()
dg['SalePrice'] = y_test
dg.head()
outfile = dg[['Id','SalePrice']]
outfile[outfile['SalePrice'] < 0]
outfile.loc[756,'SalePrice']  = 0
outfile.to_csv('output.csv', index = False)
