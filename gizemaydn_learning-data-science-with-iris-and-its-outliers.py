import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_iris = pd.read_csv('/kaggle/input/iris-dataset-with-outliers/Iris_with_outliers.csv')

print('Read')
data_iris.info()
data_iris.columns
data_iris.drop(labels=data_iris.columns[0], axis=1, inplace=True)

print("dropped")
data_iris.head()
data_iris.info()
data_iris.describe()
data_iris.groupby('Species').agg(["min","max","std","mean"])
data_iris.isnull().values.any()
data_iris.isna().sum()
for column in data_iris.columns[1:-1]:

    data_iris[column].fillna(value=data_iris[column].mean(), inplace=True)
sns.scatterplot(data=data_iris, x="Id",y="SepalLengthCm",hue="Species")
sns.pairplot(data = data_iris, hue="Species", markers=["o","s","d"]);
sns.pairplot(data = data_iris, kind="reg", hue="Species");
data_iris.shape
for column in data_iris.columns[1:-1]:

    for specy in data_iris["Species"].unique():

        Specy_type=data_iris[data_iris["Species"]==specy]

        Selected_column=Specy_type[column]

        avg = Selected_column.mean()

        std = Selected_column.std()

        upper_lmt = avg + (3 * std) 

        lower_lmt= avg - (3 * std)

        outliers=Selected_column[((Selected_column > upper_lmt) | (Selected_column< lower_lmt))].index # picking outliers' indeces

        data_iris.drop(index=outliers, inplace=True) # dropping outliers

        print(column,specy,outliers)               
for column in data_iris.columns[1:-1]:

    for specy in data_iris["Species"].unique():

        Specy_type = data_iris[data_iris["Species"] == specy]

        Selected_column = Specy_type[column]

        q1 = Selected_column.quantile(0.25) # for select first quartile

        q3 = Selected_column.quantile(0.75) # for select third quartile

        iqr = q3 - q1 # this is interquartile range

        upper_limit = q3 + 1.5 * iqr

        lower_limit = q1 - 1.5 * iqr        

        outlierss = Selected_column[(Selected_column > upper_limit) | (Selected_column < lower_limit)].index # picking outliers' indeces

        print(outlierss)

        data_iris.drop(index = outlierss, inplace=True) # dropping outliers

        

        

        
data_iris.to_csv("updated_data.csv")
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/working/updated_data.csv")

data.head()
data.drop(data.columns[0:2], axis=1, inplace=True)

data.head()

print("dropped columns")
data.head()
labenc= LabelEncoder()

data["Species"] = labenc.fit_transform(data["Species"]) # transforming Species column into label encoding format
data.head() # check this out
data.isna().sum()
data.dtypes
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( data.iloc[:,0:-1] , data.iloc[:,-1] , test_size=0.2 )
import xgboost as xgb 
xgb_clsfr = xgb.XGBClassifier(objective="multiclass:softmax", num_class=3)
xgb_clsfr.fit(x_train,y_train)
predictions = xgb_clsfr.predict(x_test)

predictions
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, predictions)
confusion_matrix(y_test,predictions)