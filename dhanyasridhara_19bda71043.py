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
#Reading the Train  dataset and test dataset

train_data = pd.read_csv("//kaggle//input//bda-2019-ml-test//Train_Mask.csv")

test_data = pd.read_csv("//kaggle//input//bda-2019-ml-test//Test_Mask_Dataset.csv")
train_data.head()
train_data.shape
train_data.describe()
train_data.motorTempFront.unique()
#Finding correlation between feature variables using heatmap



import seaborn as sns

import matplotlib.pyplot as plt 

corrmat = train_data.corr()

f, ax = plt.subplots(figsize =(9,8))

sns.heatmap(corrmat, ax = ax, cmap="Pastel2_r", linewidths = 0.1)

# Finding important(ranking) features using a barplot



data = pd.read_csv("//kaggle//input//bda-2019-ml-test//Train_Mask.csv")

X = data.iloc[:,0:15]  #independent columns

y = data.iloc[:,1]    #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model4 = ExtraTreesClassifier()

model4.fit(train_X,train_y)

print(model4.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model4.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
#boxplot to check for outliers

import seaborn as sns

sns.boxplot(data=train_data,orient='h')
#Histogram to check for the skewness

train_data.hist()
train_data.columns
#Assigning target variable

y = train_data.flag

#y.describe()
#Assigning Features/predicting variable

data_features = ['timeindex', 'currentBack', 'motorTempBack', 'positionBack',

       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',

       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',

       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',

       'velocityFront']

X = train_data[data_features]

test_data_X = test_data[data_features]
X.shape
X.describe()
#Splitting the data for training and validation

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y = train_test_split(X, y, random_state=4572)

#from sklearn.tree import DecisionTreeClassifier

#model1_data = DecisionTreeClassifier()

#model1_data.fit(train_X, train_y)

#Fitting Random forest Model

from sklearn.ensemble import RandomForestClassifier



# Create the model with 10 trees

model = RandomForestClassifier(n_estimators=15, 

                               bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(train_X, train_y)
#Calculating F1 score

from sklearn.metrics import f1_score

val_predictions = model.predict(val_X)

f1_score(val_y,val_predictions)
#Predict values for test data

val_pred = model.predict(test_data_X)

sample = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
sample["flag"] = val_pred
sample.tail()
sample.to_csv("submit_1.csv", index=False)