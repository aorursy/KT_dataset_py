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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv("../input/into-the-future/train.csv")
dataset.head()
dataset.shape
# FIRST WE WILL PERFORM : Exploratory Data Analysis

dataset.info()
dataset.describe()
dataset.corr()
sns.heatmap(dataset.corr(),annot=True,cmap="viridis")
sns.pairplot(dataset)
## Here we will check the null value present
sns.heatmap(dataset.isnull(),yticklabels=False,cmap="viridis")
# We can see in above heatmap that there's no null values present
# Find features only with numerical values
numerical_features= [feature for feature in dataset.columns if dataset[feature].dtypes!="O"]
print("Number Of Numerical Variables : ", len(numerical_features))
#or
print("Number Of Numerical Variables : ", len(dataset.corr()))

#visualize numerical variables

dataset[numerical_features].head()
# Find features only with numerical values


numerical_features= [feature for feature in dataset.columns if dataset[feature].dtypes!="O"]
print("Number Of Numerical Variables : ", len(numerical_features))
#or
print("Number Of Numerical Variables : ", len(dataset.corr()))

#visualize numerical variables

dataset[numerical_features].head()
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25]
print("Discrete Variable Count : {}".format(len(discrete_feature)))
# THEREFORE, THERE ARE NO DISCRETE VARIABLE PRESENT IN DATASET
# Now let's find Continous variable
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print( "Number Of Continuos Variable : {}".format(len(continuous_feature)))
dataset[continuous_feature]
## Lets analyse the continuous values by creating histograms to understand the distribution
sns.set_style("whitegrid")
for feature in continuous_feature:
    data=dataset.copy()
    data[feature].plot.hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
## We will be using logarithmic transformation
for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data["feature_2"]=np.log(data["feature_2"])
        plt.scatter(data[feature],data["feature_2"])
        plt.xlabel(feature)
        plt.ylabel('feature_2')
        plt.title(feature)
        plt.show()
# OUTLIERS DETECTION
for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
x=dataset[["feature_1"]] # We will not use the unnecessary features like id and time
x.head()
y=dataset[["feature_2"]] # dependant feature
y.head()
# Let's begin the prediction process
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print(model.intercept_)
print(model.coef_)
coeff_df = pd.DataFrame(model.coef_,x.columns,columns=['Coefficient'])
coeff_df
predictions = model.predict(x_test)
predictions
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
test_data=pd.read_csv("../input/into-the-future/test.csv")
test_data.head()
final_test_data=pd.DataFrame(test_data["feature_1"],columns=["feature_1"])
final_test_data
pred=model.predict(final_test_data)
pred
final_test_data["feature_2"]=pred
final_test_data
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
