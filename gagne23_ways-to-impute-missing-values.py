import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#Import the dataset
housing = pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv')

housing.head()
#Explore it
housing.describe()
#More exploration
housing.info()
#Look at the sum of missing values in each columns
housing.isna().sum()
#Lets see a correlation matrixs of the dataset
corrMatrix=housing.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corrMatrix,annot=True)
sns.pairplot(housing,corner=True)
#Looking at the correlation between Median house values and all the other variables
corrMatrix['MEDV'].sort_values(ascending=False)
# IMPORT MODULES
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# split data into X and y values
X=housing.drop(['CHAS','MEDV'],axis=1) #CHAS variable does seems relevent for this task.
y=housing['MEDV']

#split train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=124)
#KNN IMPUTER 

# Impute with KNNImputer
knn_impute=KNNImputer(n_neighbors=5,weights='distance')

# transform the Na with the strategy
X_train_filled_knn=knn_impute.fit_transform(X_train)
X_test_filled_knn=knn_impute.transform(X_test)

#Convert the arrays created back to Dataframe
X_train_filled_knn= pd.DataFrame(X_train_filled_knn,columns=X_train.columns)
X_test_filled_knn=pd.DataFrame(X_test_filled_knn,columns=X_test.columns)

#Perform the model
knn_imputed_model = LinearRegression()
knn_imputed_model.fit(X_train_filled_knn,y_train)
y_pred_knn=knn_imputed_model.predict(X_test_filled_knn)

#Check the RMSE score
RMSE_knn_model = np.sqrt(mean_squared_error(y_pred_knn,y_test))
print('This is the score of KNNImputer:', RMSE_knn_model)


# MEAN IMPUTER

# impute with the mean
mean_imputer = SimpleImputer(strategy='mean')

#transform the Na with the strategy
X_train_filled_mean=mean_imputer.fit_transform(X_train)
X_test_filled_mean = mean_imputer.transform(X_test)

#Convert the arrays created back to Dataframe
X_train_filled_mean = pd.DataFrame(X_train_filled_mean,columns=X_train.columns)
X_test_filled_mean = pd.DataFrame(X_test_filled_mean,columns=X_test.columns)

#Perform the model
mean_imputed_model = LinearRegression()
mean_imputed_model.fit(X_train_filled_mean,y_train)
y_pred_mean= mean_imputed_model.predict(X_test_filled_mean)

#Check the RMSE score
RMSE_mean_model = np.sqrt(mean_squared_error(y_pred_mean,y_test))
print('This is the RMSE of the score with Nas impute with the mean: ',RMSE_mean_model)

# MEDIAN IMPUTER

#impute with the median
median_imputer=SimpleImputer(strategy='median')

# fill the Na with the strategy
X_train_filled_median=median_imputer.fit_transform(X_train)
X_test_filled_median=median_imputer.transform(X_test)

#Convert the arrays created back to Dataframe
X_train_filled_median=pd.DataFrame(X_train_filled_median,columns=X_train.columns)
X_test_filled_median=pd.DataFrame(X_test_filled_median,columns=X_test.columns)

#Perform the model
median_imputed_model=LinearRegression()
median_imputed_model.fit(X_train_filled_median,y_train)
y_pred_median=median_imputed_model.predict(X_test_filled_median)

#Check the RMSE score
RMSE_median_model=np.sqrt(mean_squared_error(y_pred_median,y_test))
print('This is the score for the median imputed Nas: ',RMSE_median_model)

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
