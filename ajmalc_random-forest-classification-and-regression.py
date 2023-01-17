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
import matplotlib.pyplot as plt
%matplotlib Inline 
import seaborn as sns

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LogisticRegression

#from sklearn.model_selection import cross_val_score, KFold
#from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
#from sklearn import metrics
data = pd.read_csv('/kaggle/input/dbdata/dbdata.csv')
data.describe()

data.columns
sns.barplot(data.last_updated_time,data.soil_moisture)
sns.distplot(data.soil_moisture )#Plot the distribution
sns.kdeplot(data.soil_moisture)
sns.kdeplot(data.soil_moisture, shade=True, color='pink')
#sns.kdeplot(data.last_updated_time, data.soil_moisture )

############################################################################################

#                         Convert object into Date time Formate

####################                                              ###########################

data['last_updated_time'] = pd.to_datetime(data['last_updated_time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
print(data.head(2))
print(data.info())
# to check the DateTime  data not converted data
assert data.last_updated_time.isnull().sum() == 0
#  Splite the Date time into year,month hour etc
# And create mew column in data set
# Then inser those values
data['last_updated_time'] = data['last_updated_time'] +pd.Timedelta('1d')-pd.Timedelta('1s')
data['last_updated_time'].head()
data['dated_time_year'] = data['last_updated_time'].dt.year
data['dated_time_month'] = data['last_updated_time'].dt.month
data['dated_time_week'] = data['last_updated_time'].dt.week
data['dated_time_day'] = data['last_updated_time'].dt.day
data['dated_time_hour'] = data['last_updated_time'].dt.hour
data['dated_time_minute'] = data['last_updated_time'].dt.minute
data['dated_time_dayofweek'] = data['last_updated_time'].dt.dayofweek

data.columns

##########################################################################################
#               Date Time convertion
tim =data['last_updated_time']
X = ((tim-tim[0]).dt.total_seconds()/(60*60*24)) # to get the tototal secound
X.dtype


X.shape
X

# FOr Soil Moisture data
x1 =X.values
y1 = data.soil_moisture.values
# fit train set and test set
x1_train,x1_test, y1_train,y1_test  = train_test_split(x1,y1, random_state =0,test_size = 0.25)

# reshape the data set , 1D to 2D
x1_test = x1_test.reshape(-1,1)
x1_train = x1_train.reshape(-1,1)
####             Fitting Model Random forest Classifier For Soil Moistire                                 #
rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rf.fit(x1_train, y1_train) # model for soil moisture
#                                                                                   #
# Prediction
y1_pred = rf.predict(x1_test)
y1_pred

# ACCURACY CHECKING Using
#   1.Accuracy score
#accuracy_score(y1_test, y_pred1)
print("Random forest clasifier in Soil moisture is =",round(accuracy_score(y1_test, y1_pred)*100,2),'%')# Accuracy 75
# R2 score
acc1 = r2_score(y1_test,y1_pred)
acc1 # Accuracy 94.055

#      PRediction on Future Value
rf.predict([[4.0501]])# Predicted soil moisture


cm = confusion_matrix(y1_test,y1_pred)
cm


######      ACCURACY TESTING
y1_test_size = y1_test.size
y1_train_size = y1_train.size

accu_train = np.sum(rf.predict(x1_train) == y1_train)/y1_train_size
accu_test = np.sum(rf.predict(x1_test) == y1_test)/y1_test_size
print("Accuracy on Train: ", round(accu_train*100,2),'%')
print("Accuracy on Test: ", round(accu_test*100,2),'%')



#                                     VisualizATION


# arange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
X_grid = np.arange(min(X), max(X), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value
X_grid = X_grid.reshape((len(X_grid), 1))
# Scatter plot for original data
plt.scatter(X, y1, color = 'blue')
# plot predicted data
plt.plot(X_grid, rf.predict(X_grid), color = 'green')
plt.title('Random Forest Clasifier')
plt.xlabel('Date_Time')
plt.ylabel('Soil Moisture')
plt.show()

