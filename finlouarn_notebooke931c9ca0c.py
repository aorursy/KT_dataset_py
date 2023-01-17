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
# save filepath to variable for easier access
melbourne_file_path = '../input/melb-data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe().transpose()
melbourne_data.head()
import pandas_profiling
pandas_profiling.ProfileReport(melbourne_data)
pearsoncorr = melbourne_data.corr(method='pearson')
pearsoncorr
import seaborn as sb
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
import numpy as np
SEED = 42
np.random.seed(SEED)
target_column = 'Price'
features = melbourne_data.drop(target_column, axis=1)
features.head()
from sklearn.model_selection import train_test_split
X = features['BuildingArea'].values + features['Landsize'].values
y = melbourne_data[target_column].values
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=35)
from matplotlib import pyplot as plt
%matplotlib inline
plt.plot(X_train, y_train, 'b.')
melbourne_data.info()
melbourne_data.isnull().sum()
melbourne_data.isnull().sum()/len(melbourne_data)
melbourne_data = melbourne_data.dropna()
melbourne_data.info()
melbourne_data.describe().transpose()
melbourne_data = melbourne_data[melbourne_data['BuildingArea']!=0]
melbourne_data.describe().transpose()
target_column = 'Price'
features = melbourne_data.drop(target_column, axis=1)

X = features['BuildingArea'].values
y = melbourne_data[target_column].values

TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=35)

plt.plot(X_train, y_train, 'b.')
melbourne_data.select_dtypes(['category']).columns
melbourne_data['Regionname'].describe().transpose
sb.boxplot(x = 'Regionname', y = 'Price', data = melbourne_data)
xlabel('Regionname')
#axes[1,0].set_ylabel('Price')
set_title('Region Name v Price')
# Melbourne_data['Regionname'] = melbourne_data['Regionname'].map({'Eastern Metropolitan':'EM',
#                                                                 'Eastern Victoria':'EV',
#                                                                 'Northern Metropolitan : 'NM'})
melbourne_data.groupby(['Regionname']).describe()
melbourne_data.select_dtypes(['float64','int64']).columns
melbourne_data = melbourne_data.drop('Unnamed: 0', axis=1)
melbourne_data.select_dtypes(['float64','int64']).columns
X = melbourne_data[['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude',
       'Propertycount']].values
y = melbourne_data[target_column].values
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=35)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
plt.scatter(y_test, y_pred)
sb.distplot((y_test - y_pred))


