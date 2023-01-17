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
# Read data
import pandas as pd                   # Linear Algebra (calculate the mean and standard deviation)
import numpy as np                    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

# Visualization
import matplotlib.pyplot as plt       # Visualization using matplotlib
import seaborn as sns                 # Visualization using seaborn

# style
plt.style.use("fivethirtyeight")      # Set Graphs Background style using matplotlib
sns.set_style("darkgrid")             # Set Graphs Background style using seabornsns.set()

import warnings                       # To ignore any warnings
warnings.filterwarnings('ignore')
train = pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')
test = pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Test_set.xlsx')
display(train.head())
display(test.head())
display(train.shape)
display(test.shape)
display(train.info())
display(test.info())
# Let’s append the train and test data
df = train.append(test,sort=False)
df.head()
df.shape
# Check the datatypes
df.dtypes
# Split Date_of_Journey column into date, month, year
df['Date'] = df['Date_of_Journey'].str.split('/').str[0]
df['Month'] = df['Date_of_Journey'].str.split('/').str[1]
df['Year'] = df['Date_of_Journey'].str.split('/').str[2]
df.head()
# Let’s change the data dtype of date, month, year object to an integer.
df['Date'] = df['Date'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)
# Check the datatypes
df.dtypes
# Since we have converted Date_of_Journey column into integers, Now we can drop as it is of no use.

df.drop(["Date_of_Journey"], axis = 1, inplace = True)
df.head()
df[df['Total_Stops'].isnull()]
# We can see the null value is present is tota_stops columns, so we can fill that assuming as at least 1 stop.
df['Total_Stops'] = df['Total_Stops'].fillna('1 stop')
# Also, let’s replace non stop values as 0 stop
df['Total_Stops'] = df['Total_Stops'].replace('non-stop','0 stop')
# Splits Departure_Hour and Departure_Minute to Dep_Time
df['Departure_Hour'] = df['Dep_Time'] .str.split(':').str[0]
df['Departure_Minute'] = df['Dep_Time'] .str.split(':').str[1]
df.drop(["Dep_Time"], axis = 1, inplace = True)
df.drop(["Arrival_Time"], axis = 1, inplace = True)
df.drop(["Total_Stops"], axis = 1, inplace = True)
df.head(3)
# Let’s replace and split the Route as route_1,route_2,route_3,route_4,route_5
df['Route_1'] = df['Route'].str.split('→ ').str[0]
df['Route_2'] = df['Route'].str.split('→ ').str[1]
df['Route_3'] = df['Route'].str.split('→ ').str[2]
df['Route_4'] = df['Route'].str.split('→ ').str[3]
df['Route_5'] = df['Route'].str.split('→ ').str[4]
df['Price'].fillna((df['Price'].mean()),inplace=True)
df['Route_1'].fillna("None",inplace=True)
df['Route_2'].fillna("None",inplace=True)
df['Route_3'].fillna("None",inplace=True)
df['Route_4'].fillna("None",inplace=True)
df['Route_5'].fillna("None",inplace=True)
# Now, drop the unwanted features
df = df.drop(['Route'],axis=1)
df = df.drop(['Duration'],axis=1)
df.head(3)
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df["Airline"] = LE.fit_transform(df['Airline'])
df["Source"] = LE.fit_transform(df['Source'])
df["Destination"] = LE.fit_transform(df['Destination'])
df["Additional_Info"] = LE.fit_transform(df['Additional_Info'])
df["Route_1"] = LE.fit_transform(df['Route_1'])
df["Route_2"] = LE.fit_transform(df['Route_2'])
df["Route_3"] = LE.fit_transform(df['Route_3'])
df["Route_4"] = LE.fit_transform(df['Route_4'])
df["Route_5"] = LE.fit_transform(df['Route_5'])
df.head()
# Lasso used for regularization
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
train = df[0:10683]
test = df[10683:]
# Splitting into Train and Test
X = train.drop(['Price'],axis=1)
y = train.Price
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)
model.get_support()

selected_features = X_train.columns[(model.get_support())]
from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# cross-validation
# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2,
                               random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
# predict the model on test data set
y_pred = rf_random.predict(X_test)
import seaborn as sns

sns.distplot(y_test-y_pred)
# Scatter plot on the predicted data point price
plt.scatter(y_test,y_pred)