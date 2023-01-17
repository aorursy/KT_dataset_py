import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_excel("../input/flightpriceprediction/Data_Train.xlsx")
test_data = pd.read_excel("../input/flightpriceprediction/Test_set.xlsx")
train_data.head()
data = train_data.append(test_data,sort=False)
data.tail()
data.dtypes
data['Date'] = data['Date_of_Journey'].str.split('/').str[0]
data['Month'] = data['Date_of_Journey'].str.split('/').str[1]
data['Year'] = data['Date_of_Journey'].str.split('/').str[2]

data.head()
data.dtypes
data['Date'] = data['Date'].astype(int)
data['Month'] = data['Month'].astype(int)
data['Year'] = data['Year'].astype(int)

data.dtypes
data = data.drop(['Date_of_Journey'],axis=1)
data.head()
data['Arrival_Time'] = data['Arrival_Time'].str.split(' ').str[0]
data.head()
data[data['Total_Stops'].isnull()]
data['Total_Stops'] = data['Total_Stops'].fillna('1 stop')
data['Total_Stops'] = data['Total_Stops'].replace('non-stop', '0 stop')

data.head()
data['Stop'] = data['Total_Stops'].str.split(' ').str[0]
data.head()
data.dtypes
data['Stop'] = data['Stop'].astype(int)
data = data.drop(['Total_Stops'],axis=1)

data.head()
data['Arrival_Hour'] = data['Arrival_Time'].str.split(':').str[0]
data['Arrival_Minute'] = data['Arrival_Time'].str.split(':').str[1]

data['Arrival_Hour'] = data['Arrival_Hour'].astype(int)
data['Arrival_Minute'] = data['Arrival_Minute'].astype(int)

data = data.drop(['Arrival_Time'],axis=1)

data.head()
data['Departure_Hour'] = data['Dep_Time'].str.split(':').str[0]
data['Departure_Minute'] = data['Dep_Time'].str.split(':').str[0]

data['Departure_Hour'] = data['Departure_Hour'].astype(int)
data['Departure_Minute'] = data['Departure_Minute'].astype(int)

data = data.drop(['Dep_Time'],axis=1)

data.head()
data['Route_1'] = data['Route'].str.split('→ ').str[0]
data['Route_2'] = data['Route'].str.split('→ ').str[1]
data['Route_3'] = data['Route'].str.split('→ ').str[2]
data['Route_4'] = data['Route'].str.split('→ ').str[3]
data['Route_5'] = data['Route'].str.split('→ ').str[4]

data.head()
data['Price'].fillna((data['Price'].mean()),inplace=True)
data['Route_1'].fillna("None",inplace=True)
data['Route_2'].fillna("None",inplace=True)
data['Route_3'].fillna("None",inplace=True)
data['Route_4'].fillna("None",inplace=True)
data['Route_5'].fillna("None",inplace=True)

data.head()
data = data.drop(['Route'],axis=1)
data = data.drop(['Duration'],axis=1)

data.head()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data['Airline'] = encoder.fit_transform(data['Airline'])
data['Source'] = encoder.fit_transform(data['Source'])
data['Destination'] = encoder.fit_transform(data['Destination'])
data['Additional_Info'] = encoder.fit_transform(data['Additional_Info'])
data['Route_1'] = encoder.fit_transform(data['Route_1'])
data['Route_2'] = encoder.fit_transform(data['Route_2'])
data['Route_3'] = encoder.fit_transform(data['Route_3'])
data['Route_4'] = encoder.fit_transform(data['Route_4'])
data['Route_5'] = encoder.fit_transform(data['Route_5'])

data.head()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

train = data[0:10683]
test = data[10683:]
X = train.drop(['Price'],axis=1)
y = train.Price
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
model.fit(X_train,y_train)
model.get_support()
selected_features = X_train.columns[(model.get_support())]
selected_features
X_train = X_train.drop(['Year'],axis=1)
X_test = X_test.drop(['Year'],axis=1)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples  required at each leaf
min_samples_leaf = [1, 2, 5, 10]
# Create random grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparamters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

# random search of parameters, using 3 fold cross validation
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator=rf_regressor, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=50,
                              cv=5, verbose=5, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)
y_pred = rf_random.predict(X_test)
import seaborn as sns

sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)
