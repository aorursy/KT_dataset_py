import numpy as np
import pandas as pd
import seaborn as sns


# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score

# Preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder

# Metrics :
# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


dataset = pd.read_csv('../input/diamonds.csv')
print('Dataset loaded!')
# Check for null values
dataset.isnull().sum()
# Check column names
dataset.columns
# Dropping first column
dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)
# Describe data
dataset.describe()
# Check first few entries
dataset.head(10)
dataset.info()
# Exploring features
dataset.cut.value_counts()
dataset.color.value_counts()
dataset.clarity.value_counts()
# Plotting data
sns.barplot(x = 'cut', y = 'price', data = dataset)
sns.barplot(x = 'cut', y = 'price', hue = 'clarity', data = dataset)
sns.barplot(x = 'clarity', y = 'price', data = dataset)
sns.barplot(x = 'clarity', y = 'price', hue = 'cut', data = dataset)
sns.lineplot(x = 'cut', y = 'price', hue = 'clarity', data = dataset)
corr = dataset.corr()
sns.heatmap(data = corr, square = True, annot = True, cbar = True)
# Using labelencoder to encode data to numeric values
le = LabelEncoder()
columns = ['cut', 'clarity', 'color']
for col in columns:
    dataset[col] = le.fit_transform(dataset[col])
# Checking if the above code worked
dataset.head(10)
# Making a new feature 'volume' and dropping columns x, y and z
dataset['volume'] = dataset['x'] * dataset['y'] * dataset['z']
dataset.head(5)
# Dropping x, y and z columns
dataset.drop(['x', 'y', 'z'], axis = 1, inplace = True)
dataset.head(5)
# Removing rows with volume = 0 which is absurd
dataset = dataset[dataset['volume'] != 0]
dataset[dataset['volume'] == 0].count()
# Splitting train and test data
X = dataset.drop(['price'], axis = 1)
Y = dataset['price']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 1)
models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge(normalize = True)),
          ('Lasso', Lasso(normalize = True)),
          ('RidgeCV', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
          ('ElasticNet', ElasticNet(random_state = 1)),
          ('RandomForestRegressor', RandomForestRegressor(max_depth=2, random_state=1, n_estimators=100)),
          ('BaggingRegressor', BaggingRegressor()),
          ('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1,max_depth = 1, random_state = 1)),
          ('AdaBoostRegressor', AdaBoostRegressor(n_estimators = 1000)),
          ('KNeighborsRegressor', KNeighborsRegressor())
         ]

for name, model in models:
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print(name, model.score(test_x, test_y))
