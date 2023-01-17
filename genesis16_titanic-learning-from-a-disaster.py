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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

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

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
print('Data Loaded')
# Checking out the data
data.describe()
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
data.head(5)
# Checking for NULL values
data.isnull().sum()
data['Embarked'].value_counts()
# Since majority of people Embarked from S, i'm gonna fill the missing data with S
data['Embarked'] = data['Embarked'].fillna(value = 'S')
data.isnull().sum()
# Filling missing Age values with mean of all ages
data['Age'] = data['Age'].fillna(value = data['Age'].mean())
data.head(5)

# Generating heatmap to find correlation between features
corr = data.corr()
sns.heatmap(data = corr, square = True, annot = True, cbar = True)
sns.barplot(x = 'Embarked', y = 'Survived', hue = 'Sex', data = data)
sns.barplot(x = 'Survived', y = 'Fare', hue = 'Pclass', data = data)
sns.barplot(x = 'Pclass', y = 'Parch', hue = 'Sex', data = data)
# Using LabelEncoder to generate numeric data
columns = ['Sex', 'Embarked']
le = LabelEncoder()
for col in columns:
    data[col] = le.fit_transform(data[col])
# Generating test and train cases
array = data.values
X = array[ : , 1: ]
Y = array[ : , 0]

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.20, random_state = 1)
models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge(normalize = True)),
          ('Lasso', Lasso(normalize = True)),
          ('RidgeCV', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
          ('ElasticNet', ElasticNet(random_state = 1)),
          ('RandomForestRegressor', RandomForestRegressor(max_depth=2, random_state=1, n_estimators=1000)),
          ('BaggingRegressor', BaggingRegressor()),
          ('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.05,max_depth = 1, random_state = 1)),
          ('AdaBoostRegressor', AdaBoostRegressor(n_estimators = 1000)),
          ('KNeighborsRegressor', KNeighborsRegressor()),
          ('XGBoost', XGBClassifier(learning_rate = 0.2, n_estimators = 1000, max_depth = 5, min_child_weight = 1, gamma = 0.2, seed = 7))
         ]

for name, model in models:
    model.fit(train_x, train_y)
    pred = model.predict(test_x).astype(int)
    print(name, accuracy_score(test_y, pred))
