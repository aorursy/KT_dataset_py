# Linear algebra
import numpy as np  

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# For building a model
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import warnings
warnings.filterwarnings('ignore')
santander_df = pd.read_csv('../input/train.csv')
santander_df.head()
X = santander_df.drop(['ID', 'target'], axis=1)
y = santander_df.target
chi_select = SelectKBest(score_func=chi2, k=12)
y = np.array(y).astype('int')
y.dtype
chi_select.fit(X, y)
chi_support = chi_select.get_support() # Contains the values either True or False, True means the feature has been
                                       # selected
chi_features = X.loc[:, chi_support].columns.tolist() # Storing the selected features
chi_features
X = santander_df[chi_features]  # Limiting our X to only 12 selected features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=6)
reg_tree = DecisionTreeRegressor(random_state=2) # No HyperParameters
reg_tree.fit(X_train, y_train)  # Training the model
predictions = reg_tree.predict(X_test) # Predicting on the unseen data
predictions = np.array(predictions).astype('int')
mse = metrics.mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
test_df = pd.read_csv('../input/test.csv')
X = test_df[chi_features]
ID = test_df.ID
target = reg_tree.predict(X)
target
submit_df = pd.DataFrame()
submit_df['ID'] = ID
submit_df['target'] = target
submit_df.head()
submit_df.to_csv('submission.csv')