import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/BlackFriday.csv")
train.head()
train.shape
train.dtypes
train.isna().sum()
train.drop(['Product_Category_2','Product_Category_3'],inplace=True,axis=1)
ids_train = train['User_ID'].copy()
product_ids_train = train['Product_ID'].copy()
cutoff_purchase = np.percentile(train['Purchase'], 99.9)  # 99.9 percentile
train.ix[train['Purchase'] > cutoff_purchase, 'Purchase'] = cutoff_purchase
from sklearn.preprocessing import LabelEncoder

# Label Encoding User_IDs
le = LabelEncoder()
train['User_ID'] = le.fit_transform(train['User_ID'])

# Label Encoding Product_IDs
train['Product_ID'] = le.fit_transform(train['Product_ID'])
# Dummy Variable
train = pd.get_dummies(train, drop_first=True)

X = train.drop('Purchase', axis=1)
y = train['Purchase']
from  sklearn.preprocessing  import StandardScaler

slc= StandardScaler()
X = slc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

num_folds = 10
seed = 0
scoring = 'neg_mean_squared_error'

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(criterion= 'mse', n_estimators= 50)
model.fit(X_train,y_train)
_ = model.predict(X_test)
print("RMSE :", np.sqrt(mean_squared_error(y_test, _)))
print("R2 :", r2_score(y_test, _))
