import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
df = pd.read_csv("../input/insurance.csv")
df.head()
X = df.iloc[:,:-1].values
y = df.iloc[:,6].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

X[:,1] = LabelEncoder().fit_transform(X[:,1])
X[:,4] = LabelEncoder().fit_transform(X[:,4])
X[:,5] = LabelEncoder().fit_transform(X[:,5])
onehot = OneHotEncoder(categorical_features=[5])
X = onehot.fit_transform(X).toarray()
X = X[:,1:]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(max_leaf_nodes=18,random_state=0,n_estimators=500)

regressor.fit(x_train,y_train)
print(regressor.score(x_test,y_test) * 100,"%")

