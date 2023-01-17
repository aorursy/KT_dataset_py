# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df1 = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df1.head()
df1.describe()
df1['RainToday'].replace({'No':0,'Yes':1},inplace = True)    

df1['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)   # replacing label's values

df = df1.drop(['Date'],axis=1)  # unsignificance feature

df.shape
categorical_features = df.select_dtypes(include = ["object"]).columns

categorical_features
df = pd.get_dummies(df,columns=categorical_features,drop_first=True)
df.isnull().sum(axis=0)
df = df.fillna(df.mean())
from sklearn.preprocessing import StandardScaler 



scaler = StandardScaler() 



scaled = scaler.fit_transform(df) 
scaled
from sklearn.decomposition import PCA 

  

pca_model = PCA(n_components = 2) 

pca = pca_model.fit_transform(scaled)  
variance=np.var(pca,axis=0)

variance_ratio = variance/np.sum(variance)

print(variance_ratio)
import matplotlib.pyplot as plt

plt.figure(figsize =(8, 6)) 

  

plt.scatter(pca[:, 0], pca[:, 1], c = df1['RainTomorrow'], cmap ='plasma') 

  

plt.xlabel('First Principal Component') 

plt.ylabel('Second Principal Component') 
import seaborn as sns

df_comp = pd.DataFrame(pca_model.components_, columns = df.columns)

  

plt.figure(figsize =(14, 6)) 

  

sns.heatmap(df_comp) 

test = df.copy()

test = test["RainTomorrow"].values
from sklearn.model_selection import train_test_split 



X_train, X_test, y_train, y_test = train_test_split(pca, test, test_size = 0.25) 
import xgboost as xgb 

xgb = xgb.XGBClassifier() 

xgb.fit(X_train, y_train) 

y_pred = xgb.predict(X_test) 
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: %f" % (rmse))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f" % (accuracy * 100.0))