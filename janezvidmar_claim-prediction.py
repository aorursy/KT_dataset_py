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

import seaborn as sns
podatki = pd.read_csv("../input/podatki/podatki.csv")

podatki
podatki.info()
podatki["Cat12"].mode()

podatki["Cat12"].fillna("B", inplace=True)

cat_features = list(podatki.select_dtypes(include=['object']).columns)

print ("Categorical: {} features".format(len(cat_features)))



cont_features = [cont for cont in list(podatki.select_dtypes(

                 include=['float64', 'int64']).columns) if cont not in ['Row_ID', 'Claim_Amount']]

print ("Continuous: {} features".format(len(cont_features)))
podatki[cont_features].hist(bins=50, figsize=(16,12))
podatki["Claim_Amount"].hist(bins=50)
plt.subplots(figsize=(16,9))

correlation_mat = podatki[cont_features].corr()

sns.heatmap(correlation_mat, annot=True)
possitive_claims = podatki.loc[podatki["Claim_Amount"] != 0]

possitive_claims["Claim_Amount"].hist(bins=50)
np.log(possitive_claims["Claim_Amount"]).hist(bins=50)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



categorical_data=podatki[cat_features]



for col in categorical_data.columns:

    categorical_data[col] = le.fit_transform(categorical_data[col])
categorical_data
podatki.drop(cat_features, axis=1, inplace=True)

data_encoded = podatki.join(categorical_data)

data_encoded
data_encoded = data_encoded.assign(Claim_Exists=(podatki['Claim_Amount'] != 0).astype(int))
data_encoded
X = data_encoded.drop(["Row_ID", "Claim_Amount", "Claim_Exists"], axis=1)

y = data_encoded["Claim_Exists"]
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=15)

new_features = selector.fit_transform(X, y)
new_features
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
X = new_features

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

clf.fit(train_X, train_y)

predictions = clf.predict(val_X)

predictions = pd.Series(predictions, index = val_y.index)

comparison = pd.concat([val_y, predictions], axis=1)

comparison.columns=["Actual_Claim","Claim_Predicted"]

confusion_matrix(comparison["Actual_Claim"],comparison["Claim_Predicted"])
X = data_encoded.drop(["Row_ID", "Claim_Amount", "Claim_Exists"], axis=1)

y = data_encoded["Claim_Amount"]
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)



tree_regressor = DecisionTreeRegressor()

tree_regressor.fit(train_X, train_y)



val_predictions = tree_regressor.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn.ensemble import RandomForestRegressor





forest_model = RandomForestRegressor()

forest_model.fit(train_X, train_y)

forest_predictions = forest_model.predict(val_X)

print(mean_absolute_error(val_y, forest_predictions))
from xgboost import XGBRegressor



xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

xgb_model.fit(train_X, train_y)

XGB_predictions = xgb_model.predict(val_X)

mean_absolute_error(XGB_predictions, val_y)
XGB_predictions = pd.Series(XGB_predictions, index= val_y.index)

XGB_results = pd.concat([val_y, XGB_predictions], axis=1)

XGB_results.columns=["Actual_Claim","Claim_Predicted"] 



sns.scatterplot(x= "Actual_Claim", y= "Claim_Predicted", data=XGB_results)