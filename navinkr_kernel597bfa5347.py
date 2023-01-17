# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_data():

	train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") 

	test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") 

	return train_df , test_df
train_df , test_df = load_data()

print("Train Data Shape " , train_df.shape)

print("Test Data Shape" , test_df.shape)
train_df.head()
test_df.head()
def data_analysis(train_df):

	corrmat = train_df.corr()

	top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.60]

	print(top_corr_features)

	plt.figure(figsize=(6,6))

	sns.heatmap(

		train_df[top_corr_features].corr(), 

		annot = True, cmap = "Blues", #Blues 

		cbar = False, vmin = .5, 

		vmax = .7, square=True

		)

	plt.show()
data_analysis(train_df)

pred_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'GarageArea' ]

pred_cols_id = ['Id','OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'GarageArea' ]



def random_forest_model(train_df):

	y = train_df.SalePrice

	X = train_df[pred_cols]

	X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=0)

	model = RandomForestRegressor(max_depth=200,n_estimators=6)

	model.fit(X_train, y_train)

	print("Accuracy on test data: ",model.score(X_test, y_test))

	return model

model = random_forest_model(train_df)

def test_model(test_df,model):	

	test_X = test_df[pred_cols_id]

	test_X[test_X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

	test_X = test_X[np.isfinite(test_X).all(1)]

	test_X1 = test_X.drop('Id', axis =1)

	predicted_prices = model.predict(test_X1)

	test_X['SalePrice'] =predicted_prices

	submit = test_X[['Id' ,'SalePrice']]

	print(submit)

	submit.to_csv("sample_submission.csv",index=False)

test_model(test_df,model)
