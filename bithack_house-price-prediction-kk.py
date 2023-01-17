# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
print(os.getcwd())

# reading the input data
df_train = pd.read_csv("../input/comp_train.csv")
df_test = pd.read_csv("../input/comp_test.csv")
print(df_train.head())
print(df_test.head())

# removing the id column from the df_train
df_train = df_train.drop(columns=['Id'])
print(df_train.head())
sns.set(rc={'figure.figsize':(11.7,8.27)})

# separating the 
df_trainfeatures = df_train.drop(columns=['SalePrice'])
df_trainpredict = pd.DataFrame(df_train[['SalePrice']])
display(df_trainfeatures.head())
display(df_trainpredict.head())
def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts.plot.scatter('time', 'val',ax=ax)


f = pd.melt(df_train, id_vars=["SalePrice"], value_vars=df_trainfeatures)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False,height=7)
g = g.map(pairplot, "value", "SalePrice")
# Calculate correlations and multiply it with 100 to increase the range from [-1 , +1 ] to [-100 , +100]
corr = df_train.corr()*100
 
# Visualising the correlations using Heatmap
sns.heatmap(corr , annot=True , vmin=-100 , vmax=100 , fmt=".1f")
# adding the quadratic features
df_trainfeatures['YearBuiltSqr'] = np.square(df_trainfeatures['YearBuilt'])
df_trainfeatures['2ndFlrSFSqr'] = np.square(df_trainfeatures['2ndFlrSF'])
df_trainfeatures['TotRmsAbvGrdSqr'] = np.square(df_trainfeatures['TotRmsAbvGrd'])
display(df_trainfeatures.head())

# removing the BsmtFinSF2 feature
df_trainfeatures = df_trainfeatures.drop(columns=['BsmtFinSF2'])
display(df_trainfeatures.head())
# First convert this pandas dataframe to a numpy array.
X = np.array(df_trainfeatures , dtype=np.float64)
Y = np.array(df_trainpredict , dtype=np.float64)

# let's confirm the shape of the X and Y
print ("Shape of df_trainfeatures: {0} \nShape of X: {1}".format(df_trainfeatures.shape , X.shape))
print ("Shape of df_trainpredict: {0} \nShape of Y: {1}".format(df_trainpredict.shape , Y.shape))

# X is a matrix where each row contains features for a certain house in the training set
# Y is a column matrix where each element in a row is sale price of a corresponding house described by corresponding row in X

# let's find out the weightage of each feature in the model.
# before doing that first stack the a columns of ones to the X matrix
X = np.column_stack( ( np.ones(X.shape[0]) , X ) )
print("Shape of X after stacking a column vector containing ones: {0}".format(X.shape))
# to finding the weights
w = np.linalg.pinv(X).dot(Y)
print(w)
# let's try a prediction on test data
print("Test Data:")
display(df_test.head())

# extracting the features out
df_testfeatures = df_test.drop(columns=['Id'])
print("Numerical Features of the Test Data:")
display(df_testfeatures.head())

# adding the quadratic features
df_testfeatures['YearBuiltSqr'] = np.square(df_testfeatures['YearBuilt'])
df_testfeatures['2ndFlrSFSqr'] = np.square(df_testfeatures['2ndFlrSF'])
df_testfeatures['TotRmsAbvGrdSqr'] = np.square(df_testfeatures['TotRmsAbvGrd'])
print("Adding Quadratic Features on test data:")
display(df_testfeatures.head())

# removing the BsmtFinSF2 feature
df_testfeatures = df_testfeatures.drop(columns=['BsmtFinSF2'])
print("Removing the BsmtFinSF2 feature on test data:")
display(df_testfeatures.head())
X_test = np.array(df_testfeatures , dtype=np.float64)
X_test = np.column_stack( ( np.ones(X_test.shape[0]) , X_test ) )
Y_test = X_test.dot(w)

df_testprediction = pd.DataFrame()
df_testprediction['Id'] = df_test['Id']
df_testprediction['SalePrice'] = Y_test
df_testprediction.to_csv('submission.csv' , index=False)