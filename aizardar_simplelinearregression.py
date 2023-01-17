import pandas as pd

import numpy as np

import seaborn as sns   

import matplotlib.pyplot as plt





test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

# Lets have a look at the data and its shape



train.head()
train.shape
test.head()
test.shape
# Here, we want to see if there is any correlation in the data



plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(10,5))         # Sample figsize in inches

sns.heatmap(train.corr(), linewidths=.5, ax=ax)

plt.show()
# Lets split our training data into features and target



X_train_feature_cols = train.columns.values[(train.columns.values != 'Id') & (train.columns.values != 'y')]

X_train = train[X_train_feature_cols]

y_train = train['y']



# Also with test data



X_test_feature_cols = test.columns.values[test.columns.values != 'Id']

X_test = test[X_test_feature_cols]

# We do a simple linear regression using sklearn 



from sklearn.linear_model import LinearRegression



reg = LinearRegression().fit(X_train, y_train) # fitting the data

reg.score(X_train, y_train)  # How well do we fit 
# Coefficients 



reg.coef_
# to get the predicted y



y_pred = reg.predict(X_test)
X_test_get_Id = test.columns.values[test.columns.values == 'Id']

Id = test[X_test_get_Id]
Id = test.iloc[:, :1].values

Id = np.reshape(Id,(2000,))
# To write to a file 



df = pd.DataFrame({'Id':Id, 'y':y_pred})



df.to_csv(r'solution.csv', index = False)