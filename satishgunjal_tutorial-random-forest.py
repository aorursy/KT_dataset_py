import pandas as pd

import numpy as np

from sklearn import datasets

from sklearn import model_selection

from sklearn import ensemble
wine = datasets.load_wine()

print('Dataset structure= ', dir(wine))



df = pd.DataFrame(wine.data, columns = wine.feature_names)

df['target'] = wine.target

df['wine_class'] = df.target.apply(lambda x : wine.target_names[x]) # Each value from 'target' is used as index to get corresponding value from 'target_names' 



print('Unique target values=',df['target'].unique())



df.head()
# label = 0 (wine class_0)

df[df.target == 0].head(3)
# label = 1 (wine class_1)

df[df.target == 1].head(3)
# label = 2 (wine class_2)

df[df.target == 2].head(3)
#Lets create feature matrix X  and y labels

X = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium','total_phenols', 'flavanoids', 'nonflavanoid_phenols',

       'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]

y = df[['target']]



print('X shape=', X.shape)

print('y shape=', y.shape)
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)

print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
"""

To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute

Also note that default value of criteria to split the data is 'gini'

"""

rfc = ensemble.RandomForestClassifier(random_state = 1)

rfc.fit(X_train ,y_train.values.ravel()) # Using ravel() to convert column vector y to 1D array 
print('Actual Wine type for 10th test data sample= ', wine.target_names[y_test.iloc[10]][0])

print('Wine type prediction for 10th test data sample= ',wine.target_names[rfc.predict([X_test.iloc[10]])][0])



print('Actual Wine type for 30th test data sample= ', wine.target_names[y_test.iloc[30]][0])

print('Wine type prediction for 30th test data sample= ',wine.target_names[rfc.predict([X_test.iloc[30]])][0])
rfc.score(X_test, y_test)
boston = datasets.load_boston()

print('Dataset structure= ', dir(boston))



df = pd.DataFrame(boston.data, columns = boston.feature_names)

df['target'] = boston.target



df.head()
#Lets create feature matrix X  and y labels

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]

y = df[['target']]



print('X shape=', X.shape)

print('y shape=', y.shape)
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)

print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
"""

To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute

Also note that default value of criteria to split the data is 'mse' (mean squared error)

"""

rfr = ensemble.RandomForestRegressor(random_state= 1)

rfr.fit(X_train ,y_train.values.ravel())  # Using ravel() to convert column vector y to 1D array 
prediction = pd.DataFrame(rfr.predict(X_test), columns = ['prediction'])

# If you notice X_test index starts from 307, so we must reset the index so that we can combine it with prediction values

target = y_test.reset_index(drop=True) # dropping the original index column

target_vs_prediction = pd.concat([target,prediction],axis =1)

target_vs_prediction.T
rfr.score(X_test, y_test)