import sklearn 

import pandas as pd

import numpy as np
from sklearn.feature_selection import VarianceThreshold
X = [[0,1,2,3],[0,4,5,3],[0,4,6,3]]
X_df = pd.DataFrame(X)
X_df
feature_selector = VarianceThreshold()

new_features = feature_selector.fit_transform(X_df)
new_features
feature_selector.variances_
feature_selector01 = VarianceThreshold(threshold = 2.5)

new_features01 = feature_selector01.fit_transform(X_df)

new_features01
# Create feature matrix with two highly correlated features

X = np.array([[1, 1, 1],

              [2, 2, 0],

              [3, 3, 1],

              [4, 4, 0],

              [5, 5, 1],

              [6, 6, 0],

              [7, 7, 1],

              [8, 7, 0],

              [9, 7, 1]])



# Convert feature matrix into DataFrame

df = pd.DataFrame(X)



# View the data frame

df
# Create correlation matrix

corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features 

df.drop(df[to_drop], axis=1)