import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
from scipy.stats import norm
#10 columns, 100000 samples

l = [norm.rvs(size = 10**5, loc = 0, scale = i+1) for i in range(10)]

l_test = [norm.rvs(size = 10**5, loc = 0, scale = i+1) for i in range(10)]

df = pd.DataFrame(l).T

df_test = pd.DataFrame(l_test).T
c = {}

for i in range(10):

    c[i] = 'Variable ' + str(i+1)



df.rename(columns = c, inplace = True)

df_test.rename(columns = c, inplace = True)
noise = norm.rvs(size = 10**5, loc = 0, scale = 0.05)

noise_test = norm.rvs(size = 10**5, loc = 0, scale = 0.05)
df['y'] = df['Variable 1']

df_test['y'] = df_test['Variable 1']

for i in range(2,11):

    df['y'] = df['y'] + df['Variable ' + str(i)]

    df_test['y'] = df_test['y'] + df_test['Variable ' + str(i)]

    

df['y'] = df['y'] + noise

df_test['y'] = df_test['y'] + noise_test
df.head()
df_test.head()
df1 = df.copy()

df2 = df.copy()

y = df.y
df2.drop('y', axis = 1, inplace = True)
df1.head()
df2.head()
from sklearn.decomposition import PCA
# We create two instances of the PCA

pca1 = PCA(n_components = 5)

pca2 = PCA(n_components = 5)
# Fit transform of the PCA in each case

principalcomponents1 = pca1.fit_transform(df1)

principalcomponents2 = pca2.fit_transform(df2)
# PCA explained variance in each dimension

_, ax = plt.subplots(figsize = (10,5))

ax.plot(pca1.explained_variance_ratio_, color = 'b')

ax.plot(pca2.explained_variance_ratio_, color = 'r')
# We take the data frame with the data represented in the principal components

principalDf1 = pd.DataFrame(data = principalcomponents1, columns = ['1 - principal component {0}'.format(i+1) for i in range(5)])

principalDf2 = pd.DataFrame(data = principalcomponents2, columns = ['2 - principal component {0}'.format(i+1) for i in range(5)])
from sklearn.linear_model import LinearRegression
linreg1 = LinearRegression()

linreg2 = LinearRegression()



linreg1.fit(principalDf1, y)

linreg2.fit(principalDf2, y)
linreg1.coef_, linreg2.coef_
df_test_1 = df_test.copy()

df_test_2 = df_test.copy()



y_test = df_test.y



df_test_2.drop('y', axis = 1, inplace = True)
principalcomponents_test_1 = pca1.transform(df_test_1)

principalcomponents_test_2 = pca2.transform(df_test_2) # Transforming our data with PCA



# Constructing our data frames.

principalDf1_test = pd.DataFrame(data = principalcomponents_test_1, columns = ['1 - principal component {0}'.format(i+1) for i in range(5)])

principalDf2_test = pd.DataFrame(data = principalcomponents_test_2, columns = ['2 - principal component {0}'.format(i+1) for i in range(5)])



predict_test_1 = linreg1.predict(principalDf1_test)

predict_test_2 = linreg2.predict(principalDf2_test) # Making our predicitions in each case
from sklearn.metrics import mean_squared_error
e1 = mean_squared_error(predict_test_1, y_test)

e2 = mean_squared_error(predict_test_2, y_test)



e1, e2
r = 0

for i in range(5):

    r = r + np.std(df_test_2['Variable {0}'.format(i+1)])**2
r