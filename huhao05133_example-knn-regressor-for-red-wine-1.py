import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
df = pd.read_csv('../input/wine-whitered/Wine_red.csv',sep=';')
print(df.shape)
df.head(3)
(df_train,df_test) = train_test_split(df,train_size=0.8,test_size=0.2,shuffle=True,random_state=0)
features_train = df_train.iloc[:,0:-1]
features_test = df_test.iloc[:,0:-1]
targets_train = df_train.iloc[:,-1]
targets_test = df_test.iloc[:,-1]
features_train = scale(features_train)
features_test = scale(features_test)
features_train.std(axis=0)
targets_test
num_neighbors = []
R2_train = []
R2_test = []
for K in np.arange(100)+1:
    knn = KNeighborsRegressor(n_neighbors=K)
    knn.fit(features_train,targets_train)
    num_neighbors.append(K)
    R2_train.append(knn.score(features_train,targets_train))
    R2_test.append(knn.score(features_test,targets_test))

errors = pd.DataFrame()
errors['neighbors'] = num_neighbors
errors['train R2'] = R2_train
errors['test R2'] = R2_test
errors.head(3)
ax1 = errors.plot.line(x='neighbors', y='train R2')
errors.plot.line(x='neighbors',y='test R2',ax=ax1)
plt.ylabel('error R2')
print(' max test R-sqr = ',errors['test R2'].max())
ix = errors['test R2'].idxmax()
print('optimal n_neighbors =', errors.neighbors[ix])
knn = KNeighborsRegressor(n_neighbors=89)
knn.fit(features_train,targets_train)
yh = knn.predict(features_test)
y = targets_test
mse_knn = ((y-yh)**2).mean()
mse_knn
#simple bias regressor
b = y.mean()
mse_bias = ((y-b)**2).mean()
mse_bias
R2 = 1-mse_knn/mse_bias
round(R2,7)
