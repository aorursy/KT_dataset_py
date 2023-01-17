# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/swedish-motor-insurance/SwedishMotorInsurance.csv')
df.head()
df = df[df['Payment']!= 0]
df = df[df['Payment'] <=1000000]
df['payment_per_insured'] = df['Payment']/df['Insured']
df['payment_per_claims'] = df['Payment']/df['Claims']
df['payment_per_insured_per_claims'] = df['Payment']/(df['Claims']*df['Insured'])
df['insured_per_claims'] = df['Insured']/df['Claims']
df['Payment'] = np.log1p(df['Payment'])
df.corr()
from sklearn.metrics import r2_score

train = df[['Claims','Insured','Payment']]
x=train.drop('Payment',axis =1)
y=train['Payment']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state = 42)

from sklearn.ensemble import RandomForestRegressor

param_grid = {
                 'n_estimators': range(10,110,10),
                 'max_depth': range(2,10)
             }
from sklearn.model_selection import GridSearchCV
clf = RandomForestRegressor()
grid_clf = GridSearchCV(clf, param_grid, cv=10)
grid_clf.fit(x_train, y_train)


print('train R2:',r2_score(y_train , grid_clf.predict(x_train)),'test R2:',r2_score(y_test, grid_clf.predict(x_test)))
df['PLS'] = grid_clf.predict(df[['Claims','Insured']])
df.drop(['Claims','Insured'] , axis = 1 , inplace = True)
df['Kilometres'] = df['Kilometres'].apply(lambda x:str(x))
df['Zone'] = df['Zone'].apply(lambda x:str(x))
df['Bonus'] = df['Bonus'].apply(lambda x:str(x))
df['Make'] = df['Make'].apply(lambda x:str(x))
df = pd.get_dummies(df)
df.head()
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

mms = MinMaxScaler()
mms.fit(df)
data_transformed = mms.transform(df)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
knn = KMeans(n_clusters=5).fit(data_transformed)
df['cluster'] = knn.predict(data_transformed)
cluster_means = df.groupby('cluster')['Payment'].mean().reset_index()
cluster_means.columns = ['cluster','mean']
df = df.merge(cluster_means , on = 'cluster' )
df.drop('cluster' , axis = 1 , inplace = True)
df.head()
df['avg'] = (df['PLS']+df['mean'])/2
from sklearn.linear_model import LinearRegression
x=df.drop(['Payment','PLS','mean'],axis =1)
y=df['Payment']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state = 42)
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)

model = LinearRegression().fit(scaled_x_train , y_train)
from sklearn.metrics import mean_squared_error as mse
print('training RMSE:',np.sqrt(mse(np.expm1(y_train) , np.expm1(model.predict(scaled_x_train)))))
print('test RMSE:',np.sqrt(mse(np.expm1(y_test), np.expm1(model.predict(scaled_x_test)))))
print(r2_score(y_test,model.predict(scaled_x_test)))
print(r2_score(y_train,model.predict(scaled_x_train)))
