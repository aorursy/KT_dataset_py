import pandas as pd

import numpy as np
train_np = np.load("/kaggle/input/eval-lab-4-f464/train.npy",allow_pickle=True)

train_pd = pd.DataFrame(train_np.tolist(), columns=['name', 'image'])



test_np = np.load("/kaggle/input/eval-lab-4-f464/test.npy",allow_pickle=True)

test_pd = pd.DataFrame(test_np.tolist(), columns=['id', 'image'])
from sklearn.preprocessing import LabelEncoder, tests

from sklearn.model_selection import train_test_split

labels = LabelEncoder()

y_train = labels.fit_transform(train_pd['name'])

X_train = [train_pd['image'][i].flatten() for i in range(len(train_pd))]

X_test = [test_pd['image'][i].flatten() for i in range(len(test_pd))]
from sklearn.decomposition import PCA

n_components = 100

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=108).fit(X_train)
X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

 

svc = SVC(kernel='rbf', class_weight='balanced')

param_grid = {'C': [3,5,7,9],

              'gamma': [0.006,0.008,0.01]}

grid = GridSearchCV(svc, param_grid, cv=5,n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

y_pred = model.predict(X_test)
y_pred = labels.inverse_transform(y_pred)
ans = pd.DataFrame

cols= ['ImageId','Celebrity']

ans = pd.DataFrame(None,columns = cols)

ans['Celebrity'] = y_pred

ans['ImageId'] = test_pd['id']
ans.to_csv('sub.csv',index = False)