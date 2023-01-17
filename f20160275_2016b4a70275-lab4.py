import numpy as np

from matplotlib import pyplot as plt

import pandas as pd
data = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
list(set(data[:,0]))
cols = ['ImageId','Celebrity']
df = pd.DataFrame(None,columns=cols)
df['ImageId'] = test[:,0]

df.head()
plt.imshow(data[0][1],interpolation='nearest')

plt.show()
from sklearn.preprocessing import LabelEncoder, tests

from sklearn.model_selection import train_test_split

le = LabelEncoder()

y = le.fit_transform(data[:,0])

X = [data[i][1].flatten() for i in range(data.shape[0])]



X_test = [test[i][1].flatten() for i in range(test.shape[0])]



# X_train, X_test, y_train, y_test = train_test_split(

#     X, y, test_size=0.33, random_state=42)
from sklearn.decomposition import PCA

 

pca = PCA(n_components=120, svd_solver='randomized', whiten=True, random_state=42).fit(X)

 

X_pca = pca.transform(X)

X_test_pca = pca.transform(X_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

 

svc = SVC(kernel='rbf', class_weight='balanced')

param_grid = {'C': [1,5,10,50],

              'gamma': [0.005,0.01,0.05,0.1]}

grid = GridSearchCV(svc, param_grid, cv=5,verbose=3,n_jobs=-1)
grid.fit(X_pca, y)

print(grid.best_params_)
model = grid.best_estimator_

y_pred = model.predict(X_test_pca)

y_pred
y_pred = le.inverse_transform(y_pred)
len(y_pred)
df['Celebrity'] = y_pred
df.to_csv('ans2.csv',index=False)