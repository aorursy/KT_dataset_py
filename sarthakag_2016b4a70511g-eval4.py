import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
train_data = np.load('eval-lab-4-f464/train.npy',allow_pickle=True)

test_data = np.load('eval-lab-4-f464/test.npy',allow_pickle=True)
X_train = np.array([train_data[i][1] for i in range(2275)])

X_test = np.array([test_data[i][1] for i in range(976)])

y_train=np.array([train_data[i][0] for i in range(2275)])
train_data.shape
X_train_new=[]

X_test_new=[]

for z in range(2275):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_train[z][i][j][k]/255.0)

    X_train_new.append(temp)
for z in range(976):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_test[z][i][j][k]/255.0)

    X_test_new.append(temp)
pca = PCA(n_components=100, svd_solver='randomized', whiten=True, random_state=42).fit(X_train_new)





# pca = PCA().fit(new_X_train)

 

X_test_pca = pca.transform(X_test_new)

X_train_pca = pca.transform(X_train_new)



 

svc = SVC(kernel='rbf', class_weight='balanced')

# param_grid = {'C': [3,5,7,9],

#               'gamma': [0.006,0.008,0.01]}

param_grid = {'C': [1, 5, 10, 15],

              'gamma': [0.005,0.01,0.02,0.03]}

# param_grid = {'C': [4,5,6,7],

#               'gamma': [0.01,0.06,0.008]}

grid = GridSearchCV(svc, param_grid, cv=5)

 

grid.fit(X_train_pca, y_train)

# print(grid.best_params_)
model = grid.best_estimator_

y_pred = model.predict(X_test_pca)

# y_pred
y_out = [[i,y_pred[i]] for i in range(len(y_pred))]

out_df = pd.DataFrame(data = y_out,columns = ['ImageId','Celebrity'])

out_df.to_csv(r'out_7.csv',index = False)