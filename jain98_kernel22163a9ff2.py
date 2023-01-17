import numpy as np

import pandas as pd

loaded_data = np.load('train.npy',allow_pickle='true')

frame_data = pd.DataFrame(loaded_data)
from skimage.feature import daisy

X = []

for i in range(2275):

    X.append(frame_data.iloc[i,1].ravel())



raw_y = np.array(frame_data.iloc[:,0])



from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

y = encoder.fit_transform(raw_y)

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.decomposition import PCA



n_components = 130

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

X_pca=pca.transform(X)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



# svc = SVC(kernel='rbf', class_weight='balanced')

# param_grid = {'C': [1, 5, 10, 50],

#               'gamma': [0.0001, 0.0005, 0.001, 0.005]}

# grid = GridSearchCV(svc, param_grid, cv=5)



# grid.fit(X_train_pca, y_train)

# print(grid.best_params_)

# #{'C': 5, 'gamma': 0.005}

model=SVC(kernel='rbf', class_weight='balanced',C=4,gamma = 0.1)

model.fit(X_train_pca, y_train)
#model = grid.best_estimator_



y_pred = model.predict(X_test_pca)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
loaded_test_data = np.load('test.npy',allow_pickle='true')

frame_test_data = pd.DataFrame(loaded_test_data)

X_task = []

for i in range(976):

    X_task.append(frame_test_data.iloc[i,1].ravel())

X_task_pca = pca.transform(X_task)

model = SVC(kernel='rbf', class_weight='balanced',C= 3, gamma= 0.1)
model.fit(X_pca,y)
y_pred_task = model.predict(X_task_pca)
y_ans=encoder.inverse_transform(y_pred_task)
y_ans
#storing y_test in reuired format

ID = frame_test_data[0]

# #y_test= y_test.reshape(len(y_test),1)

ans = pd.concat([ID,pd.DataFrame(y_ans)],axis=1)

#check the things

ans.columns=["ImageId","Celebrity"]

ans["ImageId"]=ans["ImageId"].astype('int64')

ans.dtypes
ans.to_csv("submit4.csv",index=None,header=["ImageId","Celebrity"])
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



explained_variance = pca.explained_variance_

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=19, random_state=0)

kmeans.fit(X)
y_pred = kmeans.predict(X_test)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred, average='weighted')  
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, random_state=0)

clf.fit(X_train, y_train)
explained_variance