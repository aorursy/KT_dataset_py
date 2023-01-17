import numpy as np 

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

#del Unnamed: 32, id

del df['Unnamed: 32'],df['id']

df.head(3)
X = df.drop(['diagnosis'],axis=1).values

y = df['diagnosis']

y = (y == 'M') + 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

X_train.shape,X_test.shape
clf = LogisticRegression(max_iter=10000)

y_pred = clf.fit(X_train,y_train).predict(X_test)

f1 = f1_score(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

print(' features accuracy:',round(acc,4),'F1: ',round(f1,4))
w_shapes = [128,512,1024,4096]

w_sels=[32,128,256,1024]

for w_size,w_sel in zip(w_shapes,w_sels):

    feature_size = X.shape[1]

    input_weights = np.random.normal(size=[feature_size,w_size])

    new_feature_train = np.dot(X_train,input_weights)

    new_feature_test = np.dot(X_test,input_weights)

    

    clf = LogisticRegression(max_iter = 20000)

    y_pred = clf.fit(new_feature_train,y_train).predict(new_feature_test)

    f1 = f1_score(y_test,y_pred)

    acc = accuracy_score(y_test,y_pred)

    print('Features: ',w_size,'Accuracy: ',round(acc,4),'F1: ',round(f1,4))

    if w_sel < min(new_feature_train.shape[0], new_feature_train.shape[1]):



        pca = PCA(n_components=w_sel)

        new_feature_selected_train_pca = pca.fit_transform(new_feature_train, y_train)

        new_feature_selected_test_pca = pca.transform(new_feature_test)

        y_pred = clf.fit(new_feature_selected_train_pca,y_train).predict(new_feature_selected_test_pca)

        f1 = f1_score(y_test,y_pred)

        acc = accuracy_score(y_test,y_pred)

        print('Selected features <pca>: ',w_sel,'Accuracy: ',round(acc,4),'F1:',round(f1,4))

    else:

        print("PCA doesn't work!")

    print('')