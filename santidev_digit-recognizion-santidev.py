# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print('------ DIMENSIONES DE LOS DATA-FRAMES ------')

print('df_train: ', df_train.shape)

print('df_test: ', df_test.shape)
y = df_train['label']

X = df_train.drop(['label'], axis=1)
X_ = df_test
import matplotlib.pyplot as plt

import random



plt.figure(figsize=(16,9))

for i in range(16):

    i_rdn = random.randint(0,42000)

    digit = np.array(X.iloc[i_rdn]).reshape((28,28))

    plt.subplot(4, 4, i+1)

    plt.imshow(digit, cmap='gray')

    

plt.show()
from sklearn import preprocessing
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import decomposition
pca = decomposition.PCA(n_components=70)

pc = pca.fit_transform(X)
pc_df = pd.DataFrame(data = pc)

pc_df['label'] = y

pc_df.head()
sum(pca.explained_variance_ratio_)
X = pc_df.drop(['label'], axis=1)



pca = decomposition.PCA(n_components=70)

pc1 = pca.fit_transform(X_)



pc_df1= pd.DataFrame(data = pc1)

pc_df1

X_ = pc_df1
print(X.shape)

print(X_.shape)
def Scaler(data_train, data_test):

    scaler = preprocessing.StandardScaler()

    data_train = scaler.fit_transform(data_train)

    data_test = scaler.fit_transform(data_test)    

    

    return data_train, data_test
X, X_ = Scaler(X,X_)
from sklearn.model_selection import train_test_split

data_X, data_y = X, y



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.2,random_state=42,stratify=data_y)
from sklearn.svm import SVC



clf = SVC(C=100)

clf.fit(train_X, train_y)

clf_predictions = clf.predict(val_X)

acc_svc = clf.score(val_X,val_y)

print('ACCURACY OF SVC: ', acc_svc)

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(val_y,clf_predictions)



cm_df = pd.DataFrame(cm,

                     index = ['0','1','2','3','4','5','6','7','8','9'], 

                     columns = ['0','1','2','3','4','5','6','7','8','9'])

plt.figure(figsize=(10,7))

sns.heatmap(cm_df, annot=True)

plt.title('SVC Radial Kernel \nAccuracy:{0:.3f}'.format(acc_svc))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(train_X, train_y) 

knn_predictions = knn.predict(val_X)

acc_knn = knn.score(val_X,val_y)

print('ACCURACY OF SVC: ', acc_knn)
cm_knn = confusion_matrix(val_y,knn_predictions)



cm_df_knn = pd.DataFrame(cm_knn,

                     index = ['0','1','2','3','4','5','6','7','8','9'], 

                     columns = ['0','1','2','3','4','5','6','7','8','9'])

plt.figure(figsize=(10,7))

sns.heatmap(cm_df, annot=True)

plt.title('KNeighborsClassifier \nAccuracy:{0:.3f}'.format(acc_knn))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

rfc.fit(train_X,train_y)

rfc_predictions = rfc.predict(val_X)

acc_rfc = rfc.score(val_X,val_y)

print('ACCURACY OF RFC: ', acc_rfc)
cm_rfc = confusion_matrix(val_y,rfc_predictions)



cm_df_rfc = pd.DataFrame(cm_rfc,

                     index = ['0','1','2','3','4','5','6','7','8','9'], 

                     columns = ['0','1','2','3','4','5','6','7','8','9'])

plt.figure(figsize=(10,7))

sns.heatmap(cm_df, annot=True)

plt.title('RandomForestClassifier \nAccuracy:{0:.3f}'.format(acc_rfc))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
Y_pred = clf.predict(X_)



ImageId = []

idx = 1

for i in range(0,len(X_)):

    ImageId.append(idx)

    idx+=1

sample_submission = pd.DataFrame({

    'ImageId':ImageId,

    'Label':Y_pred

}) 





sample_submission.to_csv('submission.csv', index=False)

sample_submission.tail()
digit = np.array(df_test.iloc[27998]).reshape((28,28))

plt.imshow(digit, cmap='gray')