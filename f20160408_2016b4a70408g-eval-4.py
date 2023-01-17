import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
data = np.load('/kaggle/input/eval-lab-4-f464/train.npy', allow_pickle=True)

X = [data[i][1] for i in range(data.shape[0])]

X = np.array(X)

X.shape



y = [data[i][0] for i in range(data.shape[0])]

y = np.array(y)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit_transform(y)
test_data = np.load('/kaggle/input/eval-lab-4-f464/test.npy', allow_pickle=True)



X_test = [test_data[i][1] for i in range(test_data.shape[0])]

X_test = np.array(X_test)



test_ind = [test_data[i][0] for i in range(test_data.shape[0])]

test_ind = np.array(test_ind)

X_test.shape
data.shape[0]
from skimage.color import rgb2grey

X_grey = rgb2grey(X)

# X_grey = X

X_grey.shape
X_test = rgb2grey(X_test)

X_test.shape
# import matplotlib as mpl

# plt.imshow(X_grey[2], cmap=mpl.cm.gray)

# plt.show()
X_flat = np.reshape(X_grey, (X_grey.shape[0], -1), order='C')

X_flat.shape
X_test = np.reshape(X_test, (X_test.shape[0], -1), order='C')

X_test.shape
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



# ss = StandardScaler()

# X_ss = ss.fit_transform(X_flat)

pca = PCA(n_components=100, svd_solver='randomized', whiten=True, random_state=42)

X_pca = pca.fit_transform(X_flat)
# X_test = ss.transform(X_test)

X_test = pca.transform(X_test)
from sklearn.svm import SVC



svm = SVC(C=0.01, kernel='linear', class_weight='balanced',probability=True, random_state=42)



svm.fit(X_pca, y)
# y_gen = svm.predict(X_val)

# y_gen
# from sklearn.metrics import f1_score

# f1_score(y_val, y_gen, average='macro')
y_pred = svm.predict(X_test)



df = pd.DataFrame(None,columns=['ImageId','Celebrity'])

df['ImageId'] = test_ind

df['Celebrity'] = y_pred



df.to_csv('pred.csv',index=False)
df.shape