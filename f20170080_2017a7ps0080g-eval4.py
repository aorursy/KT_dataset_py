from IPython.core.display import display, HTML

display(HTML("<style>.container { width:98% !important; }</style>"))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score



import warnings

warnings.filterwarnings("ignore")
train = np.load('./eval-lab-4-f464/train.npy', allow_pickle=True)

test = np.load('./eval-lab-4-f464/test.npy', allow_pickle=True)



print(len(train))

print(len(test))
images_train_raw = []

labels_train = []



for i in range(len(train)):

    labels_train.append(train[i][0])

    images_train_raw.append(train[i][1])

    

images_train_raw = np.asarray(images_train_raw)

labels_train = np.asarray(labels_train)



images_train_raw.shape, labels_train.shape
images_test_raw = []



for i in range(len(test)):

    images_test_raw.append(test[i][1])

    

images_test_raw = np.asarray(images_test_raw)

images_test_raw.shape
ulabel, ucount = np.unique(labels_train,return_counts=True)

plt.figure(figsize=[20,6])

plt.barh(ulabel, ucount, color='k')

plt.grid(axis='x')

plt.xticks(np.arange(0,161,5))

plt.plot()

print("Baseline : ", max(ucount)/ucount.sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

labels_train = le.fit_transform(labels_train)

labels_train



le.classes_
for i,img in enumerate(images_train_raw[:1]):

    

#     print(images_train_raw[i])

    plt.imshow(img)

    plt.axis('off')

    plt.savefig("1.jpg",frameon=False)

    plt.show()
'''IMPORTS'''



from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.decomposition import PCA, KernelPCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier





def plot_cm(y_test, preds_test):

    m = confusion_matrix(y_test, preds_test)

    df_cm = pd.DataFrame(m)

    plt.figure(figsize = (10,7))

    sns.heatmap(df_cm, annot=True)

    plt.show()
images_train = images_train_raw.reshape(images_train_raw.shape[0], -1)

images_test = images_test_raw.reshape(images_test_raw.shape[0], -1)



images_train.shape, images_test.shape
''' ROBUST SCALER'''



ss = StandardScaler()

images_train = ss.fit_transform(images_train)

images_train.shape
'''PCA'''



y = 300



pca = PCA(y)



pca.fit(images_train)

images_combined = pca.transform(images_train)

print(images_train.shape)
'''TRAIN TEST'''

x = 100

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images_train[:,:x], labels_train, test_size=0.1)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# X_train, y_train = ADASYN().fit_resample(X_train, y_train)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
'''LDA'''



z = 20



lda = LinearDiscriminantAnalysis(n_components=z)

lda.fit(X_train, y_train)

X_train = lda.transform(X_train)

X_test = lda.transform(X_test)



X_train.shape ,X_test.shape, y_train.shape, y_test.shape
'''SVC'''



from sklearn.svm import SVC

svc = SVC(

    C = 1.1, 

    kernel='rbf', 

    tol = 0.13,

#     gamma = 0.1,

    random_state = 0

)



svc.fit(X_train, y_train)

preds_train = svc.predict(X_train)

preds_test = svc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
'''SVC'''



from sklearn.svm import SVC

svc = SVC(

#     C = 11, 

#     kernel='rbf', 

#     tol = 0.13,

#     gamma = 1/18,

#     random_state = 0

)



svc.fit(X_train, y_train)

preds_train = svc.predict(X_train)

preds_test = svc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
# plot_cm(y_test, preds_test)
'''XGBClassifier'''



# lr = 0.03, n_est = 300, max_depth=5



from xgboost import XGBClassifier



xgb = XGBClassifier(

#         max_depth = 5,

#         learning_rate=0.03,

#         n_estimators = 300,

#         reg_alpha = 0.2,

#         reg_lambda = 1

    )



xgb.fit(X_train, y_train)

preds_train = xgb.predict(X_train)

preds_test = xgb.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
'''LinearSVC'''



from sklearn.svm import LinearSVC

svc = LinearSVC(C = 1.3)

svc.fit(X_train, y_train)

preds_train = svc.predict(X_train)

preds_test = svc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
'''LGBMClassifier'''



from lightgbm import LGBMClassifier



model = LGBMClassifier(

#     boosting_type='gbdt',

#     n_estimators = 300,

#     learning_rate = 0.01, 

#     n_estimators=1000

    )



for i in range(17,18):

    model.fit(X_train[:,:i], y_train)

    train_preds = model.predict(X_train[:,:i])

    test_preds = model.predict(X_test[:,:i])

#     print(i-1)

    print(f1_score(train_preds, y_train, average='weighted'))

    print(f1_score(test_preds, y_test, average='weighted'))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

preds_train = rfc.predict(X_train)

preds_test = rfc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

abc.fit(X_train, y_train)

preds_train = abc.predict(X_train)

preds_test = abc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()

etc.fit(X_train, y_train)

preds_train = etc.predict(X_train)

preds_test = etc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(

    n_estimators=1000,

    learning_rate=0.001

)



gbc.fit(X_train, y_train)

preds_train = gbc.predict(X_train)

preds_test = gbc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
import skimage

from skimage.feature import hog

from skimage.io import imread

from skimage.transform import rescale
from sklearn.base import BaseEstimator, TransformerMixin



class RGB2GrayTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        return np.array([skimage.color.rgb2gray(img) for img in X])





class HogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, y=None, orientations=9,

                 pixels_per_cell=(9, 9),

                 cells_per_block=(3, 3), block_norm='L2-Hys'):

        self.y = y

        self.orientations = orientations

        self.pixels_per_cell = pixels_per_cell

        self.cells_per_block = cells_per_block

        self.block_norm = block_norm



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):



        def local_hog(X):

            return hog(X,

                       orientations=self.orientations,

                       pixels_per_cell=self.pixels_per_cell,

                       cells_per_block=self.cells_per_block,

                       block_norm=self.block_norm)



        try: # parallel

            return np.array([local_hog(img) for img in X])

        except:

            return np.array([local_hog(img) for img in X])
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import StandardScaler

import skimage



# create an instance of each transformer

grayify = RGB2GrayTransformer()

hogify = HogTransformer(

    pixels_per_cell=(8,8),

    cells_per_block=(3,3),

    orientations=9,

    block_norm='L2-Hys'

)

# scalify = RobustScaler()

scalify = StandardScaler()
# X_train

X_train_gray = grayify.fit_transform(images_train.reshape(images_train_raw.shape))

X_train_hog = hogify.fit_transform(X_train_gray)

X_train = scalify.fit_transform(X_train_hog)



X_train, X_test, y_train, y_test = train_test_split(X_train, labels_train, test_size=0.1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''PCA'''



y = 1296



images_combined = np.append(X_train, X_test, axis=0)



pca = PCA(

    n_components=y, 

)



pca.fit(images_combined)

images_combined = pca.transform(images_combined)

print(images_combined.shape)
x = 900



X_train = images_combined[:X_train.shape[0],:x]

X_test = images_combined[X_train.shape[0]:,:x]



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
'''SVC'''



from sklearn.svm import SVC

svc = SVC(

    C = 1.35, 

    kernel='rbf', 

    tol = 0.14,

#     gamma = 0.1,

#     random_state = 0

)



svc.fit(X_train, y_train)

preds_train = svc.predict(X_train)

preds_test = svc.predict(X_test)

print("Train f1: ", f1_score(preds_train, y_train, average='weighted'))

print("Test f1: ", f1_score(preds_test, y_test, average='weighted'))
X_train_gray = grayify.fit_transform(images_train.reshape(images_train_raw.shape))

X_train_hog = hogify.fit_transform(X_train_gray)

X_train = scalify.fit_transform(X_train_hog)
test_gray = grayify.fit_transform(images_test.reshape(images_test_raw.shape))

test_hog = hogify.fit_transform(test_gray)

test = scalify.fit_transform(test_hog)
'''PCA'''



y = 1296



images_combined = np.append(X_train, test, axis=0)



pca = PCA(

    n_components=y, 

)



pca.fit(images_combined)

images_combined = pca.transform(images_combined)

print(images_combined.shape)
x = 900



X_train = images_combined[:X_train.shape[0],:x]

test = images_combined[X_train.shape[0]:,:x]

y_train = labels_train



print(X_train.shape, test.shape, y_train.shape)
from sklearn.svm import SVC



svc = SVC(

    C = 1.35, 

    kernel='rbf', 

    tol = 0.14,

#     gamma = 0.1,

#     random_state = 0

)



svc.fit(X_train, y_train)

preds_test = svc.predict(test)
encoded_pred = [le.classes_[idx] for idx in preds_test]

encoded_pred[:10]
submission = pd.DataFrame({

    'ImageId' : np.arange(len(encoded_pred)),

    'Celebrity' : encoded_pred

})



submission.to_csv("sub6.csv", index=False)