import numpy as np

import pandas as pd



from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
train = np.load('/kaggle/input/eval-lab-4-f464/train.csv',allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
from skimage.color import rgb2gray



num_train = np.size(train,0)

num_test = np.size(test,0)

train_grey = np.zeros((num_train,50,50))

test_grey = np.zeros((num_test,50,50))



for i,element in enumerate(train):

    train_grey[i,:,:] = rgb2gray(element[1])



for i,element in enumerate(test):

    test_grey[i,:,:] = rgb2gray(element[1])

from skimage import feature



#x_hog_0 = feature.hog(train_grey[1])

train_daisy_linear = np.zeros((num_train, 5000))

test_daisy_linear = np.zeros((num_test, 5000))



for i,image in enumerate(train_grey):

    train_daisy_linear[i,:] = np.reshape(feature.daisy(image),(1,5000))

    

for i,image in enumerate(test_grey):

    test_daisy_linear[i,:] = np.reshape(feature.daisy(image),(1,5000))
from sklearn.decomposition import PCA

n_components = 200

train_grey_linear = np.reshape(train_grey,(num_train,2500))

test_grey_linear = np.reshape(test_grey,(num_test,2500))



#FERAL HOG TIME



pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(train_daisy_linear);



train_pca = pca.transform(train_daisy_linear)

test_pca = pca.transform(test_daisy_linear)





from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train[:,0] = le.fit_transform(train[:,0])



y_train = train[:,0]

y_train = y_train.astype(int)
from sklearn.model_selection import GridSearchCV



svc = SVC(kernel='rbf', class_weight='balanced')

svc.fit(train_pca,y_train)



y_pred = svc.predict(test_pca)
y_pred = y_pred.astype(int)

y_pred = le.inverse_transform(y_pred)
df_final = pd.DataFrame({'ImageId': test[:,0], 'Celebrity': y_pred})

df_final.to_csv('submission.csv', index=False)