from glob import glob



train_images = glob ('../input/train/train/*/*.jpg')
train_images[:3], len(train_images)
from scipy import stats

def image_to_features(image, easy = True):

    features = []

    np_image = np.array(image) #преобразуем в массив

    split1 = np.array_split(np_image, 7,axis = 0)

    split2 = np.array_split(np_image, 7,axis = 1)

    for im_part in [np_image]+split1[:2]+split1[-2:]+split2[:2]+split2[-2:]:

#     for im_part in [np_image]+split1+split2:

        for i in range(3):

            if easy:

                features += [im_part[:,:,i].mean(), im_part[:,:,i].std()]

            else:

                s = stats.describe(im_part[:,:,i],axis = None)

                features += [s.mean,

                            s.variance,

                            s.minmax[0],

                            s.minmax[1],

                            s.skewness,

                            s.kurtosis]

            

    return features
%%time

from PIL import Image

import numpy as np



indoor=[]

outdoor=[]



easy = 0



for i,path in enumerate(train_images):

#     if i%60 != 0: continue

    image = Image.open(path) #открваем картинку



    image_class = path.split('/')[-2] #определяем класс

    

    if image_class == 'indoor':

        indoor.append(image_to_features(image, easy))

    else:

        outdoor.append(image_to_features(image, easy))
X=np.array(indoor+outdoor)

y=np.array([0 for i in range(len(indoor))]+[1 for i in range(len(outdoor))])



indices = np.random.permutation(X.shape[0])







# training_idx, test_idx = indices[:80], indices[80:]

# X_test, X_train = X[training_idx,:], X[test_idx,:]

# Y_test, Y_train = y[training_idx], y[test_idx]



X_train = X

Y_train = y





x_final = X_train / np.linalg.norm(X_train, axis = 0)

y = Y_train



alpha=0.3

n = x_final.shape[0]

x0=np.ones((n,1))

beta=np.random.rand(1,x_final.shape[1])

for i in range(20000):

    y_pred=np.dot(x_final,beta.T)

    cost= np.sum((y_pred-y.reshape(n,1))**2)/n

    diff= (y_pred-y.reshape(n,1))

    derivative= 2*np.dot(diff.T,x_final)/n

    beta= (beta)-alpha*(derivative)

#print(beta)    



# model = np.dot(X_test / np.linalg.norm(X_test, axis = 0),beta.T)

# model = (model-model.min())/(model.max()-model.min())



# print (roc_auc_score(Y_test, model))
import sys

if 'sklearn' in sys.modules:

    from sklearn.metrics import roc_auc_score

    print (roc_auc_score(y, model))

else:

    print ('sklearn is not avaliable')
test_images = glob ('../input/test/test/*.jpg')

test=[]

test_ids = []

for i,path in enumerate(test_images):

#     if i%60 != 0: continue

    image = Image.open(path) #открваем картинку



    imageid = int(path.split('/')[-1].replace('.jpg',''))

    test.append(image_to_features(image, easy))

    test_ids.append(imageid)
X_test = np.array(test)

test_predict = np.dot(X_test / np.linalg.norm(X_test, axis = 0),beta.T)

test_predict = (test_predict-test_predict.min())/(test_predict.max()-test_predict.min())
import pandas as pd

df = pd.DataFrame({'image_number':test_ids}).join(pd.DataFrame(test_predict,columns = ['prob_outdoor']))
df
df.to_csv('submission.csv',index=False)