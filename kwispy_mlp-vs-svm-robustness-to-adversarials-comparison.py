%matplotlib inline

import numpy as np

from sklearn import datasets

from sklearn import model_selection

import matplotlib.pyplot as plt 

from keras.utils.np_utils import to_categorical 

import pandas as pd 

from sklearn.neural_network import MLPClassifier

from sklearn import svm

import itertools

import tensorflow as tf
def F_standardize(X_):



    X_ = X_.astype('float32')

    X_ = X_ / 255

    #X_-=np.mean(X_,axis=0)

    X_=X_.reshape(X_.shape[0],784)

    print(X_.shape)

    #X -= np.mean(X, axis=0, keepdims=True) 

    #X /= (np.std(X, axis=0, keepdims=True) + 1e-16)

    return X_
def get_grad(clf_fct,x, alpha):

    x_orig= x.copy()

    grads=np.zeros((x_orig.ravel().shape[0]))

    preds_orig=clf_fct(x_orig.reshape(1,784))

    idx_max=np.argmax(preds_orig)

    for i,pixel in enumerate(x_orig):

        x_=x_orig.copy().ravel()

        x_[i]=x_[i]+alpha

        #print(np.array(preds_orig-clf_fct(x_.reshape(784,1))).shape)

        grads[i]=np.array(preds_orig-clf_fct(x_.reshape(1,784)))[0,idx_max]

    return grads
def get_advs(x_orig,adv_rate,clf_fct):

    advs=np.zeros((10,784))

    preds_advs=np.zeros((10))

    preds_orig=clf_fct(x_orig.reshape(1,784))

    pred_orig=np.argmax(preds_orig)

    dist=np.inf

    best_adv=None

    best_adv_pred=None

    is_found=False

    grads=get_grad(clf_fct,x_orig,0.1)

    #grads=get_grad2(clf_svm.predict_proba,Xno0[:2],0.1)





    for i,rate in enumerate(np.arange(0,adv_rate,adv_rate/10)):

        advs[i]=x_orig+np.sign(grads)*rate

        preds_advs[i]=np.argmax(clf_fct(advs[i].reshape(1,784)))

        if pred_orig != preds_advs[i] and is_found==False:

            dist=np.linalg.norm(advs[i]-x_orig)

            best_adv=advs[i]

            best_adv_pred=preds_advs[i]

            is_found=True

    return best_adv,best_adv_pred,advs,preds_advs,dist 
def get_grads_multi(clf_fct,x, alpha):

    x_orig= x.copy()

    preds_orig=clf_fct(x_orig.reshape(x_orig.shape[0],784))

    grads=np.zeros((x_orig.shape))

    idxs_max=np.argmax(preds_orig,axis=1)

    #print(grads.shape)

    for i,img in enumerate(x_orig):

        #imgs = np.array(itertools.repeat(img, 784)) # 20 copies of "a"

        #print(imgs.shape)

        #rates = np.

        #imgs_grad=[ for j, imgj in enumerate(imgs)]

        #grads=np.zeros((x_orig.ravel().shape[0]))

        for j,pixel in enumerate(img):

            x_=img.copy().ravel()

            x_[j]=x_[j]+alpha

            #print(np.array(preds_orig-clf_fct(x_.reshape(784,1))).shape)

            #print(np.array(preds_orig[i]-clf_fct(x_.reshape(1,784))).ravel()[idxs_max[i]])

            grads[i,j]=np.array(preds_orig[i]-clf_fct(x_.reshape(1,784))).ravel()[idxs_max[i]]

    return grads
def get_accuracy(y_orig,predictions):

    y_origs=np.array([i for i in itertools.repeat(y_orig, 10)])

    #print(y_origs.shape)

    equals_=np.equal(y_origs,predictions.reshape(10,y_orig.shape[0]))

    #print(np.equal(y_origs,predictions.reshape(10,y_orig.shape[0])))

    accuracies=np.mean(equals_,axis=1)

    #print(accuracies)

    #print(y_orig.shape)

    #print(predictions.shape)

    return accuracies
def get_advs_multi(x_orig,y_orig,adv_rate,clf_fct):

    advs=np.zeros((10,784))

    preds_advs=np.zeros((10))

    preds_orig=clf_fct(x_orig.reshape(x_orig.shape[0],784))

    pred_orig=np.argmax(preds_orig, axis=1)

    #print(pred_orig)

    dist=np.inf

    best_adv=None

    best_adv_pred=None

    is_found=False

    grads=get_grads_multi(clf_svm.predict_proba,x_orig,0.1)

    grads_sign=np.sign(grads)

    advs_all=np.array([x_orig+i*grads_sign for i in np.arange(0,adv_rate,adv_rate/10)])

    #print(advs_all.shape)

    #print(advs_all.ravel().shape)

    predictions=np.argmax(clf_fct(advs_all.reshape(x_orig.shape[0]*10,784,order="C")),axis=1)

    #predictions=np.apply_along_axis(clf_fct,1,pertubations_all.reshape(20,784))

    #print(predictions.shape)

    advs=advs_all

    preds_advs=predictions.reshape(10,x_orig.shape[0])

    dist1=advs[:,:,:]-x_orig

    dist=np.linalg.norm(dist1,axis=2)

    accuracies=get_accuracy(y_orig,predictions)

    return advs,preds_advs,dist ,accuracies
(x_traink, y_traink), (x_testk, y_testk) = tf.keras.datasets.mnist.load_data()
X,y=np.array(x_traink[:1000]),y_traink[:1000]
print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))

#print(set(y))



# X is (nbExamples, nbDim)

# y is (nbExamples,)



# --- Standardize data

X = F_standardize(X)



# --- Split between training set and test set

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)



# --- Convert to proper shape: (nbExamples, nbDim) -> (nbDim, nbExamples)

X_train = X_train.T

X_test = X_test.T



# --- Convert to proper shape: (nbExamples,) -> (1, nbExamples)

y_train = y_train.reshape(1, len(y_train))

y_test = y_test.reshape(1, len(y_test))



# --- Convert to oneHotEncoding: (1, nbExamples) -> (nbClass, nbExamples)

n_in = X_train.shape[0]

n_out = 10



print("X_train.shape: {}".format(X_train.shape))

print("X_test.shape: {}".format(X_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("y_train.shape: {}".format(y_train.shape))

print("y_test.shape: {}".format(y_test.shape))

print("n_in: {} n_out: {}".format(n_in, n_out))
XX=X_train.copy().T

#XX=XX.reshape(800,784)

XX.shape

Xno0=XX[y_train.reshape(y_train.shape[1])!=0]

yno0=y_train.reshape(-1)[y_train.reshape(-1)!=0].reshape(1,-1,)
Xno0.shape

mlp=MLPClassifier(hidden_layer_sizes=(100,20), max_iter=1000)

mlp.fit(X_train.T, y_train.ravel())

score = mlp.score(X_test.T, y_test.ravel())

print(score)
clf_svm= svm.SVC(C=20, gamma=0.001, probability=True)

clf_svm.fit(X_train.T,y_train.ravel())

clf_svm.score(X_test.T,y_test.ravel())
best_adv_svm,best_adv_pred_svm,advs_svm,preds_svm,dist_svm =get_advs(Xno0[0],0.2,clf_svm.predict_proba)
best_adv_mlp,best_adv_pred_mlp,advs_mlp,preds_mlp,dist_mlp =get_advs(Xno0[0],0.2,mlp.predict_proba)
plt.figure(figsize=(15,10))

plt.subplot(1,2,1)

plt.title("mlp pred {} dist {:.2f}".format(best_adv_pred_mlp,dist_mlp))

plt.imshow(best_adv_mlp.reshape(28,28))

plt.subplot(1,2,2)

plt.title("svm pred {} dist {:.2f}".format(best_adv_pred_svm,dist_svm))

plt.imshow(best_adv_svm.reshape(28,28))
plt.figure(figsize=(20,10))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.title("pred {}".format(preds_svm[i]))

    plt.imshow(advs_svm[i].reshape(28,28))

plt.show()
plt.figure(figsize=(20,10))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.title("pred {}".format(preds_mlp[i]))

    plt.imshow(advs_mlp[i].reshape(28,28))

plt.show()
advs_mlp_multi,preds_mlp_multi,dist_mlp_multi,accuracies_mlp_multi=get_advs_multi(Xno0[1:30],yno0.ravel()[1:30],0.2,mlp.predict_proba)
advs_svm_multi,preds_svm_multi,dist_svm_multi, accuracies_svm_multi =get_advs_multi(Xno0[1:30],yno0.ravel()[1:30],0.2,clf_svm.predict_proba)
mean_dist_mlp=np.mean(dist_mlp_multi,axis=1)

mean_dist_svm=np.mean(dist_svm_multi,axis=1)
plt.title("Accuracy vs l2")

plt.plot(mean_dist_mlp,accuracies_mlp_multi, label="mlp")

plt.plot(mean_dist_svm,accuracies_svm_multi, label="svm")





plt.legend()

#plt.scatter(range(10),accuracies2)