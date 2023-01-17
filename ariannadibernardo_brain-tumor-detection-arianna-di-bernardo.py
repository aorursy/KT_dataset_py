#importo le librerie necessarie
import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import keras
import numpy as np
import requests
import random 
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
%matplotlib inline
import cv2
sys.path.append('..')
!pip install imutils
import imutils
#installo e importo efficientnetB5
!pip install keras_efficientnets
!pip install efficientnet
!pip install git+https://github.com/qubvel/efficientnet
!git clone https://github.com/qubvel/efficientnet.git
    
from keras_efficientnets import EfficientNetB5
from keras.applications.imagenet_utils import decode_predictions
model = EfficientNetB5(weights='imagenet')
model.summary()
layer_name = 'global_average_pooling2d'          
intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

image_size = model.input_shape[1]
#leggo i dati in due formati utili per i due modelli di features extraction
yes=[]
yes2=[]
for i in range(300):
    try:
        yes.append(plt.imread('../input/brain-tumor-detection/yes/yes%s.jpg' %i))  
        yes2.append(cv2.imread('../input/brain-tumor-detection/yes/yes%s.jpg' %i, cv2.IMREAD_GRAYSCALE))  
    except:
        pass 
len(yes)
#disegno un campione di cervelli che presentano tumore
#come si può vedere le immagini sono molto varie
#di dimensioni diverse,
#alcune in bianco e nero (1 canale) altre a colori (3 canali, RGB)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

G=[yes[0], yes[1], yes[2], yes[3], yes[4]]

for ax, g in zip(axs.flat, G):
    ax.imshow(g)
    ax.set_title('brain with tumor')

plt.tight_layout()
plt.show()
no=[]
no2=[]
for i in range(300):
    try:
        no.append(plt.imread('../input/brain-tumor-detection/no/no%s.jpg' %i)) 
        no2.append(cv2.imread('../input/brain-tumor-detection/no/no%s.jpg' %i, cv2.IMREAD_GRAYSCALE)) 
    except:
        pass  

len(no)
#disegno un campione di cervelli sani
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

G=[no[0], no[1], no[2], no[3], no[4]]

for ax, g in zip(axs.flat, G):
    ax.imshow(g)
    ax.set_title('health brain')

plt.tight_layout()
plt.show()
print('initial dataset dimension:', len(yes+no))
print('the dataset is balanced:', 'positive:', len(yes), 'negative:', len(no))
#test set creation: creo un test set bilanciato
#prendo il 10% del dataset, essendo esso di piccole dimensioni 
#25 positive and 25 negative

test_yes=yes[:25]
test_no=no[:25]
test_yes2=yes2[:25]
test_no2=no2[:25]
#remove test set from the dataset
yes=yes[25:]
no=no[25:]
yes2=yes2[25:]
no2=no2[25:]
print('training set dimension:', len(yes+no))
print('the dataset is still bilanced:', 'positive:', len(yes), 'negative:', len(no))
#funzione che pulisce l'immagine, per il primo modello di features extraction
#tale funzione identifica il contorno della figura, nel nostro caso del cervello, e una volta identificati i 4 punti estremi del bordo
#ritaglia un rettangolo attorno all'immagine (tolgo le parti laterali delle immagini che sono senza significato perchè composte da pixel neri)
#ho richiesto le dimensioni dell'immagine finale uguali alle dimensioni dell'input che deve avere un'immagine per essere passata come
#input in efficientnetB5

def crop_image(img):   

                                       
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)      #Read the image in gray format (1 channel instead of 3)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)          #Image blurring for removing noise
    
    # threshold the image and then perform a series of erosions + dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]  
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

         
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #find the contours of the brain
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)     

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    
    #pudding
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    new_img = cv2.resize(new_img, dsize=(image_size,image_size))
        
    return new_img
#funzione che pulisce l'immagine, per il secondo modello di features extraction
#stessa funzione, ma è già "letta" in bianco e nero

def crop_image2(img):  

    thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]  
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

  
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    new_img = cv2.resize(new_img, dsize=(image_size,image_size))
        
    return new_img
#applico la funzione a tutte le immagini per fare preprocessing e lavorare con immagini della stessa dimensione e ben pulite

yes_pre=[]
yes_pre2=[]
no_pre=[]
no_pre2=[]

for i in range(len(yes)):
    try:
        yes_pre.append(crop_image(yes[i]))
        yes_pre2.append(crop_image2(yes2[i]))
    except:
        pass
    
    
for i in range(len(no)):
    try:
        no_pre.append(crop_image(no[i]))
        no_pre2.append(crop_image2(no2[i]))
    except:
        pass
#guardo i risultati su un campione

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

G=[yes_pre[0], yes_pre[1], no_pre[0], no_pre[1]]

for ax, g in zip(axs.flat, G):
    ax.imshow(g)
    ax.set_title('preprocessed image')

plt.tight_layout()
plt.show()
#per il test set
test_yes_pre=[]
test_yes_pre2=[]
test_no_pre=[]
test_no_pre2=[]

for i in range(len(test_yes)):
    try:
        test_yes_pre.append(crop_image(test_yes[i]))
        test_yes_pre2.append(crop_image2(test_yes2[i]))
    except:
        pass
    
    
for i in range(len(test_no)):
    try:
        test_no_pre.append(crop_image(test_no[i]))
        test_no_pre2.append(crop_image2(test_no2[i]))
    except:
        pass
#pass the images in efficientnet and prepare the features vectors
Fvector_yes=[]
for i in range(len(yes_pre)):
    try:
        x = np.expand_dims(yes_pre[i], 0)   #sistemo le dimensioni
        intermediate_output = intermediate_layer_model.predict(x)   #predizione che esce dal layer di interesse
        f_vector=np.reshape(intermediate_output, (np.size(intermediate_output), 1))   #sistemo le dimensioni
        Fvector_yes.append(f_vector)  #creo una lista che contiene i features vectors
    except:
        print(i)
Fvector_no=[]
rem=[]
for i in range(len(no_pre)):
    try:
        x = np.expand_dims(no_pre[i], 0)
        intermediate_output = intermediate_layer_model.predict(x)
        f_vector=np.reshape(intermediate_output, (np.size(intermediate_output), 1))
        Fvector_no.append(f_vector)  
    except:
        print(i)
        rem.append(i)      #la funzione non lavora su 5 immagini che sono dimensionalmente complesse, 
                           #vado a toglierle anche nel secondo metodo per avere stesso training set
        
#per la fase di test
Fvector_yes_test=[]
for i in range(len(test_yes_pre)):
    try:
        x = np.expand_dims(test_yes_pre[i], 0)
        intermediate_output = intermediate_layer_model.predict(x)
        f_vector=np.reshape(intermediate_output, (np.size(intermediate_output), 1))
        Fvector_yes_test.append(f_vector)  
    except:
        print(i)
Fvector_no_test=[]
for i in range(len(test_no_pre)):
    try:
        x = np.expand_dims(test_no_pre[i], 0)
        intermediate_output = intermediate_layer_model.predict(x)
        f_vector=np.reshape(intermediate_output, (np.size(intermediate_output), 1))
        Fvector_no_test.append(f_vector)  
    except:
        print(i)
#train label creation
yes_target=[]
no_target=[]

for i in range(len(Fvector_yes)):
    yes_target.append(1)
for i in range(len(Fvector_no)):
    no_target.append(0) 
#test label creation 
yes_target_test=[]
no_target_test=[]

for i in range(len(Fvector_yes_test)):
    yes_target_test.append(1)
for i in range(len(Fvector_no_test)):
    no_target_test.append(0) 
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
#normalize the train
for i in range(len(yes_pre2)):
    yes_pre2[i]=normalize(yes_pre2[i])
for i in range(len(no_pre2)):
    no_pre2[i]=normalize(no_pre2[i]) 
    
#normalize the test
for i in range(len(test_yes_pre2)):
    test_yes_pre2[i]=normalize(test_yes_pre2[i])
for i in range(len(test_no_pre2)):
    test_no_pre2[i]=normalize(test_no_pre2[i]) 
#applico nmf sulle immagini preprocessate

nmf_yes=[]
nmf_no=[]

nmf = NMF(n_components=1, random_state=0)  #nmf rango 1, per ottenere vettori

for i in range(len(yes_pre2)):
    nmf_yes.append(nmf.fit_transform(yes_pre2[i]))
for i in range(len(no_pre2)):
    if i!=111 and i!=113 and i!=120 and i!=121 and i!=122:   #da no_pre2 rimuovo rem, le immagini che mi hanno dato problemi nel metodo 1
        nmf_no.append(nmf.fit_transform(no_pre2[i]))
#nmf sul test set
nmf_yes_test=[]
nmf_no_test=[]

nmf = NMF(n_components=1, random_state=0)
for i in range(len(test_yes_pre2)):
    nmf_yes_test.append(nmf.fit_transform(test_yes_pre2[i]))
for i in range(len(test_no_pre2)):
    nmf_no_test.append(nmf.fit_transform(test_no_pre2[i]))
#creo il target per il train 
target_no_2=[]
for i in range(len(nmf_no)):
    target_no_2.append(0)
    
target_yes_2=[]
for i in range(len(nmf_yes)):
    target_yes_2.append(1) 
    
#per il test
target_no_2T=[]
for i in range(len(nmf_no_test)):
    target_no_2T.append(0)
    
target_yes_2T=[]
for i in range(len(nmf_yes_test)):
    target_yes_2T.append(1) 
#metto insieme i risultati e li mescolo
Fvector=Fvector_yes+Fvector_no   #features metodo 1
target=yes_target+no_target   

nmf=nmf_yes+nmf_no            #features metodo 2
target2=target_yes_2+target_no_2

target==target2  #True, posso usare solo uno dei due come target, che chiamo unicamente target
#shuffle train data
c = list(zip(Fvector, nmf, target))

random.shuffle(c)

Fvector, nmf, target = zip(*c) 
#faccio lo stesso per il test

test_Fvector=Fvector_yes_test+Fvector_no_test  #modello 1
test_target=yes_target_test+no_target_test

nmf_test=nmf_yes_test+nmf_no_test   #modello 2
target2_test=target_yes_2T+target_no_2T

test_target==target2_test  #True 
#shuffle test data
c = list(zip(test_Fvector,nmf_test, test_target))

random.shuffle(c)

test_Fvector, nmf_test, test_target = zip(*c) 
#definisco una funzione che calcoli true positive, false positive, true negative, false negative

def TF(y,pred):    #y=ground truth, pred=prediction del modello
    TN=0
    TP=0
    FN=0
    FP=0

    for i in range(len(pred)):
        if y[i]==0:
            if pred[i]==0:
                TN=TN+1
            else:
                FP=FP+1
        elif y[i]==1:
            if pred[i]==1:
                TP=TP+1
            else:
                FN=FN+1
                
    return(TP, TN, FP, FN)
#definisco una funzione che calcoli la metrica 'sensitivity'
#essa si serve dei valori di true positive, false positive, true negative, false negative

def sensitivity(y,pred):
    
    TP, TN, FP, FN = TF(y,pred)
    
    sens=(TP)/(TP+FN)    #def
    
    return(sens)
from sklearn import model_selection
from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
#sistemo le dimensioni del training set
a=np.shape(Fvector)[0]
b=np.shape(Fvector)[1]

Fvector=np.array(Fvector)
Fvector=Fvector.reshape(a,b)
#sistemo le dimensioni del test set
#per la successiva fase di test
a=np.shape(test_Fvector)[0]
b=np.shape(test_Fvector)[1]

test_Fvector=np.array(test_Fvector)
test_Fvector=test_Fvector.reshape(a,b)
print('vectors length:',np.shape(Fvector)[1],'... it is necessary dimensionality reduction in order to work with easiest objects')
#dim reduction: PCA (principal component analysis)
#questa tecnica di riduzione di dimensionalità, basata sulla fattorizzazione SVD, riduce dataset complessi a dimensioni minori
#in modo da preservarne le informazioni principali

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA

np.random.seed(0)

X = np.concatenate((Fvector,test_Fvector))        #train + test per avere stessa dimensione finale 
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)     #Non-linear dimensionality reduction (make use of a kernel rbf)
X_kpca = kpca.fit_transform(X)   
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X = pca.fit_transform(X)

np.shape(X)
#resplit in the original sets
Xtrain=X[:len(Fvector),:]
Xtest=X[len(Fvector):,:]
#matrix normalization
Xtrain=normalize(Xtrain)
Xtest=normalize(Xtest)
#cross validation
#support vector machine con kernel polinomiale

kf = KFold(n_splits=20, random_state=None, shuffle=False)
kf.get_n_splits(Xtrain)
targetcv=np.array(target)
av=0
for train_index, test_index in kf.split(Xtrain):
    X_train, X_test = Xtrain[train_index], Xtrain[test_index]
    y_train, y_test = targetcv[train_index], targetcv[test_index]
    SVMcv=svm.SVC(C=15, kernel='poly', probability=True).fit(X_train, y_train)   #c=1, rbf
    av=av+sensitivity(SVMcv.predict(X_test), y_test)
    print(sensitivity(SVMcv.predict(X_test), y_test))
print('average sensitivity with cross validation:',av/20)     #0.74
#divido il training set in validation set e training set, per la fase di model evaluation ma soprattutto per la regressione finale
#stesso random seed per i due modelli
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Xtrain,target,random_state = 50, test_size=0.20)  
#support vector machine con kernel polinomiale
SVM=svm.SVC(C=15, kernel='poly', probability=True).fit(train_x, train_y) 
p=SVM.predict_proba(valid_x)[:,1]   #preparo le prediction per la regressione logistica finale (per unire i risultati dei due modelli)
                                    #lista che esprime per ogni paziente, la probabilità che esso sia malato (P di ottenere 1)
#sistemo le dimesnioni del training e del test set
M=np.transpose((np.transpose(nmf)[0])) 
N=np.transpose((np.transpose(nmf_test)[0]))
#cross validation
#support vector machine con kernel rbf

kf = KFold(n_splits=20, random_state=None, shuffle=False)
kf.get_n_splits(M)
targetcv=np.array(target)
av=0
for train_index, test_index in kf.split(M):
    X_train, X_test = M[train_index], M[test_index]
    y_train, y_test = targetcv[train_index], targetcv[test_index]
    SVMcv=svm.SVC(C=2, kernel='rbf', probability=True).fit(X_train, y_train)  
    av=av+sensitivity(SVMcv.predict(X_test), y_test)
    print(sensitivity(SVMcv.predict(X_test), y_test))
print('average sensitivity with cross validation:',av/20)   #0.71
#divido il training set in validation set e training set, per la fase di model evaluation ma soprattutto per la regressione finale
#stesso random seed per i due modelli
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(M, target, random_state = 50, test_size=0.20)
#support vector machine con kernel rbf
SVM2 = svm.SVC(C=2, kernel='rbf', probability=True).fit(train_x, train_y)  
q=SVM2.predict_proba(valid_x)[:,1] #preparo le pred per la regressione logistica finale (per unire i risultati dei due modelli)
#organizzo i dati in una tabella
pred_valid = pd.DataFrame({"target": valid_y,"pred1": p, "pred2": q}) 
pred_valid
#feature matrix and target
Xf = pred_valid.iloc[:, 1:3]
yf = pred_valid.target
#train-valid split
SEED = 30
x_trainLR , x_validationLR, y_trainLR , y_validationLR = model_selection.train_test_split(Xf, yf, test_size = 0.1,random_state = SEED)
#uso gridsearch per avere un'idea dei parametri ottimali (ma per l'accuracy...)
#poi provo a modificare i parametri che ottengo

GSL=GridSearchCV(estimator=linear_model.LogisticRegression(),
             param_grid={'multi_class':('auto', 'ovr', 'multinomial'), 'C': (0.001,0.1,0.01,0.8,0.9,1,1.1,1.2,2,10), 'max_iter': (100,200,300,400), 'solver': ('newton-cg', 'lbfgs', 'sag', 'saga')})
GSL=GSL.fit(Xf, yf)
print ("best parameter choice:", GSL.best_params_)  
#cross validation
#logistic reegression

kf = KFold(n_splits=10, random_state=None, shuffle=False)
kf.get_n_splits(Xf)
targetcv=np.array(yf)
Xfcv=np.array(Xf)
av=0
for train_index, test_index in kf.split(Xfcv):
    X_train, X_test = Xfcv[train_index], Xfcv[test_index]
    y_train, y_test = targetcv[train_index], targetcv[test_index]
    lrClfcv = LogisticRegression(C = 1, max_iter= 100, multi_class= 'multinomial', solver= 'newton-cg').fit(X_train, y_train) 
    av=av+sensitivity(lrClfcv.predict(X_test), y_test)
    print(sensitivity(lrClfcv.predict(X_test), y_test))
print('average sensitivity with cross validation:',av/10)  #0.74
#rialleno e prendo le prediction (ovviamente con gli iperparametri prima trovati)

SVM_=svm.SVC(C=15, kernel='poly', probability=True).fit(Xtrain,target)
p2=SVM_.predict_proba(Xtest)[:,1] 

SVM2_ = svm.SVC(C=2, kernel='rbf', probability=True).fit(M, target)
q2=SVM2_.predict_proba(N)[:,1]
#organizzo i dati in una tabella
pred_test= pd.DataFrame({"target": test_target,"pred1": p2, "pred2": q2}) 
pred_test
#test feature matrix and test target
Xf2 = pred_test.iloc[:, 1:3]
yf2 = pred_test.target
#rialleno e prendo le prediction 
lrClf = LogisticRegression(C= 1, max_iter= 100, multi_class= 'multinomial', solver= 'newton-cg').fit(Xf, yf)  #c=0.1

final_pre=lrClf.predict(Xf2)         #final prediction
Pfinal_pre=lrClf.predict_proba(Xf2)  #final prob prediction 
sensitivity(list(yf2), final_pre)  #risultato finale: sensitivity 0.9
# per una visione d'insieme:
# Roc Curve (Receiver operating characteristic Curve) and AUC (area under the curve)
# auc = 0.86, risultati buoni ma non ottimi (auc circa 1 per un modello ottimo, che classifica perfettamente)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(test_target,Pfinal_pre[:,1])
roc_auc = auc(fpr, tpr) 

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)'% roc_auc )

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()