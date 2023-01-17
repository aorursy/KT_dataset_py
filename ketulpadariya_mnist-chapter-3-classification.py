# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "classification"



def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)



# Any results you write to the current directory are saved as output.
import scipy.io as sio
mat = sio.loadmat('/kaggle/input/mnist-original/mnist-original.mat')
mat
X = mat['data']

y = mat['label']
print(X.shape)# there is 784 rows and 70000 columns so i need to transpose it

X = X.T

print(X.shape)
print(y.shape)  # 1 row 70000 columns

y = y.T

y = y.reshape(-1)

print(y.shape)
some_digit = X[34000]

some_digit.shape
plt.imshow(some_digit.reshape(28,28),cmap = mpl.cm.binary,

          interpolation= 'nearest')

plt.axis('off')
plt.imshow(some_digit.reshape(28,28),#cmap = mpl.cm.binary,

          interpolation= 'nearest')

plt.axis('off')
plt.imshow(some_digit.reshape(28,28),cmap = mpl.cm.binary,

          )#interpolation= 'nearest')

plt.axis('off')
plt.imshow(some_digit.reshape(28,28))
y[34000]
def plot_digit(somedigit):

    plt.imshow(somedigit.reshape(28,28),cmap = mpl.cm.binary,interpolation='nearest')

    plt.axis('off')

    plt.show()
def plot_digits(instances,images_per_row,**options):

    

    size  = 28

    

    images_per_row = min(len(instances),images_per_row)

    

    images = [instance.reshape(size,-1) for instance in instances]

    

    n_rows  = (len(instances)-1) // images_per_row +1 

    

    row_images = []

    

    n_empty = n_rows * images_per_row -len(instances)

    

    images.append(np.zeros((size,size*n_empty)))

    

    for row in range(n_rows):

        

        rimages  = images [ row * images_per_row : (row+1) * images_per_row ]

        row_images.append(np.concatenate(rimages, axis = 1 ))

        

    image = np.concatenate(row_images,axis  = 0)

    plt.imshow(image,cmap = mpl.cm.binary,**options)

    plt.axis('off')    
plt.figure(figsize=(7,7))

example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]

plot_digits(example_images, images_per_row=10)

#save_fig("more_digits_plot")

plt.show()

instances = 274



im_per_rows =10

nrow = ( instances-1) // im_per_rows +1

print(f'n_of rows {nrow} +\n n_ of empty { nrow * im_per_rows - instances}')
shuffle_index = np.random.permutation(60000)

X_train,y_train = X[shuffle_index],y[shuffle_index]
X_test,y_test = X[60000:],y[60000:]
# for performance measure we use Trainning Binary clssifier

y_train_5 = (y_train == 5 )

y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(max_iter = 5 , tol = np.infty)

sgd_clf.fit(X_train,y_train_5,)

sgd_clf.predict([some_digit])
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone





skfolds = StratifiedKFold(n_splits= 3 ,random_state= 42)



for train_index,test_index in skfolds.split(X_train,y_train_5):

    clone_clf  = clone(sgd_clf)

    X_train_folds = X_train[train_index]

    y_train5_folds = (y_train_5[train_index])

    

    X_test_folds = X_train[test_index]

    y_test5_folds = (y_train_5[test_index])

    

    clone_clf.fit(X_train_folds,y_train5_folds)

    predictions = clone_clf.predict(X_test_folds)

    n_correct = np.sum(predictions == y_test5_folds)

    print(n_correct,len(predictions),len(y_test5_folds))
from sklearn.model_selection import cross_val_score



cross_val_score(estimator=sgd_clf,

               X = X_train,

               y = y_train_5,

               cv = 3,

               scoring= 'accuracy')
from sklearn.base import BaseEstimator



class Never5Classifier(BaseEstimator):

    

    def fit(self,X,y=None):

        pass

    

    def predict(self,X):

        return np.zeros((len(X),1),dtype = bool)
never_5_clf = Never5Classifier()

cross_val_score(never_5_clf,X_train,

               y_train_5,scoring='accuracy')
from sklearn.model_selection import cross_val_predict



y_train_predict = cross_val_predict(estimator= sgd_clf,

                                   X = X_train,

                                   y = y_train_5,

                                   cv = 3)

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_train_predict,y_train_5)
cm
from sklearn.metrics import precision_score,recall_score
precision_score(y_train_predict,y_train_5)
recall_score(y_train_predict,y_train_5)
sgd_clf
y_score = sgd_clf.decision_function([some_digit])
y_score
Threshold = 0000

y_score > Threshold # then it is positive 
Threshold = 200000

y_score > Threshold # then it is positive 
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

sgd_clf.fit(X_train, y_train_5)
y_scores = cross_val_predict(

    sgd_clf,

    X_train,

    y_train_5,

    cv =3,

    method = 'decision_function')
X_train.shape , y_train_5.shape 
y_scores =  np.array([sgd_clf.decision_function([i]) for i in X_train])



for i  in X_train:

    print(i.shape)

    print(i)

    break

sgd_clf.decision_function(X_train[i])
y_train_predict
from sklearn.metrics import precision_recall_curve

print('y_train_5',y_train_5.shape,' y_score shape',y_scores.shape)

precisions, recalls, thresholds =  precision_recall_curve(y_train_5,y_scores)

print('precisions ',precisions,'\n recalls' ,recalls,'\n thresholds ', thresholds )

print('precisions shape',precisions.shape,'recalls shape' ,recalls.shape, 'thresholds shape',thresholds.shape)
plt.figure(figsize=(18,8))

def plot_precision_recall_vs_thresold(precisions=precisions,recalls=recalls,thresholds=thresholds):

    plt.plot(thresholds,precisions[:-1],'b--',label = 'Precision')

    plt.plot(thresholds,recalls[:-1],'g-',label = 'Recall')

    plt.xlabel('Thresold')

#    plt.xticks([range(-125000,125000,35000)])

    plt.ylim([0,1])

    plt.legend()

    

plot_precision_recall_vs_thresold()
plt.plot(recalls,precisions)

plt.ylabel('prscisions')

plt.xlabel('recalls')

plt.show()
from sklearn.metrics import roc_curve

print('y_train_5',y_train_5.shape,' y_score shape',y_scores.shape)





false_positive_ratio,True_positive_ratio,thresold = roc_curve(y_train_5,y_scores)



print('fpr ratio',false_positive_ratio,'\n Tpr ratio',True_positive_ratio,thresold)

print('fpr ratio.shape' , false_positive_ratio.shape,'Tpr ratio.shape',True_positive_ratio.shape,thresold.shape)

print(f'size redcution {y_train_5.shape[0]/false_positive_ratio.shape[0]*100}%')
roc_curve(y_train_5,y_scores)
plt.figure(figsize= (12,7))



def plot_roc_cuve(fpr = false_positive_ratio, tpr = True_positive_ratio,label = None):

    plt.plot(fpr,tpr,linewidth =2 , label = label)

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend()

plot_roc_cuve()

plt.show()
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=10,random_state= 42)



y_scores_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = 'predict_proba')
y_train_5.shape
y_scores_forest.shape # it gives the probability of select and not)
help(cross_val_predict)
y_score_forest = y_scores_forest[:,1]
fpr_rfc, tpr_rfc  , threshold = roc_curve(y_train_5,y_score_forest)
plt.figure(figsize= (16,8))

plot_roc_cuve(fpr_rfc,tpr_rfc,'Random Forest Classifier')

plot_roc_cuve(label = 'Gradient Descent Algorithem')

plt.legend()

plt.show()
from sklearn.metrics import roc_auc_score



roc_auc_score(y_train_5,y_score_forest)
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])
some_digit_score = sgd_clf.decision_function([some_digit])
np.argmax(some_digit_score)
sgd_clf.classes_
from sklearn.multiclass import OneVsOneClassifier



ovoc_sgd = OneVsOneClassifier(SGDClassifier(random_state= 42))
ovoc_sgd.fit(X_train,y_train)
ovoc_sgd.predict([some_digit])
ovoc_sgd.classes_
ovoc_sgd.estimators_[:2]
len(ovoc_sgd.estimators_)
# random forest training on multiclasses

forest_clf.fit(X_train,y_train)
forest_clf.classes_
len(forest_clf.estimators_)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
# Validation_score



scores = cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring='accuracy')
print( f'score is {scores} \n mean: {scores.mean()} \n standard deviation {scores.std()}.')
from sklearn.preprocessing import StandardScaler



standardscaler = StandardScaler()

X_train_scaled = standardscaler.fit_transform(X_train)#.astype(np.float64))



scores = cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring='accuracy')

print( f'score is {scores} \n mean: {scores.mean()} \n standard deviation {scores.std()}.')
from sklearn.preprocessing import StandardScaler



standardscaler = StandardScaler()

X_train_scaled = standardscaler.fit_transform(X_train.astype(np.float64))



scores = cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring='accuracy')

print( f'score is {scores} \n mean: {scores.mean()} \n standard deviation {scores.std()}.')
y_train_pred = cross_val_predict( sgd_clf, X_train, y_train,cv = 3 )
conf_mx = confusion_matrix(y_true= y_train, y_pred= y_train_pred)

conf_mx
fig = plt.figure(figsize = (16,7))

plt.matshow(conf_mx,cmap = plt.cm.gray)

plt.show()
row_sum = np.sum(conf_mx,axis = 1 ,keepdims=True)

print(f'row sum on each row wise \n {row_sum} \n and its shape is {row_sum.shape} and data type is {type(row_sum)}')
norm_conf_mx = conf_mx  / row_sum

plt.matshow(norm_conf_mx,cmap = plt.cm.gray)

plt.show()
import matplotlib

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(18.5, 10.5)
np.fill_diagonal(norm_conf_mx,0)

plt.matshow(norm_conf_mx,cmap = plt.cm.gray)

plt.show()
cl_a ,cl_b = 3,5

#true_3 = y_train == cl_a



#true_5 = y_train == cl_b

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]

X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]

X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]

X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(16,16))

plt.subplot(221); plot_digits(X_aa[:30],images_per_row= 6)

plt.subplot(222); plot_digits(X_ab[:30],images_per_row= 6)

plt.subplot(223); plot_digits(X_ba[:30],images_per_row= 6)

plt.subplot(224); plot_digits(X_bb[:30],images_per_row= 6)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



y_graterthan_7 =(y_train >= 7)

y_odd = (y_train % 2 ==1)



y_train_multilabel = np.c_[y_graterthan_7,y_odd]



kneighborsclassifier = KNeighborsClassifier()

kneighborsclassifier.fit(X_train,y_train_multilabel)





kneighborsclassifier.predict([some_digit])
#y_train_knn_pred = cross_val_predict(kneighborsclassifier,X_train,y_train_multilabel,cv =3)
#f1_score(y_train,y_train_knn_pred,average = 'macro')
noise = np.random.randint(0,100,(len(X_train),784))

test_noise = np.random.randint(0,100,(len(X_test),784))
X_train_mc = X_train + noise

X_test_mc = X_test + test_noise



y_test_mc =  X_test

y_train_mc = X_train
kneighborsclassifier.fit(X_train_mc,y_train_mc)
clean_digit = kneighborsclassifier.predict([X_test_mc[3000]])

plot_digit(clean_digit),plot_digit(X_test_mc[3000])