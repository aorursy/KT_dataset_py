# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline 

import matplotlib as mpl 

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set_style("darkgrid")

mpl.rc("axes",labelsize=16)

mpl.rc("xtick",labelsize=14)

mpl.rc("ytick",labelsize=14)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.info()
test.info()
train.head()
train_label=train.label

train_data=train.drop("label",axis=1)

print(train_label)

print(train_data)
test.head()
from xgboost import XGBClassifier

xgb=XGBClassifier()

smldata=train_data[0:200]

smllabel=train_label[0:200]

smldata2=train_data[200:400]

smllabel2=train_label[200:400]

xgb.fit(smldata,smllabel)

initial_predictions=xgb.predict(smldata2)

print(initial_predictions[0:5])

print(train_label[200:205])
from sklearn.metrics import mean_absolute_error

init_diff=mean_absolute_error(smllabel,initial_predictions)

print(np.sqrt(init_diff))
train_all=train_data.to_numpy()

some_digit= train_data.loc[36000]

some_digit_image=some_digit.values.reshape(28,28)

plt.imshow(some_digit_image,cmap='Greys_r',interpolation="nearest")

plt.axis("off")

plt.show()
train_data.shape
def plot_digit(data):

    image=data.values.reshape(28,28)

    plt.imshow(image,cmap='Greys_r',interpolation='nearest')

    plt.axis("off")

    

plot_digit(train_data.loc[1])

print(type(train_data.loc[1]))
def plot_digits(instances, images_per_row=10, **options):

    flx=instances.to_numpy()

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in flx]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = "Greys_r", **options)

    plt.axis("off")

plt.figure(figsize=(10,10))

plot_digits(train_data[5:69],images_per_row=10)
shuffle_index=np.random.permutation(42000)

train_shuffle,label_shuffle=train_data.loc[shuffle_index],train_label.loc[shuffle_index]
train_5=(label_shuffle==5)
from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier(max_iter=5,tol=-np.infty,random_state=17)

sgd_clf.fit(train_shuffle,train_5)
sgd_clf.predict([some_digit])
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,train_shuffle,train_5,cv=3,scoring="accuracy")
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

skfolds=StratifiedKFold(n_splits=3)



for train_index,test_index in skfolds.split(train_shuffle, train_5):

    clone_clf=clone(sgd_clf)

    train_shuffle_folds=train_shuffle.loc[train_index]

    train_5_folds=train_5.loc[train_index]

    Test_shuffle_folds=train_shuffle.loc[test_index]

    Test_5_folds=train_5.loc[test_index]

    

    clone_clf.fit(train_shuffle_folds,train_5_folds)

    pred=clone_clf.predict(Test_shuffle_folds)

    n_correct=sum(pred==Test_5_folds)

    print(n_correct/len(pred))
from sklearn.base import BaseEstimator 

class Never5Estimator(BaseEstimator):

    def fit(self,X,y=None):

        pass

    def predict(self,X):

        return np.zeros((len(X),1),dtype=bool)

never_5_clf=Never5Estimator()

cross_val_score(never_5_clf,train_shuffle,train_5,cv=3,scoring="accuracy")
from sklearn.model_selection import cross_val_predict

train_pred=cross_val_predict(sgd_clf,train_shuffle,train_5,cv=3)
from sklearn.metrics import confusion_matrix

confusion_matrix(train_5,train_pred)
from sklearn.metrics import precision_score, recall_score

precision_score(train_5,train_pred)
2809/(2809+712)
recall_score(train_5,train_pred)
2809/(2809+986)
from sklearn.metrics import f1_score

f1_score(train_5,train_pred)
scores=sgd_clf.decision_function([some_digit])

scores
threshold=0

some_digit_pred=(scores >threshold)
some_digit_pred
threshold=140000

some_digit_pred=(scores >threshold)

some_digit_pred
scores=cross_val_predict(sgd_clf,train_shuffle,train_5,cv=3,method='decision_function')
scores.shape
from sklearn.metrics import precision_recall_curve

precisions, recalls , thresholds =precision_recall_curve(train_5,scores)

def plot_precision_recall_vs_thresholds(precisions,recalls,thresholds):

    plt.plot(thresholds,precisions[:-1],"b--",label="Precision",linewidth=2)

    plt.plot(thresholds,recalls[:-1],"g-",label="Recall",linewidth=2)

    plt.xlabel("Threshold",fontsize=16)

    plt.legend(loc="upper right",fontsize=16)

    plt.ylim([0,1])

    

plt.figure(figsize=(14,8))

plot_precision_recall_vs_thresholds(precisions,recalls,thresholds)

plt.xlim([-800000,700000])

plt.show()
(train_pred==(scores >0)).all()
train_pred_90=(scores > 170000)
precision_score(train_5,train_pred_90)
recall_score(train_5,train_pred_90)
def plot_precision_vs_recall(precisions,recalls):

    plt.plot(recalls,precisions,"b-",linewidth=2)

    plt.xlabel('Recall',fontsize=14)

    plt.ylabel('Precision',fontsize=14)

    plt.axis([0,1,0,1])

    

plt.figure(figsize=(14,8))

plot_precision_vs_recall(precisions,recalls)

plt.title("Precision vs Recall Plot")

plt.show()
from sklearn.metrics import roc_curve

fpr,tpr, thresholds=roc_curve(train_5,scores)

def plot_roc_curve(fpr,tpr,label=None):

    plt.plot(fpr,tpr,linewidth=2,label=label)

    plt.plot([0,1],[0,1],'r--')

    plt.axis([0,1,0,1])

    plt.xlabel('False Positive Rate',fontsize=16)

    plt.ylabel('True Postive Rate',fontsize=16)

    

plt.figure(figsize=(14,8))

plot_roc_curve(fpr,tpr)

plt.title('Recieving Operator Characterstic Curve')

plt.show()
from sklearn.metrics import roc_auc_score

roc_auc_score(train_5,scores)
from xgboost import XGBClassifier

xgb_clf=XGBClassifier(n_estimators=10,random_state=17)

proba_xgb=cross_val_predict(xgb_clf,train_shuffle,train_5,cv=3,method="predict_proba")
scores_xgb=proba_xgb[:,1]

fpr_xgb,tpr_xgb,thresholds_xgb=roc_curve(train_5,scores_xgb)
plt.figure(figsize=(14,8))

plt.plot(fpr,tpr,"b:",linewidth=2,label="SGD")

plot_roc_curve(fpr_xgb,tpr_xgb,"XGB")

plt.legend(loc="lower right",fontsize=16)

plt.title('Recieving Operator Characterstic Curve')

plt.show()
roc_auc_score(train_5,scores_xgb)
xgb_pred=cross_val_predict(xgb_clf,train_shuffle,train_5,cv=3)

precision_score(train_5,xgb_pred)
recall_score(train_5,xgb_pred)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

train_scaled=scaler.fit_transform(train_data)

test_scaled=scaler.fit_transform(test)
#xgb_clf.fit(train_scaled,train_label)

#predictions=xgb_clf.predict(test_scaled)

#Output=pd.DataFrame({"ImageId":np.arange(start=1,stop=28001),"Label":predictions})

#Output.to_csv('submissi1on_csv',index=False)

#print(Output)
sgd_clf.fit(train_data,train_label)

sgd_clf.predict([some_digit])
some_digit_scores=sgd_clf.decision_function([some_digit])

some_digit_scores
np.argmax(some_digit_scores)
sgd_clf.classes_[1]
#from sklearn.multiclass import OneVsOneClassifier

#ovo_clf=OneVsOneClassifier(SGDClassifier(max_iter=5,tol=np.infty,random_state=42))

#ovo_clf.fit(train_data,train_label)

#ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)
xgb_clf.fit(train_all,train_label)

some_digit=train_all[15030]

xgb_clf.predict(train_all[34:36])
xgb_clf,predict_proba(some_digit)
cross_val_score(sgd_clf,train_data, train_label,cv=3,scoring="accuracy")
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

train_scaled=scaler.fit_transform(train_data.astype(np.float64))

cross_val_score(sgd_clf,train_scaled,train_label,cv=3,scoring="accuracy")
predictions=cross_val_predict(sgd_clf,train_scaled,train_label,cv=3)

conf_mx=confusion_matrix(train_label,predictions)  
def plot_confusion_matrix(matrix):

    fig=plt.figure(figsize=(8,8))

    ax=fig.add_subplot(111)

    cax=ax.matshow(matrix)

    fig.clorbar(cax)
plt.matshow(conf_mx,cmap="Greens_r")

plt.show()
row_sums=conf_mx.sum(axis=1,keepdims=True)

norm_conf_mx=conf_mx/row_sums
np.fill_diagonal(norm_conf_mx,0)

plt.matshow(norm_conf_mx,cmap="Greens_r")

plt.show()
cl_a,cl_b=3,5

X_aa=train_data[(train_label==cl_a) & (predictions == cl_a)]

X_ab=train_data[(train_label==cl_a) & (predictions == cl_b)]

X_ba=train_data[(train_label==cl_b) & (predictions == cl_a)]

X_bb=train_data[(train_label==cl_b) & (predictions == cl_b)]



plt.figure(figsize=(8,8))

plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)

plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)

plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)

plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



train_large=(train_label >=7)

train_odd=(train_label % 2==1)

train_multilabel=np.c_[train_large,train_odd]



knn_clf=KNeighborsClassifier()

knn_clf.fit(train_data,train_multilabel)
knn_clf.predict([some_digit])
noise = np.random.randint(0, 100, (len(train), 784))

X_train_mod = train_all + noise

noise = np.random.randint(0, 100, (len(test), 784))

test_all=test.to_numpy()

X_test_mod = test + noise

y_train_mod = train

y_test_mod = test

X_test_mod.info()
some_index=5500

plt.subplot(121); plot_digit(X_test_mod.loc[some_index])

plt.subplot(122); plot_digit(y_test_mod.loc[some_index])

plt.show()