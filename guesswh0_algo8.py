# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

path = "../input/sample_data_intw.csv"



# Any results you write to the current directory are saved as output.
ds = pd.read_csv(path)
ds.columns
ds.info()
ds.describe()
ds.shape
ds.head(15)
ds.pcircle.unique()
df = ds[['label', 'aon', 'daily_decr30', 'daily_decr90',

       'rental30', 'rental90', 'last_rech_date_ma', 'last_rech_date_da',

       'last_rech_amt_ma', 'cnt_ma_rech30', 'fr_ma_rech30',

       'sumamnt_ma_rech30', 'medianamnt_ma_rech30', 'medianmarechprebal30',

       'cnt_ma_rech90', 'fr_ma_rech90', 'sumamnt_ma_rech90',

       'medianamnt_ma_rech90', 'medianmarechprebal90', 'cnt_da_rech30',

       'fr_da_rech30', 'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30',

       'amnt_loans30', 'maxamnt_loans30', 'medianamnt_loans30', 'cnt_loans90',

       'amnt_loans90', 'maxamnt_loans90', 'medianamnt_loans90', 'payback30',

       'payback90']]
df.shape
from sklearn.model_selection import train_test_split

X = df[['aon', 'daily_decr30', 'daily_decr90',

       'rental30', 'rental90', 'last_rech_date_ma', 'last_rech_date_da',

       'last_rech_amt_ma', 'cnt_ma_rech30', 'fr_ma_rech30',

       'sumamnt_ma_rech30', 'medianamnt_ma_rech30', 'medianmarechprebal30',

       'cnt_ma_rech90', 'fr_ma_rech90', 'sumamnt_ma_rech90',

       'medianamnt_ma_rech90', 'medianmarechprebal90', 'cnt_da_rech30',

       'fr_da_rech30', 'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30',

       'amnt_loans30', 'maxamnt_loans30', 'medianamnt_loans30', 'cnt_loans90',

       'amnt_loans90', 'maxamnt_loans90', 'medianamnt_loans90', 'payback30',

       'payback90']]

y = df['label']
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .3, random_state = 3)
from sklearn.ensemble import RandomForestClassifier
#testing using different max_features

RANDOM_STATE=5

clfs = [

           RandomForestClassifier(n_estimators=100,

                                oob_score=True,

                               max_features="sqrt",

                               random_state=RANDOM_STATE),

        RandomForestClassifier(n_estimators=100,

                                max_features='log2',

                               oob_score=True,

                               random_state=RANDOM_STATE),

        RandomForestClassifier(n_estimators=100,

                                max_features=None,

                               oob_score=True,

                               random_state=RANDOM_STATE)

]
#clf=RandomForestClassifier(n_estimators=100)

y_pred=[]

#Train the model using the training sets y_pred=clf.predict(X_test)

for i in range(3):

    clfs[i].fit(X_train,y_train)

    y_pred.append(clfs[i].predict(X_test))

    print(clfs[i].score(X_test,y_test))
#performance on training set

for i in range(3):

    print(clfs[i].oob_score_)
from sklearn.metrics import accuracy_score

for i in range(3):

    print(accuracy_score(y_test,y_pred[i]))
clfs = clfs[1]

y_pred= y_pred[1]
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
import matplotlib.pyplot as plt

%matplotlib inline
def plot_confusion_matrix(y_true, y_pred,normalize=False,cmap=plt.cm.Blues):

    title = 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)

    classes = ['tn', 'fp', 'fn', 'tp']

    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    fmt = 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred)

plt.show()
import pickle

#saving model

filename = 'final_model.sav'

pickle.dump(clfs, open(filename, 'wb'))

final = X_test.copy()

final['true_label'] = y_test.tolist()

final['pred_label'] = y_pred.tolist()
final.head(5)
final.to_csv('pred.csv')
#loading saved model



loaded_model = pickle.load(open(filename, 'rb'))