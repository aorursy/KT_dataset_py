# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

from sklearn.metrics import f1_score,precision_score,recall_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from scipy import sparse

x_train=sparse.load_npz('../input/stackoverflow-featurizing-preprocessed-data-tags/train_bi_gram.npz')

x_test=sparse.load_npz('../input/stackoverflow-featurizing-preprocessed-data-tags/test_ni_gram.npz')

y_train=sparse.load_npz('../input/stackoverflow-featurizing-preprocessed-data-tags/tags_vec_train.npz')

y_test=sparse.load_npz('../input/stackoverflow-featurizing-preprocessed-data-tags/tags_vec_test.npz')
x_train.shape,x_test.shape,y_train.shape,y_test.shape
classifier=OneVsRestClassifier(SGDClassifier(loss='log',alpha=0.00001,penalty='l1'),n_jobs=-1)

classifier.fit(x_train,y_train)
from sklearn.externals import joblib

joblib.dump(classifier,'logistic_reg_equal_wt.pkl')
# prediction=classifier.predict(x_test)

# metrics.accuracy(y_test,prediction)

# f1_score(y_test,prediction,average='macro')

# f1_score(y_test,prediction,average='micro')

# metrics.hamming_loss(y_test,prediction)

# metrics.classification_report(y_test,prediction)