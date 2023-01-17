# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# remove warnings

import warnings

warnings.filterwarnings('ignore')

# ---



%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np



np.random.seed(2)



pd.options.display.max_rows = 100



# Load the data and feature engineer

train = pd.read_csv("../input/sample.csv",index_col=False,header=None)

#drop outliner

train.drop(labels=[29115,52648],axis=0,inplace=True)

train.shape
#feature engineers



#there are 4 float64 columns, all are possitive

float_cols = [col for col in train.columns

              if(train[col].dtype == np.float64)]

train.loc[:,float_cols]=train.loc[:,float_cols]/train[float_cols].max()



#there are three int cols that are not binary, 

# I feature engineer them to catergorical cloumns

train['col4a'] = train[4].map(lambda s: 1 if s == 0 else 0)

train['col4b'] = train[4].map(lambda s: 1 if 1<=s<=2 else 0)

train['col4c'] = train[4].map(lambda s: 1 if 3<=s<=4 else 0)

train['col4d'] = train[4].map(lambda s: 1 if 5<=s else 0)



train['col23a'] = train[23].map(lambda s: 1 if s == -1 else 0)

train['col23b'] = train[23].map(lambda s: 1 if 1<=s<=4 else 0)

train['col23c'] = train[23].map(lambda s: 1 if 5<=s else 0)



train['col36a'] = train[36].map(lambda s: 1 if s == 1 else 0)

train['col36b'] = train[36].map(lambda s: 1 if (s == 0 or s==3) else 0)

train['col36c'] = train[36].map(lambda s: 1 if s==2 else 0)

train['col36d'] = train[36].map(lambda s: 1 if (4<=s<=7) else 0)

train['col36e'] = train[36].map(lambda s: 1 if (8<=s or s<0) else 0)



train.drop(labels=[4,23,36], axis =1, inplace=True)

train.shape
#oversampling

np.random.seed(2)

trainA=train[train[295]=='A']

trainB=train[train[295]=='B']

trainB=trainB.sample(n=2500)

trainC=train[train[295]=='C']

trainC=trainC.sample(n=2500)

trainD=train[train[295]=='D']

trainD=trainD.sample(n=2500)

trainE=train[train[295]=='E']







train_sample = pd.concat([ trainB, trainC, trainD,trainE])

#randomly shuffle the rows, reset the index

train = train_sample.sample(frac=1).reset_index(drop=True)

train.shape
#split the label from the data

Y_train = train[295]



# Drop 'label' column

X_train = train.drop(labels = [295],axis = 1) 



import seaborn 

g = seaborn.countplot(Y_train)
#feature engineers

#float cols -> devide by maximum
#X_train=X_train[Zero_filter]

X_train.shape
#create one-hot vector

#Y_train=pd.get_dummies(Y_train)

Y_train.shape
# Set the random seed

random_seed = 2



# Split the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)

X_train.shape
X_val.shape
from sklearn import datasets

from sklearn.multiclass import OneVsRestClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import LinearSVC

#iris = datasets.load_iris()

#X, y = iris.data, iris.target
#y


#OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

#X.shape
#y.shape
model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)

#model=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
Y_pred=model.predict(X_val)
(Y_pred==Y_val).describe()
import seaborn as seaborn

%matplotlib inline

g = seaborn.countplot(Y_pred)

g = seaborn.countplot(Y_val)
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

# compute the confusion matrix

from sklearn.metrics import confusion_matrix

import itertools

confusion_mtx = confusion_matrix(Y_val, Y_pred) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(5)) 

