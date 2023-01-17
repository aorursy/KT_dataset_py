# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Lecture des fichiers

train=pd.read_csv('../input/train.csv',sep=',')

train.head(10)

#indexation 

train.set_index('PassengerId',inplace=True,drop=True)

train.head(10)
def parse_model_0(X):

    target= X.Survived

    X=X[['Fare','SibSp','Parch']]

    return X, target

X, y=parse_model_0(train.copy())

#cette fonction prend en paramètre un classifieur, la matrice X et la cible Y. Elle fait appel au modèle cross_val_score qui réalise 5 validations croisées

from sklearn.cross_validation import cross_val_score

def compute_score(clf,X,y):

    xval = cross_val_score(clf,X,y,cv=5)

    return np.mean(xval) 
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

compute_score(lr,X,y)
survived=train[train.Survived==1]

dead=train[train.Survived==0]

def plot_hist(feature, bins=20):

    x1 = dead[feature].dropna()

    x2 = survived[feature].dropna()

    plt.hist([x1,x2], label=['Victime','Survivant'],bins=bins)

    plt.legend(loc = 'upper left')

    plt.title('distribution relative de %s' %feature)

    plt.show()

 

plot_hist('Pclass')