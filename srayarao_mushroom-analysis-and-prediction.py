# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mushrooms = pd.read_csv('../input/mushrooms.csv')
mushrooms.head(2)
mushrooms.describe()
mushrooms.isnull().sum()
for x in mushrooms.columns:

    print(x," :", mushrooms[x].unique())
for x in mushrooms.columns:

    print(x," :")

    #print("\n")

    print(mushrooms[x].value_counts())

    print("\n")
mushrooms.drop(['veil-type','stalk-root'],axis=1,inplace=True)
from sklearn import preprocessing

labelEncoder = preprocessing.LabelEncoder()

for x in mushrooms.columns:

    mushrooms[x] = labelEncoder.fit_transform(mushrooms[x])
y = mushrooms['class']

X = mushrooms.drop(['class'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
def modelResults(model):

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    modelStr=str(model)[:str(model).find("(")]

    print("\n")

    print (color.BOLD + color.RED + modelStr + color.END)

    print("\n")

    print (color.BOLD + color.UNDERLINE + "Classification Report" + color.END)

    print(classification_report(y_test,y_pred))

    print("\n")

    print (color.BOLD + color.UNDERLINE + "Confusion Matrix" + color.END)

    print(confusion_matrix(y_test, y_pred))

    print("\n")

    if modelStr=="LogisticRegression":

        print(color.BOLD + color.UNDERLINE + "Accuracy" + color.END)

        print(logmodel.score(X_test,y_test))

        print("\n")
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



from sklearn.decomposition import PCA





NBclassifier = GaussianNB()

logmodel = LogisticRegression()

rfModel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

dtModel = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

svmModel = SVC(kernel = 'linear', random_state = 0)

knnModel = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



modelResults(NBclassifier)

modelResults(logmodel)

modelResults(rfModel)

modelResults(dtModel)

modelResults(svmModel)

modelResults(knnModel)



lda = LDA(n_components = 2)

X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)

logmodel = LogisticRegression(random_state = 0)

print (color.BOLD + color.BLUE + 'Linear Discriminant Analysis' + color.END)

modelResults(logmodel)