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
df=pd.read_csv('../input/Wine.csv',header=None)

df.head(2)
df.columns = [  'name'

                 ,'alcohol'

             	,'malicAcid'

             	,'ash'

            	,'ashalcalinity'

             	,'magnesium'

            	,'totalPhenols'

             	,'flavanoids'

             	,'nonFlavanoidPhenols'

             	,'proanthocyanins'

            	,'colorIntensity'

             	,'hue'

             	,'od280_od315'

             	,'proline'

                ]



df.head(2)
df.isnull().sum()
import seaborn as sns

corr = df[df.columns].corr()

sns.heatmap(corr, cmap="YlGnBu", annot = True)
X= df.drop(['name','ash'], axis=1)



X.head()
Y=df.iloc[:,:1]

Y.head(2)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)



print(X_train.shape)

print(X_test.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

models = []



models.append(("Logistic Regression:",LogisticRegression()))

models.append(("Naive Bayes:",GaussianNB()))

models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=3)))

models.append(("Decision Tree:",DecisionTreeClassifier()))

models.append(("Support Vector Machine-linear:",SVC(kernel="linear")))

models.append(("Support Vector Machine-rbf:",SVC(kernel="rbf")))

models.append(("Random Forest:",RandomForestClassifier(n_estimators=7)))

models.append(("eXtreme Gradient Boost:",XGBClassifier()))

models.append(("MLP:",MLPClassifier(hidden_layer_sizes=(45,30,15),solver='sgd',learning_rate_init=0.01,max_iter=500)))

models.append(("AdaBoostClassifier:",AdaBoostClassifier()))

models.append(("GradientBoostingClassifier:",GradientBoostingClassifier()))



print('Models appended...')
results = []

names = []

for name,model in models:

    kfold = KFold(n_splits=10, random_state=0)

    cv_result = cross_val_score(model,X_train,Y_train.values.ravel(), cv = kfold,scoring = "accuracy")

    names.append(name)

    results.append(cv_result)

for i in range(len(names)):

    print(names[i],results[i].mean()*100)