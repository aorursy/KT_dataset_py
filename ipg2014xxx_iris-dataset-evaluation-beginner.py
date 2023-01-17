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
iris = pd.read_csv("../input/Iris.csv") #load the dataset

iris.head(2)
iris.info() 
from sklearn.preprocessing import LabelEncoder

import numpy as np



try:

    iris.drop('Id', axis=1, inplace=True)

except:

    pass



y = iris['Species']

print ('Class Labels =', np.unique(y))

le = LabelEncoder()

y = le.fit_transform(y)



# input features are stored in X

try:

    X = iris.drop('Species', axis=1)

except:

    pass



print (X.describe())

print ("")

y = pd.DataFrame(y)
import matplotlib.pyplot as plt



# Box Plot 

X.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(8, 8))

plt.show()
# histogram

X.hist(figsize=(8, 8))

plt.show()
from pandas.tools.plotting import scatter_matrix



# scatter pllot

scatter_matrix(X, figsize=(10, 10))

plt.show()
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler



seed = 7

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

y_train = pd.DataFrame.as_matrix(y_train).ravel()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, KFold



lr = LogisticRegression()

nb = GaussianNB()

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)

forest = RandomForestClassifier(criterion='entropy', n_estimators=10)

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

svm = SVC(kernel='linear', C=1.0)



models, results, names = [], [], []

models.append(('LR', lr))

models.append(('NB', nb))

models.append(('DTC', tree))

models.append(('RF', forest))

models.append(('KNN', knn))

models.append(('SVM', svm))



kfold = KFold(n_splits=10, random_state=seed)

m = -1

print ('%6s %12s %5s' %('Model', 'Accuracy', 'SD'))

print ('--------------------------------')

for name, model in models:

    score = cross_val_score(estimator=model, X=X_train_std, y=y_train, cv=kfold)

    results.append(score)

    names.append(name)

    print ('%5s %12.3f  %5.3f' %(name, np.mean(score),np.std(score)))

    

    if np.mean(score) > m:

        m = np.mean(score)

        best = name

    elif np.mean(score) == m:

        if np.std(score):

            best = name

            

print ('\nBEST CLASSIFIER =', best)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)



print ('Accuracy = %.3f' %(accuracy_score(y_test, y_pred)))

print ('\nConfusion Matrix\n', confusion_matrix(y_test, y_pred))

print ('\nClassification Report\n', classification_report(y_test, y_pred))