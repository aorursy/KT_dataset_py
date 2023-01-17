import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette="muted")
%matplotlib inline

import sklearn
from sklearn import datasets
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
boston = datasets.load_boston()
print(boston.DESCR)
plt.hist(boston.target,bins=50)
plt.xlabel('Preço em $1000s')
plt.ylabel('Num Casas');
plt.scatter(boston.data[:,5],boston.target)
plt.ylabel('Preço em $1000s')
plt.xlabel('Num Quartos');
boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target
boston_df.head()
sns.lmplot('RM','Price',data = boston_df);
X = boston_df.drop('Price', 1)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_df.Price, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
lreg = LinearRegression()
lreg.fit(X_train,Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

print("MSE com Y_train: %.2f" % sklearn.metrics.mean_squared_error(Y_train, pred_train))

print("MSE com X_test e Y_test: %.2f" % sklearn.metrics.mean_squared_error(Y_test, pred_test))
from sklearn.metrics import r2_score

print("R2 score no conjunto de testes: %.2f" % r2_score(Y_test, pred_test))

print("R2 score no conjunto de treinamento: %.2f" % r2_score(Y_train, pred_train))
diabetes = sklearn.datasets.load_diabetes()
print(diabetes.DESCR)
# Seu código aqui
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette="muted")
%matplotlib inline

import sklearn
from sklearn import datasets
import sklearn.model_selection
iris_data = sklearn.datasets.load_iris()
print(iris_data.DESCR)
X = iris_data.data
y = iris_data.target

iris = pd.DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
iris['Species'] = y
iris.Species.astype(int, inplace=True)
iris.sample(5)
def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

iris['Species'] = iris['Species'].apply(flower)
iris.sample(5)
sns.pairplot(iris,hue='Species',size=2);
sns.factorplot('Petal Length',data=iris,hue='Species', kind='count', size=8);
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3, n_jobs=4),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1,),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=4),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=4),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
results = []
for name, clf in zip(names, classifiers):
    print(name)
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    except:
        print("Could not train %s model" % name)
        continue

    results.append({
        'name': name,
        'precision': precision_score(y_test, y_pred, average="macro"),
        'accuracy': accuracy_score(y_test, y_pred),
    })
results = pd.DataFrame(results)
results = results.set_index('name')
results
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')

print('train dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )
train_dataset.head()
