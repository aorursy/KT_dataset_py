import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
colnames=['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315', 'Proline']
dataset = pd.read_csv("../input/Wine.csv",names=colnames, header=None)
dataset.head()
dataset.shape
dataset['class'].value_counts()
# Univariate graphs to see the distribution
dataset.hist(figsize=(20, 15))
plt.show()
# Correlation Matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(dataset.drop('class', axis=1).corr(), annot=True)
X = dataset.iloc[:,1:]
y = dataset.iloc[:,0]

# Feature Scaling
from  sklearn.preprocessing  import StandardScaler
slc= StandardScaler()
X = slc.fit_transform(X)

# Spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Test options and evaluation metric
num_folds = 10
seed = 0
scoring = 'accuracy'
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Spot-Check Algorithms (Classification)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Spot-Check Ensemble Models (Classification)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB',XGBClassifier()))

# evaluate each model in turn
results = {}
accuracy = {}
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results[name] = (cv_results.mean(), cv_results.std())
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, _)
results
accuracy
# Finalizing the model and comparing the test, predict results

model = SVC(random_state=seed)

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))

cm= confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)