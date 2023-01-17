from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=5)

print('Acurácia de KNeighbors simples:', scores.mean())

model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, random_state = 42)
scores = cross_val_score(model, X, y, cv=5)

print('Acurácia de KNeighbors Bagging (c/ 10 estimators):', scores.mean())

model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_estimators=100, random_state = 42)
scores = cross_val_score(model, X, y, cv=5)

print('Acurácia de KNeighbors Bagging (c/ 100 estimators):', scores.mean())
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, X, y, cv=5)
print('Acurácia de Decision Tree puro:', scores.mean())

model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, X, y, cv=5)
print('Acurácia de Random Forest:', scores.mean())

model = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, X, y, cv=5)
print('Acurácia de Extreme Randomized Trees:', scores.mean())

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings(action='ignore')

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=10, random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb',clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5)
    print("Acurácia: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print('-'*20)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='hard')

for clf, label in zip([clf1, clf2, eclf], ['Logistic Regression', 'Random Forest', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5)
    print("Acurácia: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2, random_state=0)
scores = cross_val_score(model, X, y, cv=5)

print('Acurácia de Gradient Boosting Tree:', scores.mean())
import time

start = time.time()
# Treinar modelo
end = time.time()
print(end - start)