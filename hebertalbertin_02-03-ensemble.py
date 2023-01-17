# 203142 André Felício de Sousa 
# 203143 Hebert Francisco Albertin 
# 203011 Lucas Francisco de Camargo 
# 203214 Marcelo Nogueira da Silva 
# 203144 Murilo Spinoza de Arruda 
# 191515 Rodrigo Lopes
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
end = time.time()
print(end - start)
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, whiten=True)

X_pca = pca.fit_transform(X)

print('Número original de atributos:', X.shape[1])
print('Número reduzido de atributos:', X_pca.shape[1])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

start = time.time()
print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))
end = time.time()
print('Tempo', end - start)
#######
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

start = time.time()
print('Acurácia nos dados reduzidos (PCA em tudo):', accuracy_score(y_test, y_pred))
end = time.time()
print('Tempo', end - start)
#######
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=0.95, whiten=True)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

start = time.time()
print('Acurácia nos dados originais (PCA da parte certa):', accuracy_score(y_test, y_pred))
end = time.time()
print('Tempo', end - start)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

dtc = DecisionTreeClassifier(criterion="entropy")
lr = LogisticRegression();
bnb = BernoulliNB()
gnb = GaussianNB()

base_methods=[lr, bnb, gnb, dtc]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


for bm  in base_methods:
    start = time.time()
    
    print("Método: ", bm)
    bag_model=BaggingClassifier(base_estimator=bm,n_estimators=100,bootstrap=True)
    bag_model=bag_model.fit(X_train,y_train)
    
    ytest_pred=bag_model.predict(X_test)
    print(bag_model.score(X_test, y_test))
    print(confusion_matrix(y_test, ytest_pred))
    
    end = time.time()
    print('Tempo', end - start)
    print('\n\n\n')
from sklearn.ensemble import GradientBoostingClassifier

start = time.time()

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2, random_state=0)
model.fit(X_train, y_train)

scores = cross_val_score(model, X_test, y_test, cv=5)

print('Acurácia de Gradient Boosting Tree Nos Dados Originais:', scores.mean())

end = time.time()
print('Tempo', end - start)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

for bm  in base_methods:
    start = time.time()
    
    print("Method: ", bm)
    bag_model=BaggingClassifier(base_estimator=bm,n_estimators=100,bootstrap=True)
    bag_model=bag_model.fit(X_train,y_train)
    ytest_pred=bag_model.predict(X_test)
    print(bag_model.score(X_test, y_test))
    print(confusion_matrix(y_test, ytest_pred))
    
    end = time.time()
    print('Tempo', end - start)
    print('\n\n\n')
start = time.time()

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2, random_state=0)
model.fit(X_train, y_train)

scores = cross_val_score(model, X_test, y_test, cv=5)

print('Acurácia de Gradient Boosting Tree Nos Dados Reduzidos:', scores.mean())

end = time.time()
print('Tempo', end - start)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=0.95, whiten=True)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

for bm  in base_methods:
    start = time.time()
    
    print("Method: ", bm)
    bag_model=BaggingClassifier(base_estimator=bm,n_estimators=100,bootstrap=True)
    bag_model=bag_model.fit(X_train,y_train)
    ytest_pred=bag_model.predict(X_test)
    print(bag_model.score(X_test, y_test))
    print(confusion_matrix(y_test, ytest_pred))
    
    end = time.time()
    print('Tempo', end - start)
    print('\n\n\n')
start = time.time()

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2, random_state=0)
model.fit(X_train, y_train)

scores = cross_val_score(model, X_test, y_test, cv=5)

print('Acurácia de Gradient Boosting Tree nos Dados Originais com PCA (parte certa):', scores.mean())

end = time.time()
print('Tempo', end - start)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

k_hist = []
acc = []

for k in range(1,30):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    fvalue_selector = SelectKBest(f_classif, k=k)
    X_kbest = fvalue_selector.fit_transform(X_train, y_train)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
    model.fit(X_kbest, y_train)
    X_test_kbest = fvalue_selector.transform(X_test)
    y_pred = model.predict(X_test_kbest)
    
    k_hist.append(k)
    acc.append(accuracy_score(y_test, y_pred))
import seaborn as sns;
ax = sns.lineplot(x=np.array(k_hist), y=np.array(acc))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Melhor K encontrado foi de valor 29

fvalue_selector = SelectKBest(f_classif, k=29)

X_kbest = fvalue_selector.fit_transform(X_train, y_train)

for bm  in base_methods:
    start = time.time()
    
    print("Method: ", bm)
    bag_model=BaggingClassifier(base_estimator=bm,n_estimators=100,bootstrap=True)
    bag_model=bag_model.fit(X_kbest,y_train)
    
    X_test_kbest = fvalue_selector.transform(X_test)
    y_pred = model.predict(X_test_kbest)
    
    ytest_pred=bag_model.predict(X_test_kbest)
    print(bag_model.score(X_test_kbest, y_test))
    print(confusion_matrix(y_test, ytest_pred))
    
    end = time.time()
    print('Tempo', end - start)
    print('\n\n\n')
start = time.time()

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2, random_state=0)
model.fit(X_kbest, y_train)

X_test_kbest = fvalue_selector.transform(X_test)

scores = cross_val_score(model, X_test_kbest, y_test, cv=5)

print('Acurácia de Gradient Boosting Tree em KBest 29:', scores.mean())

end = time.time()
print('Tempo', end - start)
