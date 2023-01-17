import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict #used to break dataset on smaller portions

from sklearn.svm import SVC #SVM classifier

from sklearn import tree #tree algorithm

from sklearn.linear_model import LogisticRegression #LogisticRegression Algorithm

from sklearn.metrics import mean_absolute_error #MAE

from sklearn.metrics import mean_squared_error #MSE

from sklearn import metrics #benchmark between models

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier #Random forest classifier

from sklearn.tree import DecisionTreeClassifier #Decision tree classifier

from sklearn.neighbors import KNeighborsClassifier
got_dataset = pd.read_csv('../input/game-of-thrones/character-predictions.csv')
pd.set_option('display.max_columns', None)

got_dataset.head()
got_dataset.info()
nans = got_dataset.isna().sum()

nans[nans > 0]
got_dataset.describe()
print(got_dataset["age"].mean()) #getting mean of age column
print(got_dataset["name"][got_dataset["age"] < 0])

print(got_dataset["age"][got_dataset["age"] < 0])
got_dataset.loc[1684, "age"] = 25.0

got_dataset.loc[1868, "age"] = 0.0
print(got_dataset["age"].mean())
got_dataset["age"].fillna(got_dataset["age"].mean(), inplace = True)

got_dataset["culture"].fillna('', inplace = True)



got_dataset.fillna(value = -1, inplace=True)
got_dataset.boxplot(["alive", "popularity"])
got_dataset.info()
#removing some columns

drop = ["S.No", "pred", "alive", "plod", "name", "isAlive", "DateoFdeath"]

got_dataset.drop(drop, inplace=True, axis = 1)



got_dataset_2 = got_dataset.copy(deep = True)
#changing category values to one-hot-encoder

got_dataset = pd.get_dummies(got_dataset)
got_dataset.head()
got_dataset.shape
#in/out

x = got_dataset.iloc[:, 1:].values

y = got_dataset.iloc[:, 0].values
#Cross valid, breaking dataset in 5 groups

kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
print(kfold.get_n_splits())
#building classifier models

modelos = [LogisticRegression(solver = 'liblinear'), RandomForestClassifier(n_estimators = 400, random_state = 42), 

           DecisionTreeClassifier(random_state = 42), SVC(kernel = 'linear', gamma = 'scale', random_state = 42), KNeighborsClassifier()]
#using cross valid

mean = []

std = []

for model in modelos:

    result = cross_val_score(model, x, y, cv = kfold, scoring = "accuracy", n_jobs = -1)

    mean.append(result)

    std.append(result)
classificadores = ["Regressão Logistica", "Random Forest", "Arvore de Decisão", "SVM", "KNN"]



plt.figure(figsize = (20, 10))

for i in range(len(mean)):

    sns.distplot(mean[i], hist = False, kde_kws = {"shade": True})

    

plt.title("Distribuicao de cada um dos classificadores", fontsize=15)

plt.legend(classificadores)

plt.xlabel("Acuracia", labelpad = 20)

plt.yticks([])
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.2, stratify = y, shuffle = True, random_state = 42)
#using SVM and random forest

svm_clf = SVC(C=0.9, gamma = 0.1, kernel = 'linear', probability = True, random_state = 42)

rf_clf = RandomForestClassifier(n_estimators = 400, n_jobs = -1, random_state = 42)



#training models

svm_clf.fit(x_train, y_train)

rf_clf.fit(x_train, y_train)
svm_prob = svm_clf.predict_proba(x_test)

rf_prob = rf_clf.predict_proba(x_test)



svm_preds = np.argmax(svm_prob, axis = 1)

rf_preds = np.argmax(rf_prob, axis = 1)
cm = metrics.confusion_matrix(y_test, svm_preds)

cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

cm2 = metrics.confusion_matrix(y_test, rf_preds)

cm2 = cm2.astype('float') / cm2.sum(axis = 1)[:, np.newaxis]



classes = ["Morto", "Vivo"]

f, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].set_title("SVM", fontsize=15.)

sns.heatmap(pd.DataFrame(cm, index = classes, columns = classes), cmap = 'winter', annot = True, fmt = '.2f', ax=ax[0]).set(xlabel = "Previsao", ylabel = "Valor Real")



ax[1].set_title("Random Forest", fontsize = 15.)

sns.heatmap(pd.DataFrame(cm2, index = classes, columns = classes), cmap = 'winter', annot = True, fmt = '.2f', ax=ax[1]).set(xlabel = "Previsao", ylabel = "Valor Real")