#Importando pacotes

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import sklearn



from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import Imputer , Normalizer , scale



from sklearn import tree



from sklearn.tree import DecisionTreeClassifier



from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split , KFold, StratifiedKFold, GridSearchCV, cross_val_score



%matplotlib inline 



print (sklearn.__version__)
prefixo_arquivos = '../input/serpro-wine/'

#Carga dos dados

wine = pd.read_csv(prefixo_arquivos + 'wine-train.csv')

wine_test = pd.read_csv(prefixo_arquivos + 'wine-test.csv')
wine_test.shape
wine.head()
wine['quality'] = wine['quality'].map({'good': 1, 'bad': 0})
wine['quality'].value_counts()
fig, ax = plt.subplots(1,11, figsize=(15,6))

fig.subplots_adjust(wspace=1.5)



sns.barplot(x = 'quality', y = 'fixed_acidity', data = wine,  ax=ax[0])

sns.barplot(x = 'quality', y = 'volatile_acidity', data = wine,  ax=ax[1])

sns.barplot(x = 'quality', y = 'citric_acid', data = wine,  ax=ax[2])

sns.barplot(x = 'quality', y = 'residual_sugar', data = wine,  ax=ax[3])

sns.barplot(x = 'quality', y = 'chlorides', data = wine,  ax=ax[4])

sns.barplot(x = 'quality', y = 'free_sulfur_dioxide', data = wine,  ax=ax[5])

sns.barplot(x = 'quality', y = 'total_sulfur_dioxide', data = wine,  ax=ax[6])

sns.barplot(x = 'quality', y = 'density', data = wine,  ax=ax[7])

sns.barplot(x = 'quality', y = 'ph', data = wine,  ax=ax[8])

sns.barplot(x = 'quality', y = 'sulphates', data = wine,  ax=ax[9])

sns.barplot(x = 'quality', y = 'alcohol', data = wine,  ax=ax[10])



#fig.show()

wine['quality'].value_counts()
# Retirando identificador do vinho e qualidade do conjunto

X = wine.drop(['quality','wine'], axis = 1)

y = wine['quality']

wine_test_red = wine_test.drop('wine', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Aplicando Standard scaling para otimizar algoritmos

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
X_train
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

report = classification_report(y_test, predicted)

print(report)
model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)

y_pred = model.predict( wine_test_red )

y_pred.shape
wine_id = wine_test.wine

submissiondf = pd.DataFrame( { 'wine': wine_id , 'quality': y_pred } )

submissiondf['quality'] = submissiondf['quality'].map({1: 'good', 0: 'bad'})



print(submissiondf.shape)

print(submissiondf.head())
submissiondf.to_csv( 'RG_wine_RF100.csv' , index = False )

#Random Forest 100 estimators