# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Imports

import matplotlib.pyplot as plt

import time



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc





from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
###Auslesen des Datasets in Dataframe df_origin

df_origin = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
### Analyse des originären Dataframes

#Grafische Darstellung des Dataframes als Tabelle

#df_origin.head()



#Ausgabe von Länge und Breite

#df_origin.shape



#Ausgabe der Datentypen nach Attributen

#df_origin.dtypes



#Summierung von fehlenden Values nach Attributen

#print(df_origin.isnull().sum().sort_values(ascending=False))



#Bestimmung des Wertverhältnisses NaN zu Gesamtanzahl Werte

#print((df_origin.isnull().sum() / df_origin.shape[0]).sort_values(ascending=False))
### Analyse des originären Dataframes auf potenziell verwertbare Korrelationen

#plt.subplots(figsize=(20,20))

#corrHeatmap = sns.heatmap(df_origin.corr(method='pearson'), annot=True, square=True)    
### Datenbereinigung

# Bereinigung des originären Dataframes um nicht-verwertbare Attribute mit zu vielen n/a-Values

df_removedColumns = df_origin.drop(df_origin.columns[df_origin.apply(lambda col: col.isnull().sum() > 300000)], axis = 1)
# Entfernen aller Attribute ohne plausible Hypothese, mit Obsoleszenz oder mit komplexer Wertausprägung

df_numericalValuesOnly = df_removedColumns.drop(['ID',#ausschließlich unique, daher keine Korrelation gegeben

                                                 'Source',#keine plausible Hypothese

                                                 'Description',#komplexe Ausprägung erfordert Preprocessing mit Textanalyse

                                                 'Street',#obsolet

                                                 'Side',#obsolet

                                                 'City',#obsolet

                                                 'County',#obsolet

                                                 'State',#obsolet

                                                 'Zipcode',#obsolet

                                                 'Country',#obsolet

                                                 'Timezone',#obsolet

                                                 'Airport_Code',#keine plausible Hypothese

                                                 'Weather_Timestamp',#keine plausible Hypothese

                                                 'Wind_Direction',#keine plausible Hypothese

                                                 'Weather_Condition',#komplexe Ausprägung erfordert Preprocessing mit Textanalyse

                                                 'Civil_Twilight',#obsolet

                                                 'Nautical_Twilight',#obsolet

                                                 'Astronomical_Twilight'#obsolet

                                                ], axis=1)
### Datenkonvertierung

# Konvertierung von Boolean-Werten zu Integer-Werten

for col in df_numericalValuesOnly:

    if df_numericalValuesOnly[col].dtype==np.bool:

        df_numericalValuesOnly[col] = df_numericalValuesOnly[col].astype(int)
# Berechnung der Dauer eines Unfalls

df_numericalValuesOnly['Start_Time'] = pd.to_datetime(df_numericalValuesOnly['Start_Time'], errors='coerce')

df_numericalValuesOnly['Start'] = df_numericalValuesOnly['Start_Time'].dt.hour

df_numericalValuesOnly = df_numericalValuesOnly.drop(['Start_Time'], axis=1)

df_numericalValuesOnly['End_Time'] = pd.to_datetime(df_numericalValuesOnly['End_Time'], errors='coerce')

df_numericalValuesOnly['End'] = df_numericalValuesOnly['End_Time'].dt.hour

df_numericalValuesOnly = df_numericalValuesOnly.drop(['End_Time'], axis=1)

df_numericalValuesOnly['Duration'] = (df_numericalValuesOnly['End']-df_numericalValuesOnly['Start'])

df_numericalValuesOnly = df_numericalValuesOnly.drop(['Start','End'], axis=1)
# Hot Encoding des Attributs "Sunrise_Sunset"

df_Sunrise_SunsetEncoding = pd.get_dummies(df_numericalValuesOnly.Sunrise_Sunset)

df_numericalValuesOnly = pd.concat([df_numericalValuesOnly, df_Sunrise_SunsetEncoding], axis=1)

df_numericalValuesOnly = df_numericalValuesOnly.drop(['Sunrise_Sunset'], axis=1)
### Analyse von Korrelationen auf die Zielvariable "Severity"

# Erzeugung eines Dataframes mit allen Korrelationskoeefizienten der Attribute auf die Zielvariable

df_featureTargetCorrelations = df_numericalValuesOnly.corrwith(df_numericalValuesOnly['Severity'])



#Darstellung der Korrelationskoeffizienten auf die Zielvariable in einer Heatmap

#plt.subplots(figsize=(20,20))

#corrHeatmap = sns.heatmap(df_featureTargetCorrelations.corr(method='pearson'), annot=True, square=True)
# Entfernen aller Attribute, deren Korrelationskoeffizient auf die Zielvariable kleiner als +/- 0,045 ist

df_importantCorrelationsOnly = df_numericalValuesOnly.drop(['Temperature(F)',

                                                            'Humidity(%)',

                                                            'Pressure(in)',

                                                            'Visibility(mi)',

                                                            'Bump',

                                                            'Give_Way',

                                                            'No_Exit',

                                                            'Railway',

                                                            'Roundabout',

                                                            'Traffic_Calming',

                                                            'Turning_Loop',

                                                           ], axis=1)
# Überschreiben des Dataframes mit ausschließlich signifikanten Korrelationskoeffizienten auf die Zielvariable

df_featureTargetCorrelations = df_importantCorrelationsOnly.corrwith(df_importantCorrelationsOnly['Severity']).drop(['Severity'])
#Darstellung der Korrelationskoeffizienten auf die Zielvariable in einer Heatmap

#plt.subplots(figsize=(20,20))

#corrHeatmap = sns.heatmap(df_featureTargetCorrelations.corr(method='pearson'), annot=True, square=True)
# Preprocessing und Staging des Dataframes als Input

df_training = df_importantCorrelationsOnly.drop([#'Severity',

                                                  'Start_Lat',

                                                  #'Start_Lng',

                                                  #'Distance(mi)',

                                                  'Amenity',

                                                  #'Crossing',

                                                  #'Junction',

                                                  #'Station',

                                                  #'Stop',

                                                  #'Traffic_Signal',

                                                  'Duration'

                                                  #'Day',

                                                  #'Night',

                                                 ], axis=1)
y = df_training.Severity
X = df_training.drop(['Severity'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(predictions, y_test))
### RandomizedSeachCV auf DecisionTreeClassifier



### aufgrund extrem langer Laufzeit auskommentiert



#param_dist = {"max_depth": [65, 80, 95, None],

              #"max_features": [5, 9, 12],

              #"min_samples_leaf": [1, 2, 4],

              #"min_samples_split": [8, 10, 12],

              #"criterion": ["gini", "entropy"]

             #}



#rscv = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), param_distributions=param_dist, scoring='accuracy', cv=5, verbose=0, n_jobs=-1)

#rscv.fit(X_train, y_train)
#rscv.best_params_
### GridSearchCV auf DecisionTreeClassifier



### aufgrund extrem langer Laufzeit auskommentiert



#param_grid = {"max_depth": [65, 80, 95, None],

              #"max_features": [7, 9, 11],

              #"min_samples_leaf": [1, 2, 4],

              #"min_samples_split": [8, 10, 12],

              #"criterion": ["gini", "entropy"]

             #}



#gscv = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid=param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1)

#gscv.fit(X_train, y_train)
#gscv.best_params_
### Optimierter Decision Tree

clf = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=9, min_samples_leaf=2, min_samples_split=10)



start = time.time()

clf.fit(X_train, y_train)

end = time.time()

duration = end-start

print("Trainingsdauer:", duration, "s")
predictions = clf.predict(X_test)

prob = clf.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

print("Macro ROC_AUC Score:", roc_auc_score(y_test, prob, average='macro', multi_class='ovr'))
### Kreuzvalidierung



kf = KFold(n_splits=5, shuffle=True)



results = cross_val_score(clf, X, y, scoring='accuracy', cv=kf)

results


# Binarize the output

y_binarized = label_binarize(y, classes=[1, 2, 3, 4])

n_classes = 4



X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=0)



classifier = OneVsRestClassifier(clf)

start = time.time()

classifier.fit(X_train, y_train)

end = time.time()

duration = end-start

print("Trainingsdauer:", duration, "s")



y_score = classifier.predict_proba(X_test)
print("Micro-ROC-AUC-Score:", roc_auc_score(y_test, y_score, average='micro'))

print("Macro-ROC-AUC-Score:", roc_auc_score(y_test, y_score, average='macro'))
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

#colors = cycle(['blue', 'red', 'green', 'yellow'])

#for i, color in zip(range(n_classes), colors):

plt.figure(figsize=(10,10))

for i in range(n_classes):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-AUC-Metrics for DecisionTree on Dataset "US-Accidents"')

plt.legend(loc="lower right")

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(random_state = 0, n_jobs = -1)



clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(predictions, y_test))
### RandomizedSearchCV auf RandomForest



### aufgrund extrem langer Laufzeit auskommentiert



#from sklearn.model_selection import RandomizedSearchCV



#clf = RandomForestClassifier(random_state = 42, n_estimators = 20, max_features = 'sqrt', max_depth=60, criterion='entropy', min_samples_leaf=2, min_samples_split=10, verbose = 1, n_jobs = -1)



#param_dist = {

    #'n_estimators': [100, 500, 1000],

    #'max_depth': [40, 60, 100, None],

    #'criterion': ['gini', 'entropy'],

    #'max_features': ['auto', 'sqrt'],

    #'min_samples_split': [5, 10, 20, 40],

    #'min_samples_leaf': [2, 4, 8, 16]

#}



#rscv = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring='accuracy', cv=3, n_iter=10, verbose=1)

#rscv.fit(X_train, y_train)
#rscv.best_params_
### GridSearchCV auf RandomForest



### aufgrund extrem langer Laufzeit auskommentiert



#from sklearn.model_selection import GridSearchCV



#clf = RandomForestClassifier(random_state = 42, n_estimators = 20, max_features = 'sqrt', max_depth=60, criterion='entropy', min_samples_leaf=2, min_samples_split=10, verbose = 1, n_jobs = -1)



#param_grid = {

    #'n_estimators': [50, 100, 200, 500],

    #'max_features': ['auto', 'sqrt'],

    #'max_depth': [50, 60, 80, 100]

    #'min_samples_split':

    #'min_samples_leaf':

#}



#gscv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

#gscv.fit(X_train, y_train)

#gscv_clf.best_params_
### Optimierter Random Forest

clf = RandomForestClassifier(random_state=0, n_estimators=50, max_features='auto', max_depth=100, criterion='entropy', min_samples_leaf=2, min_samples_split=10, verbose=0, n_jobs=-1)



start = time.time()

clf.fit(X_train, y_train)

end = time.time()

duration = end-start

print("Trainingsdauer:", duration, "s")
predictions = clf.predict(X_test)

prob = clf.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

print("Macro ROC_AUC Score:", roc_auc_score(y_test, prob, average='macro', multi_class='ovr'))
### Kreuzvalidierung



clf = RandomForestClassifier(random_state=0, n_estimators=50, max_features='auto', max_depth=100, criterion='entropy', min_samples_leaf=2, min_samples_split=10, verbose=0, n_jobs=1) #Anpassung der n_jobs=1, damit keine Parallelisierung, gab ansonsten Probleme



kf = KFold(n_splits=5, shuffle=True)



results = cross_val_score(clf, X, y, scoring='accuracy', cv=kf)

results
###Konfusionsmatrix



#from sklearn.metrics import confusion_matrix



#confusion_matrix(y_test, predictions)


# Binarize the output

y_binarized = label_binarize(y, classes=[1, 2, 3, 4])

n_classes = 4



X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=0)



clf = RandomForestClassifier(random_state=0, n_estimators=50, max_features='auto', max_depth=100, criterion='entropy', min_samples_leaf=2, min_samples_split=10, verbose=0, n_jobs=-1)



classifier = OneVsRestClassifier(clf)

start = time.time()

classifier.fit(X_train, y_train)

end = time.time()

duration = end-start

print("Trainingsdauer:", duration, "s")



y_score = classifier.predict_proba(X_test)
print("Micro-ROC-AUC-Score:", roc_auc_score(y_test, y_score, average='micro'))

print("Macro-ROC-AUC-Score:", roc_auc_score(y_test, y_score, average='macro'))
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,10))

for i in range(n_classes):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-AUC-Metrics for RandomForest on Dataset "US-Accidents"')

plt.legend(loc="lower right")

plt.show()