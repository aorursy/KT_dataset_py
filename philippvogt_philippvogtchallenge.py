# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importieren der benötigten Bibliotheken mit Bezeichnung des zugehörigen Kürzels
#Standardbibliotheken für Daten laden und mathematischen Formeln
import pandas as pd
import numpy as np

#Pakete für Vorhersage Modelle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Pakete zum Testen der Genauigkeit des Modells
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
#Graphische Pakete zum Visualisieren
import matplotlib.pyplot as plt
import seaborn as sns
#Visualizierung im Juypter Notebook
%matplotlib inline 

#Laden und sortierten der Datensätze
TrainData = pd.read_csv('../input/TrainData.csv', encoding = "ISO-8859-1", sep=";")
#error: 'utf-8' codec can't decode byte 0xdf in position 112: invalid continuation byte, deshalb encoding type Wechsel
TestData = pd.read_csv('../input/TestData.csv', encoding = "ISO-8859-1", sep=";")
#dient nur der Veranschaulichung des Datensatzes
#Wechsel der Zielvariable nein= 0, und 1= ja (auch Wechsel von String zum Integer)
print('Alte Typen der Zielvariable: {0} '.format(TrainData['Zielvariable'].unique()))
TrainData['Zielvariable']= TrainData['Zielvariable'].replace('nein', 0)
TrainData['Zielvariable']= TrainData['Zielvariable'].replace('ja', 1)
print('Neue Typen der Zielvariable: {0} '.format(TrainData['Zielvariable'].unique()))
TrainData.head()
print('Prozentsatz von positiven Zielvariablen: {}'.format(TrainData['Zielvariable'].astype(bool).sum(axis=0)/len(TrainData)))

print ('Laenge des Traindatensatzes: {0} '.format(len(TrainData)))
print('Laenge des Vorhersagedatensatzes: {0} '.format(len(TestData)))
print('Achtung der Datensatz tendiert dazu relativ schnell negative Ziel variablen vorherzusagen, wuerde man jetzt immer nein vorhersagen, wäre man in ca. 88% der Fälle richtig')
TrainData.groupby('Zielvariable').mean()
#Barplot
table=pd.crosstab(TrainData['Art der Anstellung'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig von der Art der Anstellung')
#X-Achse
plt.xlabel('Art der Anstellung')
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Art der Anstellung')
print('Die Art der Anstellung scheint ein guter Indikator zu sein')
#Barplot
table=pd.crosstab(TrainData['Familienstand'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Familienstand')
#X-Achse
plt.xlabel('Familenstand')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Familienstand')
print('Der Familienstand scheint kein guter Indikator für einen Abschluss zu sein')
#Barplot
table=pd.crosstab(TrainData['Schulabschluß'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Schulabschluß')
#X-Achse
plt.xlabel('Schulabschluß')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Schulabschluss')
print('Der Schulabschluss kann vielleicht ein guter Indikator für einen Abschluss zu sein')

#Barplot
table=pd.crosstab(TrainData['Geschlecht'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Geschlecht')
#X-Achse
plt.xlabel('Geschlecht')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Geschlecht')
print('Das Geschlecht ist kein guter Indikator für einen Abschluss')
#Barplot
table=pd.crosstab(TrainData['Monat'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Monat')
#X-Achse
plt.xlabel('Monat')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Monat')
print('Der Monat scheint ein guter Indikator für einen Abschluss zu sein')
TrainData.Alter.hist()
plt.title('Histogram des Alters')
plt.xlabel('Alters')
plt.ylabel('Häufigkeit')
plt.savefig('hist_alter')
print('Die meisten Kunden sind zwischen 30-40, dies gibt also voraussichtlich auch kein guten Indikator für einen Abschluss, da es sehr ungleich verteilt ist')
#Barplot
table=pd.crosstab(TrainData['Ausfall Kredit'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Ausfall Kredit')
#X-Achse
plt.xlabel('Kredit')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Ausfall Kredit')
print('Kein guter Indikator')
#Barplot
table=pd.crosstab(TrainData['Haus'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Ausfall Haus')
#X-Achse
plt.xlabel('Haus')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Haus')
print('Kein guter Indikator')
#Barplot
table=pd.crosstab(TrainData['Kredit'],TrainData.Zielvariable)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#Plot title
plt.title('Prozensatz einen Abschlusses abhängig vom Kredit')
#X-Achse
plt.xlabel('Kredit')
#y-Achse
plt.ylabel('Häufigkeit eines Abschlusses')
#Speicherung für Abschlussdokument
plt.savefig('Kredit')
print('Kein guter Indikator')
#Checkup welche Einträge nicht gefüllt sind:
count_nan = len(TrainData)-TrainData.count()
print(count_nan)
print('Löschen der Spalte: Tage seit letzter Kampange in Train und Test Data')
TrainData = TrainData.drop('Tage seit letzter Kampagne', 1)
TestData = TestData.drop('Tage seit letzter Kampagne', 1)

#Label encoding (Um die Algorithmen zu benutzen)
MasterData = TrainData.append(TestData)
#objecte kriegen ein Label
def convert(data):
    number = preprocessing.LabelEncoder()
    data['Monat'] = number.fit_transform(data.Monat)
    data['Geschlecht'] = number.fit_transform(data.Geschlecht)
    data['Art der Anstellung'] = number.fit_transform(data['Art der Anstellung'])
    data['Familienstand'] = number.fit_transform(data['Familienstand'])
    data['Schulabschluß'] = number.fit_transform(data['Schulabschluß'])
    data['Ausfall Kredit'] = number.fit_transform(data['Ausfall Kredit'])
    data['Haus'] = number.fit_transform(data['Haus'])
    data['Kredit'] = number.fit_transform(data['Kredit'])
    data['Kontaktart'] = number.fit_transform(data['Kontaktart'])
    data['Ergebnis letzte Kampagne'] = number.fit_transform(data['Ergebnis letzte Kampagne'])
    data=data.fillna(-999)
    return data
MasterDataencoded = convert(MasterData)
#Aufteilung wieder in Train und TestData (wichtig war, dass diese dieselben Label bekommen)
def split(df, headSize) :
    hd = df.head(headSize)
    tl = df.tail(len(df)-headSize)
    return hd, tl
TrainData, TestData = split(MasterDataencoded, len(TrainData))
#Der ursprüngliche Datensatz ist wieder hergestellt und wir haben eine TrainData und TestData

#In der Entwicklung wird nur der Traindatensatz verwendet, der Testdatensatz darf hier nicht benutzt werden

#Die Features werden im folgenden mit X bezeichnet, die Zielvariable mit y

X= TrainData[['Monat','Dauer', 'Alter','Geschlecht','Art der Anstellung','Familienstand','Schulabschluß'\
              ,'Ausfall Kredit','Kontostand','Haus','Kredit','Kontaktart','Anzahl der Ansprachen'\
              ,'Anzahl Kontakte letzte Kampagne','Ergebnis letzte Kampagne']]
y = TrainData[['Zielvariable']]
#Aufteilung des TrainData Sets in Trainingsdaten und Testdaten, wodurch wir erkennen können, welcher Algorithmus am besten funktioniert
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#77% liegen im Trainingdaten und 33% sind zum Testen

#DummyScore das Modell wird immer mit 'Nein' vorhergesagt in der Zielvariable
X_traindummy = X_train[['Dauer', 'Kontostand']]
X_testdummy = X_test[['Dauer']].values

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_traindummy, y_train.values.ravel())
y_majority_predicted = dummy_majority.predict(X_testdummy)
accuracyscore_dummy = dummy_majority.score(X_testdummy, y_test)

confusion_dummy = confusion_matrix(y_test, y_majority_predicted)
print('ConfusionMatrix für Dummy-Scorce:')
print(confusion_dummy)
print('R2 für Dummy-Scorce (immer Nein): {0} '.format(accuracyscore_dummy))
print('D.h. unser Vorhersagemodell wen wir immer Nein vorhersagen, würde in 9207 richtig liegen mit Nein und würde 1182 Ja fälscherlichweise als Nein vorhersagen')
print('Unser Vorhersagemodell sollte also besser sein als der Dummy')
#Festlegung der starken Indikatoren
Features = ['Monat', 'Dauer', 'Art der Anstellung', 'Kontostand']

#Logistische Regression


logistic = LogisticRegression(C=10).fit(X_train[Features], y_train.values.ravel())
logistic_predicted = logistic.predict(X_test[Features])
logistic_predicted2 = logistic.predict(X_train[Features])

confusion = confusion_matrix(y_test, logistic_predicted)
acc= accuracy_score(y_test, logistic_predicted)
acctrain = accuracy_score(y_train, logistic_predicted2)
print('Logistic Regression Results')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))


#DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5).fit(X_train[Features], y_train)
#Max_depth = 5 Schritte, da der Decision Tree sonst ein perfekte TrainScore schafft, was zu letztes der Vorhersagequalität geht
tree_predicted = tree.predict(X_test[Features])
tree_predicted2 = tree.predict(X_train[Features])

confusion = confusion_matrix(y_test, tree_predicted)
acc= accuracy_score(y_test, tree_predicted)
acctrain = accuracy_score(y_train, tree_predicted2)
print('Decision Tree Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))


clf = RandomForestClassifier(max_depth=11).fit(X_train[Features], y_train.values.ravel())
#Hier ist 11 beser als 15
clf_predicted = clf.predict(X_test[Features])
clf_predicted2 = clf.predict(X_train[Features])
confusion = confusion_matrix(y_test, clf_predicted)
acc= accuracy_score(y_test, clf_predicted)
acctrain = accuracy_score(y_train, clf_predicted2)
print('Random Forest Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))

Features = ['Monat', 'Dauer', 'Art der Anstellung', 'Kontostand', 'Familienstand', 'Schulabschluß', 'Ausfall Kredit', 'Haus', 'Kredit']
#Logistische Regression


logistic = LogisticRegression(C=10).fit(X_train[Features], y_train.values.ravel())
logistic_predicted = logistic.predict(X_test[Features])
logistic_predicted2 = logistic.predict(X_train[Features])

confusion = confusion_matrix(y_test, logistic_predicted)
acc= accuracy_score(y_test, logistic_predicted)
acctrain = accuracy_score(y_train, logistic_predicted2)
print('Logistic Regression Results')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))


#DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5).fit(X_train[Features], y_train)
#Max_depth = 5 Schritte, da der Decision Tree sonst ein perfekte TrainScore schafft, was zu letztes der Vorhersagequalität geht
tree_predicted = tree.predict(X_test[Features])
tree_predicted2 = tree.predict(X_train[Features])

confusion = confusion_matrix(y_test, tree_predicted)
acc= accuracy_score(y_test, tree_predicted)
acctrain = accuracy_score(y_train, tree_predicted2)
print('Decision Tree Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))

clf = RandomForestClassifier(max_depth=11).fit(X_train[Features], y_train.values.ravel())
#Hier ist 11 beser als 15
clf_predicted = clf.predict(X_test[Features])
clf_predicted2 = clf.predict(X_train[Features])
confusion = confusion_matrix(y_test, clf_predicted)
acc= accuracy_score(y_test, clf_predicted)
acctrain = accuracy_score(y_train, clf_predicted2)
print('Random Forest Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))

print('Diese Auswahl lohnt sich nicht wirklich ')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
anova_filter = SelectKBest(f_regression, k=5)
anova_clf = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_clf.set_params(anova__k=10).fit(X, y.values.ravel())
prediction = anova_clf.predict(X)
anova_clf.score(X, y) 
anova_clf.named_steps['anova'].get_support()
#select the features based on the feature boost
Features = ['Dauer', 'Art der Anstellung', 'Schulabschluß', 'Kontostand', 'Haus', 'Kredit', 'Kontaktart', 'Anzahl der Ansprachen', 'Anzahl Kontakte letzte Kampagne', 'Ergebnis letzte Kampagne']
#Logistische Regression


logistic = LogisticRegression(C=10).fit(X_train[Features], y_train.values.ravel())
logistic_predicted = logistic.predict(X_test[Features])
logistic_predicted2 = logistic.predict(X_train[Features])

confusion = confusion_matrix(y_test, logistic_predicted)
acc= accuracy_score(y_test, logistic_predicted)
acctrain = accuracy_score(y_train, logistic_predicted2)
print('Logistic Regression Results')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))


#DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=10).fit(X_train[Features], y_train)
#Max_depth = 5 Schritte, da der Decision Tree sonst ein perfekte TrainScore schafft, was zu letztes der Vorhersagequalität geht
tree_predicted = tree.predict(X_test[Features])
tree_predicted2 = tree.predict(X_train[Features])

confusion = confusion_matrix(y_test, tree_predicted)
acc= accuracy_score(y_test, tree_predicted)
acctrain = accuracy_score(y_train, tree_predicted2)
print('Decision Tree Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))

clf = RandomForestClassifier(max_depth=10).fit(X_train[Features], y_train.values.ravel())
#Hier ist 10 beser als 15
clf_predicted = clf.predict(X_test[Features])
clf_predicted2 = clf.predict(X_train[Features])
confusion = confusion_matrix(y_test, clf_predicted)
acc= accuracy_score(y_test, clf_predicted)
acctrain = accuracy_score(y_train, clf_predicted2)
print('Random Forest Results:')
print('Confusion Matrix für TrainData : ')
print(confusion)
print('R2 für TrainData : {0} '.format(acctrain))      
print('R2 für TestData: {0}'.format(acc))

#Das beste Vorhersagemodell ist der Decisiontree da hier sowohl der R2 mit am besten ist, als auch die meisten positiven Vorhersagen triff
#Wenn das Ziel ist es möglichst alle Abschlusswahrscheinlichkeiten vorherzusagen ist der Decisiontree auch die beste Wahl
#Die AUC Score ist hier am höchsten

PredictionData = TestData
predictions = clf.predict_proba(TestData[Features])[:, 1]
PredictionData ['Abschlusswahrscheinlichkeit'] = np.float32 (predictions)
PredictionData2 = PredictionData[['Stammnummer', 'Abschlusswahrscheinlichkeit']]
PredictionData2.rename(columns={'Stammnummer': 'ID', 'Abschlusswahrscheinlichkeit': 'Expected'}, inplace=True)
#Die Abschlusswahrscheinlichkeit wird pro Stammnummer angegeben und kann nun für weitere Analysen benutzt werden
#Speicherung in CSV-Format für weitere Kommunikation, sortiert nach höchster Wahrscheinlichkeit
PredictionData2.sort_values(by=['Abschlusswahrscheinlichkeit'], ascending=False).to_csv('Abschlusswahrscheinlichkeiten.csv')
PredictionData2