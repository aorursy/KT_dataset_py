import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/data-train-test/TrainData.csv', encoding='latin-1', sep=";")
df.head()
# pd.get_dummies(df, drop_first=True).head()
# drop der ersten Dummy Var nicht sinnvoll
# pd.get_dummies(df).head()

# nicht-nummerische Inhalte in nummerische (binäre) Informationen umwandeln
df = pd.get_dummies(df)
# Spalten von pd.get_dummies(df) anzeigen
df.describe()

# redundante Infos droppen
df = df.drop(['Zielvariable_nein', 'Geschlecht_w', 'Art der Anstellung_Unbekannt', 'Familienstand_geschieden', 'Schulabschluß_Unbekannt', 'Ausfall Kredit_nein', 'Haus_nein', 'Kredit_nein', 'Kontaktart_Unbekannt', 'Ergebnis letzte Kampagne_Unbekannt'
], axis=1)
# NaN-Werte in Spalte 'Tage seit letzter Kampagne' sind beim Plotten ein Problem -> durch mean (Mittelwert) ersetzten
# sns.heatmap(df.isnull(), yticklabels=False, cbar=Fals, cmap='viridis')
df = df.groupby(df.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))

# Aufbau des dataframes prüfen
#df.info()
# Statistische Infos zu dem dataframe
# df.describe()
# Plotten mit pairplot macht keinen Sinn - zu viele Spalten und Daten
# sns.pairplot(df)
# sns.pairplot(df.sample(1000))
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Zielvariable_ja'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Tag'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Dauer'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Alter'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Kontostand'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Anzahl der Ansprachen'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Tage seit letzter Kampagne'])
# Verteilung der angegebenen Spalte zeichnen
# sns.distplot(df['Anzahl Kontakte letzte Kampagne'])
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df2 = df.loc[:, ['Dauer', 'Alter', 'Kontostand', 'Anzahl der Ansprachen', 'Tage seit letzter Kampagne', 'Anzahl Kontakte letzte Kampagne', 'Geschlecht_m', 'Kontaktart_Festnetz', 'Kontaktart_Handy', 'Ausfall Kredit_ja', 'Zielvariable_ja']]
# df2.corr()
# sns.heatmap(df2.corr())
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df3 = df.loc[:, ['Art der Anstellung_Arbeiter', 'Art der Anstellung_Arbeitslos', 'Art der Anstellung_Dienstleistung', 'Art der Anstellung_Gründer', 'Art der Anstellung_Hausfrau', 'Art der Anstellung_Management', 'Art der Anstellung_Rentner', 'Art der Anstellung_Selbständig', 'Art der Anstellung_Student',
#'Art der Anstellung_Technischer Beruf', 'Art der Anstellung_Verwaltung', 'Zielvariable_ja']]
# df3.corr()
# sns.heatmap(df3.corr())
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df4 = df.loc[:, ['Familienstand_single', 'Familienstand_verheiratet', 'Schulabschluß_Abitur', 'Schulabschluß_Real-/Hauptschule', 'Schulabschluß_Studium', 'Zielvariable_ja']]
# df4.corr()
# sns.heatmap(df4.corr(), annot=True)
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df5 = df.loc[:, ['Ausfall Kredit_ja', 'Haus_ja', 'Kredit_ja', 'Zielvariable_ja']]
# df5.corr()
# sns.heatmap(df5.corr(), annot=True)
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df6 = df.loc[:, ['Ergebnis letzte Kampagne_Erfolg', 'Ergebnis letzte Kampagne_Kein Erfolg', 'Ergebnis letzte Kampagne_Sonstiges', 'Zielvariable_ja']]
# df6.corr()
# sns.heatmap(df6.corr(), annot=True)
# Korrelationsmatrix eingrenzen auf angegebenen Features
#df7 = df.loc[:, ['Monat_apr', 'Monat_aug', 'Monat_dec', 'Monat_feb', 'Monat_jan', 'Monat_jul', 'Monat_jun', 'Monat_mar', 'Monat_may', 'Monat_nov', 'Monat_oct', 'Monat_sep', 'Zielvariable_ja']]
# df7.corr()
# sns.heatmap(df7.corr())
# Überblick der Anzahl Kunden, die das Produkt in der Kampagne abgeschlossen haben
# sns.countplot(x='Zielvariable_ja', hue='Geschlecht_m', data=df)
# Überblick Geschlecht
# sns.countplot(x='Geschlecht_m', data=df)
# Überblick Ausfall Kredit
# sns.countplot(x='Ausfall Kredit_ja', hue='Geschlecht_m', data=df)
# Überblick 
# sns.countplot(x='Zielvariable_ja', hue='Kontaktart_Handy', data=df)
# Überblick 
# sns.countplot(x='Zielvariable_ja', hue='Ergebnis letzte Kampagne_Erfolg', data=df)
# sns.boxplot(x='Zielvariable_ja', y='Alter', data=df)
# sns.boxplot(x='Zielvariable_ja', y='Dauer', data=df)
# sns.boxplot(x='Zielvariable_ja', y='Kontostand', data=df)
# sns.boxplot(x='Zielvariable_ja', y='Anzahl der Ansprachen', data=df)
#df.drop(['Stammnummer', 'Tag', 'Anruf-ID'], axis=1, inplace=True)
df.drop(['Tag', 'Anruf-ID'], axis=1, inplace=True)
df.head()
# Aufsplitten der Features in erklärende Variablen und Zielvariable
X = df.drop(['Zielvariable_ja', 'Stammnummer'], axis=1)
y = df['Zielvariable_ja']
from sklearn.cross_validation import train_test_split
# Aufsplitten in Training set (70%) und Test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
# initiieren des Models (Parameterauswahl - siehe unten Logistic Regression-2)
logmodel = LogisticRegression(penalty='l1', C=100)
# Trainings Daten ins Model laden
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
predictions_prob = logmodel.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import auc
[fpr, tpr, thr] = roc_curve(y_test, predictions_prob)
print('Train/Test split results:')
print(logmodel.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, predictions))
print(logmodel.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, predictions_prob))
print(logmodel.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
print(logmodel.__class__.__name__+" roc_auc_score is %2.3f" % roc_auc_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" mean_squared_error is %2.3f" % mean_squared_error(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" f1_score is %2.3f" % f1_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" precision_score is %2.3f" % precision_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" r2_score is %2.3f" % r2_score(y_test, logmodel.predict(X_test)))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
##for i in range(len(X_test)):
##	print("df['Zielvariable_ja']=%s, Predicted=%s" % (df['Zielvariable_ja'][i], predictions[i]))
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
df.head()
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
predictions_prob = dtree.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, predictions_prob)
print('Train/Test split results:')
print(logmodel.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, predictions))
print(logmodel.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, predictions_prob))
print(logmodel.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
print(logmodel.__class__.__name__+" roc_auc_score is %2.3f" % roc_auc_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" mean_squared_error is %2.3f" % mean_squared_error(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" f1_score is %2.3f" % f1_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" precision_score is %2.3f" % precision_score(y_test, logmodel.predict(X_test)))
print(logmodel.__class__.__name__+" r2_score is %2.3f" % r2_score(y_test, logmodel.predict(X_test)))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print ('\n')
print(classification_report(y_test, predictions))
# Fehler liegt bei 574 + 690
from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(n_estimators = 300, oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 1)
rfc = RandomForestClassifier(n_estimators = 100, max_depth= 25, max_features = 20, min_samples_leaf = 3)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_pred_prob = rfc.predict_proba(X_test)[:, 1]
# print(rfc_pred_prob)

#for i in range(len(X_test)):
#    print("df['Zielvariable_ja']=%s, Predicted=%s" % (df['Zielvariable_ja'][i], rfc_pred[i]))
#    print("df['Zielvariable_ja']=%s, Predicted prob=%s" % (df['Zielvariable_ja'][i], rfc_pred_prob[i]))

# X_test.info()
[fpr, tpr, thr] = roc_curve(y_test, rfc_pred_prob)
print('Train/Test split results:')
print(logmodel.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, rfc_pred))
print(logmodel.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, rfc_pred_prob))
print(logmodel.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
print(logmodel.__class__.__name__+" roc_auc_score is %2.3f" % roc_auc_score(y_test, rfc_pred))
print(logmodel.__class__.__name__+" mean_squared_error is %2.3f" % mean_squared_error(y_test, rfc_pred))
print(logmodel.__class__.__name__+" f1_score is %2.3f" % f1_score(y_test, rfc_pred))
print(logmodel.__class__.__name__+" precision_score is %2.3f" % precision_score(y_test, rfc_pred))
print(logmodel.__class__.__name__+" r2_score is %2.3f" % r2_score(y_test, rfc_pred))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
print(confusion_matrix(y_test, rfc_pred))
print ('\n')
print(classification_report(y_test, rfc_pred))
# Fehler liegt bei 488 + 249
df_test = pd.read_csv('TestData.csv', encoding='latin-1', sep=";")
df_test = pd.get_dummies(df_test)
df_test.columns
# redundante Infos droppen
df_test = df_test.drop(['Zielvariable', 'Geschlecht_w', 'Art der Anstellung_Unbekannt', 'Familienstand_geschieden', 'Schulabschluß_Unbekannt', 'Ausfall Kredit_nein', 'Haus_nein', 'Kredit_nein', 'Kontaktart_Unbekannt', 'Ergebnis letzte Kampagne_Unbekannt'
], axis=1)
# NaN-Werte in Spalte 'Tage seit letzter Kampagne' sind beim Plotten ein Problem -> durch mean (Mittelwert) ersetzten
# sns.heatmap(df.isnull(), yticklabels=False, cbar=Fals, cmap='viridis')
df_test = df_test.groupby(df_test.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
# Aufbau des dataframes prüfen
df_test.info()
df_test.drop(['Tag', 'Anruf-ID'], axis=1, inplace=True)
df_test_copy = df_test.copy()
df_test.drop(['Stammnummer'], axis=1, inplace=True)
df_test.head()
df_test_copy.head()
df_test.info()
rfc_pred = rfc.predict(df_test)
rfc_pred_prob = rfc.predict_proba(df_test)[:, 1]
# print(rfc_pred_prob)

#for i in range(len(X_test)):
#    print("df['Zielvariable_ja']=%s, Predicted=%s" % (df['Zielvariable_ja'][i], rfc_pred[i]))
#    print("df['Zielvariable_ja']=%s, Predicted prob=%s" % (df['Zielvariable_ja'][i], rfc_pred_prob[i]))

# X_test.info()
df_test_copy['predictions'] = rfc_pred
df_test_copy['predictions prob'] = rfc_pred_prob
df_test_copy.info()
#df_test_copy = df_test_copy.loc[:, ['Stammnummer','predictions','predictions prob']]
df_test_copy = df_test_copy.loc[:, ['Stammnummer','predictions prob']]
#df_test_copy
import csv
df_test_copy.to_csv('Loesung_final.csv', sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, index=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
# da die Zielvariable nur 2 Werte annimmt (!= Mehrklassenwert), ist der nächste Schritt nicht notwendig
lb = LabelBinarizer()
df['Zielvariable_ja'] = lb.fit_transform(df['Zielvariable_ja'].values)
targets = df['Zielvariable_ja']
X_train, X_test, y_train, y_test = train_test_split(X, targets, stratify=targets)
# n_jobs=-1 => Prozessorbeschränkung aufheben
clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'min_samples_split': [3, 5, 10], # minimum number of samples that must be present from your data in order for a split to occur
    'n_estimators' : [100, 300], # number of trees your want to build within a Random Forest before aggregating the predictions
    'max_depth': [3, 5, 15, 25], # how deep do you want to make your trees - avoids overfitting
    'max_features': [3, 5, 10, 20] # max number of features considered when finding best split
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score)
}
def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

#grid_search_clf = grid_search_wrapper(refit_score='precision_score')
#results = pd.DataFrame(grid_search_clf.cv_results_)
#results = results.sort_values(by='mean_test_precision_score', ascending=False)

#results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()

"""
Best params for precision_score
{'max_depth': 5, 'max_features': 5, 'min_samples_split': 3, 'n_estimators': 300}

Confusion matrix of Random Forest optimized for precision_score on the test data:
     pred_neg  pred_pos
neg      6933        12
pos       880        45
"""
#grid_search_clf = grid_search_wrapper(refit_score='recall_score')
#results = pd.DataFrame(grid_search_clf.cv_results_)
#results = results.sort_values(by='mean_test_precision_score', ascending=False)

#results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
"""
Best params for recall_score
{'max_depth': 25, 'max_features': 20, 'min_samples_split': 3, 'n_estimators': 300}

Confusion matrix of Random Forest optimized for recall_score on the test data:
     pred_neg  pred_pos
neg      6678       267
pos       473       452
"""
#grid_search_clf = grid_search_wrapper(refit_score='accuracy_score')
#results = pd.DataFrame(grid_search_clf.cv_results_)
#results = results.sort_values(by='mean_test_precision_score', ascending=False)

#results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()

"""
Best params for accuracy_score
{'max_depth': 15, 'max_features': 10, 'min_samples_split': 3, 'n_estimators': 300}

Confusion matrix of Random Forest optimized for accuracy_score on the test data:
     pred_neg  pred_pos
neg      6751       194
pos       530       395
"""
#grid_search_clf = grid_search_wrapper(refit_score='roc_auc')
#results = pd.DataFrame(grid_search_clf.cv_results_)
#results = results.sort_values(by='mean_test_precision_score', ascending=False)

#results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
"""
Best params for roc_auc
{'max_depth': 25, 'max_features': 20, 'min_samples_split': 3, 'n_estimators': 100}

Confusion matrix of Random Forest optimized for roc_auc on the test data:
     pred_neg  pred_pos
neg      6676       269
pos       486       439
"""
# Aufsplitten der Features in erklärende Variablen und Zielvariable
X = df.drop(['Zielvariable_ja', 'Stammnummer'], axis=1)
y = df['Zielvariable_ja']
from sklearn.cross_validation import train_test_split
# da die Zielvariable nur 2 Werte annimmt (!= Mehrklassenwert), ist der nächste Schritt nicht notwendig
lb = LabelBinarizer()
df['Zielvariable_ja'] = lb.fit_transform(df['Zielvariable_ja'].values)
targets = df['Zielvariable_ja']
X_train, X_test, y_train, y_test = train_test_split(X, targets, stratify=targets)
from sklearn.linear_model import LogisticRegression
# initiieren des Models (default parameter)
clf = LogisticRegression()

param_grid = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score)
}
# grid_search_clf = grid_search_wrapper(refit_score='roc_auc')
# results = pd.DataFrame(grid_search_clf.cv_results_)
# results = results.sort_values(by='mean_test_precision_score', ascending=False)

# results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score']].round(3).head()

"""
Best params for roc_auc
{'C': 100, 'penalty': 'l1'}

Confusion matrix of Random Forest optimized for roc_auc on the test data:
     pred_neg  pred_pos
neg      6779       166
pos       618       307
"""

