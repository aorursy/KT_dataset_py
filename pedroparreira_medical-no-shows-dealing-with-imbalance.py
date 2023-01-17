import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE

%matplotlib inline
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
sns.countplot('No-show', data=df)
plt.title('No-Show Distribution', fontsize=14);
perc = df['No-show'].value_counts()[0]/(len(df['No-show']))*100

print("Precentage of 'No' values: ",round(perc,2),"%")
Gender_M = pd.get_dummies(df['Gender'],drop_first=True)
Noshow = pd.get_dummies(df['No-show'],drop_first=True)
dfn = pd.concat([df,Gender_M,Noshow], axis = 1)
dfn.drop(['Gender','No-show'], axis = 1, inplace = True)
dfn.rename(columns = {'M':'Gender M', 'Yes': 'Noshow'}, inplace = True)
dfn['ScheduledDay'] = dfn['ScheduledDay'].apply(np.datetime64)
dfn['AppointmentDay'] = dfn['AppointmentDay'].apply(np.datetime64)
dfn['WaitDays'] = dfn['AppointmentDay']-dfn['ScheduledDay']
dfn['WaitDays'] = dfn['WaitDays'].apply(lambda x: int(x.total_seconds() / (3600 * 24)))
dfn['Month'] = dfn['ScheduledDay'].dt.month
dfn['Year'] = dfn['ScheduledDay'].dt.year
dfn['Day'] = dfn['ScheduledDay'].dt.day
dfn['WDay'] = dfn['ScheduledDay'].apply(lambda x: x.weekday())
monthc = pd.get_dummies(dfn['Month'])
yearc = pd.get_dummies(dfn['Year'])
dayc = pd.get_dummies(dfn['Day'])
wdayc = pd.get_dummies(dfn['WDay'])
dff = pd.concat([dfn,monthc,yearc,dayc,wdayc],axis=1)
dff = dff.drop(['PatientId','AppointmentID','ScheduledDay','AppointmentDay','Neighbourhood','Month','Year','Day','WDay'], axis = 1)
X= dff.drop(['Noshow'], axis = 1)
Y= dff['Noshow']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
print("\n Classifier: ", classifier.__class__.__name__)
training_score = cross_val_score(classifier, X_test, y_test, cv=4)
print("\n Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
predictions = classifier.predict(X_test)
print("\n Classification report: \n",classification_report(y_test,predictions))
print("\n Confusion matrix: \n",confusion_matrix(y_test,predictions))
DS0 = dff[dff['Noshow']==0].sample(frac=0.25)
DS1 = dff[dff['Noshow']==1]
DS = pd.concat([DS0,DS1])
X= DS.drop(['Noshow'], axis = 1)
Y= DS['Noshow']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
print("\n Classifier: ", classifier.__class__.__name__)
training_score = cross_val_score(classifier, X_test, y_test, cv=4)
print("\n Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
predictions = classifier.predict(X_test)
print("\n Classification report: \n",classification_report(y_test,predictions))
print("\n Confusion matrix: \n",confusion_matrix(y_test,predictions))
RD0 = dff[dff['Noshow']==0].drop_duplicates()
RD1 = dff[dff['Noshow']==1]
RD = pd.concat([RD0,RD1])
X= RD.drop(['Noshow'], axis = 1)
Y= RD['Noshow']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
sm = SMOTE()
X_train, y_train = sm.fit_sample(X_train, y_train)
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
print("\n Classifier: ", classifier.__class__.__name__)
training_score = cross_val_score(classifier, X_test, y_test, cv=4)
print("\n Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
predictions = classifier.predict(X_test)
print("\n Classification report: \n",classification_report(y_test,predictions))
print("\n Confusion matrix: \n",confusion_matrix(y_test,predictions))
