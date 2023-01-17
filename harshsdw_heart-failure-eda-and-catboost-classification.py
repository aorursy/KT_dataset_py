import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import plotly.express as px
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
df.info()
df.rename(columns={'creatinine_phosphokinase' : 'cpk' ,'DEATH_EVENT':'death_event'}  , inplace=True)

df.head()
for i in df.columns:

  print(i,df[i].nunique())
ds = df['anaemia'].value_counts().reset_index()

ds.columns = ['anaemia', 'count']

fig = px.bar(ds, x='anaemia', y="count", orientation='v', title='Count of Patients with Anaemia', width=500)

fig.show()
pd.crosstab(df.anaemia  ,df.death_event).plot(kind='bar')

plt.title('Death Event as per Anaemia')

plt.xlabel('Anaemia')

plt.ylabel('Death')

plt.show()
ds = df['smoking'].value_counts().reset_index()

ds.columns = ['smoking', 'count']

fig = px.bar(ds, x='smoking', y="count", orientation='v', title='Count of Patients who Smoke', width=500)

fig.show()
pd.crosstab(df.smoking ,df.death_event).plot(kind='bar')

plt.title('Death Event as per Smoking ')

plt.xlabel('Smoking')

plt.ylabel('Death')

plt.show()
print("Percentage of people who died and are smokers:", 

      df["death_event"][df["smoking"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of people who died and are not smokers:", 

      df["death_event"][df["smoking"] == 0].value_counts(normalize = True)[1]*100)

ds = df['high_blood_pressure'].value_counts().reset_index()

ds.columns = ['high_blood_pressure', 'count']

fig = px.bar(ds, x='high_blood_pressure', y="count", orientation='v', title='Count of Patients with high blood pressure', width=500)

fig.show()
pd.crosstab(df.high_blood_pressure  ,df.death_event).plot(kind='bar')

plt.title('Death Event as per BLood Pressure ')

plt.xlabel('BP')

plt.ylabel('Death')

plt.show()
ds = df['diabetes'].value_counts().reset_index()

ds.columns = ['diabetes', 'count']

fig = px.bar(ds, x='diabetes', y="count", orientation='v', title='Count of Patients with diabetes', width=500)

fig.show()
pd.crosstab(df.diabetes ,df.death_event).plot(kind='bar')

plt.title('Death Event as per diabetes ')

plt.xlabel('diabetes ')

plt.ylabel('Death')

plt.show()
ds = df['sex'].value_counts().reset_index()

ds.columns = ['sex', 'count']

fig = px.bar(ds, x='sex', y="count", orientation='v', title='Count of Patients according to sex', width=500)

fig.show()
pd.crosstab(df.sex ,df.death_event).plot(kind='bar')

plt.title('Death Event as per Sex')

plt.xlabel('Sex')

plt.ylabel('Death')

plt.show()
print("Females:", 

      df["death_event"][df["sex"] == 0].value_counts(normalize = True)[1]*100)



print("Males:", 

      df["death_event"][df["sex"] == 1].value_counts(normalize = True)[1]*100)

g_30=list()

g_50=list()

g_70=list()

greater70 = list()

for i in df.age:

  if i<=30:

    g_30.append(1)

    g_50.append(0)

    g_70.append(0)

    greater70.append(0)



  elif i>30 and i<=50:

    g_30.append(0)

    g_50.append(1)

    g_70.append(0)

    greater70.append(0)

  

  elif i>50 and i<=70:

    g_30.append(0)

    g_50.append(0)

    g_70.append(1)

    greater70.append(0)

  

  elif i>70:

    g_30.append(0)

    g_50.append(0)

    g_70.append(0)

    greater70.append(1)
df['age_till_30'] = g_30

df['age_bet_30_50'] = g_50

df['age_bet_50_70'] = g_70

df['age_gret_70'] = greater70

df.age_till_30.value_counts()
ds = df['age_till_30'].value_counts().reset_index()

ds.columns = ['age_till_30', 'count']

fig = px.bar(ds, x='age_till_30', y="count", orientation='v', title='Count of Patients with age till 30', width=500)

fig.show()
pd.crosstab(df.age_till_30 ,df.death_event).plot(kind='bar')

plt.title('Death Event for people with Age till 30')

plt.xlabel('Age')

plt.ylabel('Death')

plt.show()
df.age_bet_30_50.value_counts()
ds = df['age_bet_30_50'].value_counts().reset_index()

ds.columns = ['age_bet_30_50', 'count']

fig = px.bar(ds, x='age_bet_30_50', y="count", orientation='v', title='Count of Patients with age between 30 and 50', width=500)

fig.show()
pd.crosstab(df.age_bet_30_50 ,df.death_event).plot(kind='bar')

plt.title('Death Event for people with Age between 30 and 50')

plt.xlabel('Age')

plt.ylabel('Death')

plt.show()
print("Mortlity Rate:", 

      df["death_event"][df["age_bet_30_50"] == 1].value_counts(normalize = True)[1]*100)
df.age_bet_50_70.value_counts()
ds = df['age_bet_50_70'].value_counts().reset_index()

ds.columns = ['age_bet_50_70', 'count']

fig = px.bar(ds, x='age_bet_50_70', y="count", orientation='v', title='Count of Patients with age between 50 and 70', width=500)

fig.show()
pd.crosstab(df.age_bet_50_70 ,df.death_event).plot(kind='bar')

plt.title('Death Event for people with Age bet 50 and 70')

plt.xlabel('Age')

plt.ylabel('Death')

plt.show()
print("Mortality Rate:", 

      df["death_event"][df["age_bet_50_70"] == 1].value_counts(normalize = True)[1]*100)
df.age_gret_70.value_counts()
ds = df['age_gret_70'].value_counts().reset_index()

ds.columns = ['age_gret_70', 'count']

fig = px.bar(ds, x='age_gret_70', y="count", orientation='v', title='Count of Patients with age greater than 70', width=500)

fig.show()
pd.crosstab(df.age_gret_70 ,df.death_event).plot(kind='bar')

plt.title('Death Event for people with Age greater than 70')

plt.xlabel('Age')

plt.ylabel('Death')

plt.show()
print("Mortality Rate:", 

      df["death_event"][df["age_gret_70"] == 1].value_counts(normalize = True)[1]*100)

fig = px.bar(df, x="diabetes", y="age", color="death_event", title="Long-Form Input")

fig.show()
fig = px.bar(df, x="sex", y="age", color="death_event", title="Long-Form Input")

fig.show()
fig = px.bar(df, x="smoking", y="age", color="death_event", title="Long-Form Input")

fig.show()
import plotly.express as px

fig = px.violin(df, y="age", x="sex", color="death_event", box=True, points="all", hover_data=df.columns)

fig.update_layout(title_text="Analysis of Age and Sex on Death Event")

fig.show()
import plotly.express as px

fig = px.violin(df, y="age", x="smoking", color="death_event", box=True, points="all", hover_data=df.columns)

fig.update_layout(title_text="Analysis of Age and Smoking on Death Event")

fig.show()
X = df.drop(columns=('death_event'),axis=1)

y = df.death_event
scaler = MinMaxScaler(feature_range=(0,100))

X['platelets'] = scaler.fit_transform(X[['platelets']])

X['cpk'] = scaler.fit_transform(X[['cpk']])

X.head()
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.25, random_state=1 )
y.value_counts()
class_weight = {0:1 , 1:2}

model = CatBoostClassifier(n_estimators=400  ,  depth = 4 , class_weights = class_weight)

model.fit(X_train , y_train)
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, model.predict(X_train))))

print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, model.predict(X_test))))
model = CatBoostClassifier(n_estimators=400  ,  depth = 4)

model.fit(X_train , y_train)
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, model.predict(X_train))))

print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, model.predict(X_test))))
pred = model.predict(X_test)



print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred)

print(cm)

print('True Positive' , cm[0,0])

print('False Positive' , cm[0,1])

print('True Negative' , cm[1,1])

print('False Negative' , cm[1,0])
plot_confusion_matrix(model, X_test, y_test)

plt.show()