import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('titanic.csv')

df
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df,palette='cubehelix').set_title('Survived Passenger (0=Not Survived; 1=Survived)')

df['Survived'].value_counts(normalize=True)*100
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df,palette='cubehelix').set_title('Survived Passenger Based on Gender') 
ss =  df.groupby(["Survived", "Sex"]).size()

ss
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=df,palette='cubehelix'). set_title('Survived Passenger Based on Ticket Class')
sp =  df.groupby(["Survived", "Pclass"]).size()

sp
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(x='Survived',hue='SibSp',data=df,palette='cubehelix'). set_title('Survived Passenger Based on siblings / spouses aboard the Titanic')

plt.legend(loc='upper right', title="Sibilings/Spouse")
si =  df.groupby(["Survived", "SibSp"]).size()

si
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(x='Survived',hue='Parch',data=df,palette='cubehelix'). set_title('Survived Passenger Based on  Parents / Children Aboard The Titanic')

plt.legend(loc='upper right', title="Parents / Children")
sa =  df.groupby(["Survived", "Parch"]).size()

sa
df['Fare'].hist(bins=40,figsize=(18,8))
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(x='Survived',hue='Embarked',data=df,palette='cubehelix'). set_title('Survived Passenger Based on  Port of Embarkation')

plt.legend(loc='upper right', title="Port")
se =  df.groupby(["Survived", "Embarked"]).size()

se
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='cubehelix')
df.isnull().sum().sort_values(ascending=False)
print('Percent of missing "Cabin" records is %.2f%%' %((df['Cabin'].isnull().sum()/df.shape[0])*100))

print('Percent of missing "Age" records is %.2f%%' %((df['Age'].isnull().sum()/df.shape[0])*100))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
print(df['Embarked'].value_counts())

sns.countplot(x='Embarked', data=df, palette='cubehelix')

plt.show()
print('The most common boarding port of embarkation is %s.' %df['Embarked'].value_counts().idxmax())
df['Embarked'] = df['Embarked'].fillna('S')
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='cubehelix')
df.drop('Cabin',axis=1,inplace=True)
df.head()
df.dropna(inplace=True)
sex = pd.get_dummies(df['Sex'],drop_first=True)

embarked = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df = pd.concat([df,sex,embarked], axis=1)

df.head()
from sklearn.model_selection import train_test_split, cross_val_score
x= df.drop('Survived', axis=1)

y=df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.469,  random_state=100, stratify=y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

model.fit(x,y)
predictions = model.predict(x_test)

x_test
predictions 
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import classification_report,confusion_matrix,roc_curve

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
# check classification scores of logistic regression



y_pred = model.predict(x_test)

y_pred_proba = model.predict_proba(x_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

idx = np.min(np.where(tpr > 0.95))

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  

      "and a specificity of %.3f" % (1-fpr[idx]) + 

      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
scores_accuracy = cross_val_score(model, x, y, cv=10, scoring='accuracy')

scores_log_loss = cross_val_score(model, x, y, cv=10, scoring='neg_log_loss')

scores_auc = cross_val_score(model, x, y, cv=10, scoring='roc_auc')

print('K-fold cross-validation results:')

print(model.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())

print(model.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())

print(model.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())
from sklearn.model_selection import GridSearchCV

import numpy as np

param_grid = {'C': np.arange(1e-05, 3, 0.1)}

gscv = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=10)

gscv.fit(x,y)
y_pred = gscv.predict(x_test)

y_pred
gscv.best_score_
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))
predict = pd.DataFrame(data=predictions, columns=['Survived'])

predict

x_test

predict.to_csv('predict.csv')

x_test.to_csv('test.csv')