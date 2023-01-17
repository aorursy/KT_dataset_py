import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
directory = "/kaggle/input/airline-passenger-satisfaction/"

feature_tables = ['train.csv', 'test.csv']



df_train = directory + feature_tables[0]

df_test = directory + feature_tables[1]



# Create dataframes

print(f'Reading csv from {df_train}...')

df = pd.read_csv(df_train)

print('...Complete')



print(f'Reading csv from {df_train}...')

df2 = pd.read_csv(df_test)

print('...Complete')
df.head()
df.info()
df.isnull().sum()
df.isnull().sum()/len(df)
df.describe().transpose()
plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')

plt.tight_layout
df['satisfaction'].value_counts()
sns.countplot(x='satisfaction',data=df)
def satisfac(string):

    if string == 'satisfied': return 1

    else: return 0



df['satisfactionN'] =df['satisfaction'].apply(satisfac)    
df.head(5)
df.drop('satisfaction',inplace=True,axis=1)
plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot = True,cmap='coolwarm')
df.corr()['satisfactionN'].sort_values().drop('satisfactionN').plot(kind='bar')
df['Online boarding'].value_counts()
df['Online boarding'].plot(kind='hist',ec='black')
sns.boxplot(x='satisfactionN',y = 'Online boarding',data=df)
df.head()
GenderN = pd.get_dummies(df['Gender'],drop_first=True)

CustomerN = pd.get_dummies(df['Customer Type'],drop_first=True)

TypeN = pd.get_dummies(df['Type of Travel'],drop_first=True)

ClassN = pd.get_dummies(df['Class'],drop_first=True)

df = pd.concat([df,GenderN,CustomerN,TypeN,ClassN],axis =1)

df.drop(['Gender','Customer Type','Type of Travel','Class'],inplace =True,axis = 1)
plt.figure(figsize=(25,20))

sns.heatmap(df.corr(),annot = True,cmap='coolwarm')
df.corr()['satisfactionN'].sort_values().drop('satisfactionN').plot(kind='bar')
df.corr()['Online boarding'].sort_values().drop(['Online boarding','satisfactionN']).plot(kind='bar')
sns.boxplot(x='Inflight wifi service',y = 'Online boarding',data=df)
df['Flight Distance'].iplot(kind='hist',bins=50)
df['Age'].iplot(kind='hist',bins=50)
import plotly.express as px

fig = px.box(df, x="satisfactionN", y="Age", color="Eco")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.show()
sns.lmplot(x='Departure Delay in Minutes',y='Arrival Delay in Minutes',data=df)
df.drop('Arrival Delay in Minutes',axis=1,inplace=True)
df.isnull().sum()/len(df)

# Only 0.002% data is missing, so we can drop the rows 

# Data was missing in only "Arrival delay in Minutes" column , so these steps are not necessary
df.dropna(axis=0,inplace=True)
df.drop(['Unnamed: 0','id'],axis=1,inplace=True)
#importing the libraries

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB
X_train = df.drop('satisfactionN',axis=1)

y_train = df['satisfactionN']
df2['satisfactionN'] =df2['satisfaction'].apply(satisfac)

GenderN = pd.get_dummies(df2['Gender'],drop_first=True)

CustomerN = pd.get_dummies(df2['Customer Type'],drop_first=True)

TypeN = pd.get_dummies(df2['Type of Travel'],drop_first=True)

ClassN = pd.get_dummies(df2['Class'],drop_first=True)

df2 = pd.concat([df2,GenderN,CustomerN,TypeN,ClassN],axis =1)

df2.drop(['Gender','Customer Type','Type of Travel','Class'],inplace =True,axis = 1)

df2.drop('Arrival Delay in Minutes',axis=1,inplace=True)

df2.drop(['Unnamed: 0','id'],axis=1,inplace=True)
df2.drop('satisfaction',axis=1,inplace=True)

X_test = df2.drop('satisfactionN',axis=1)

y_test= df2['satisfactionN']
print("X_train {}\nX_test {}\ny_train {}\ny_test {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
classifier1 = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,n_jobs=-1)

classifier1.fit(X_train,y_train)
y_pred = classifier1.predict(X_test)
# importing accuracy parameters

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))

print('\n\n\n')

print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))

print('\n')

print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))

acc_random_forest = accuracy_score(y_test,y_pred)
classifier2 = XGBClassifier(n_estimators = 500,n_jobs=-1)

classifier2.fit(X_train,y_train)
#Predicting on test set results

y_pred = classifier2.predict(X_test)
# importing accuracy parameters

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))

print('\n\n\n')

print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))

print('\n')

print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))

acc_xgboost = accuracy_score(y_test,y_pred)
classifier3 = GaussianNB()

classifier3.fit(X_train,y_train)
#Predicting on test set results

y_pred = classifier3.predict(X_test)
# importing accuracy parameters

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))

print('\n\n\n')

print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))

print('\n')

print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))

acc_naive_bayes = accuracy_score(y_test,y_pred)
print('Accuracy:-\n')

print("Random Forest {}\nXGBoost {}\nNaive Bayes {}\n".format(acc_random_forest,acc_xgboost,acc_naive_bayes))