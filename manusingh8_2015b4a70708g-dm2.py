import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv", sep=',')

dt = pd.read_csv("../input/test_1.csv", sep=',')

dft = dt
df.head()

dft.head()
df.info()
df = df.drop(['ID'], axis = 1)

df = df.drop(['Enrolled'], axis = 1)

df = df.drop(['MLU'], axis = 1)

df = df.drop(['Reason'], axis = 1)

df = df.drop(['Area','State','PREV','Fill'], axis = 1)

dft = dft.drop(['ID'], axis = 1)

dft = dft.drop(['Enrolled'], axis = 1)

dft = dft.drop(['MLU'], axis = 1)

dft = dft.drop(['Reason'], axis = 1)

dft = dft.drop(['Area','State','PREV','Fill'], axis = 1)

df.head()
dft.head()
df.info()

df= df.replace('?', np.NaN)

dft= dft.replace('?', np.NaN)
df.head()
df['Worker Class'] = df['Worker Class'].fillna(df['Worker Class'].mode()[0])

df['MIC'] = df['MIC'].fillna(df['MIC'].mode()[0])

df['MOC'] = df['MOC'].fillna(df['MOC'].mode()[0])

df['MSA'] = df['MSA'].fillna(df['MSA'].mode()[0])

df['REG'] = df['REG'].fillna(df['REG'].mode()[0])

df['MOVE'] = df['MOVE'].fillna(df['MOVE'].mode()[0])

df['Live'] = df['Live'].fillna(df['Live'].mode()[0])
df.head()
df['Teen'] = df['Teen'].fillna(df['Teen'].mode()[0])

df['COB FATHER'] = df['COB FATHER'].fillna(df['COB FATHER'].mode()[0])

df['COB MOTHER'] = df['COB MOTHER'].fillna(df['COB MOTHER'].mode()[0])
df.head()
dft['MIC'] = dft['MIC'].fillna(dft['MIC'].mode()[0])

dft['MOC'] = dft['MOC'].fillna(dft['MOC'].mode()[0])

dft['Teen'] = dft['Teen'].fillna(dft['Teen'].mode()[0])

dft['COB FATHER'] = dft['COB FATHER'].fillna(dft['COB FATHER'].mode()[0])

dft['COB MOTHER'] = dft['COB MOTHER'].fillna(dft['COB MOTHER'].mode()[0])

dft['MSA'] = dft['MSA'].fillna(dft['MSA'].mode()[0])

dft['REG'] = dft['REG'].fillna(dft['REG'].mode()[0])

dft['MOVE'] = dft['MOVE'].fillna(dft['MOVE'].mode()[0])

dft['Live'] = dft['Live'].fillna(dft['Live'].mode()[0])

dft['Worker Class'] = dft['Worker Class'].fillna(dft['Worker Class'].mode()[0])
df.isnull().values.any()
df.isnull().sum()
df['Hispanic'] = df['Hispanic'].fillna(df['Hispanic'].mode()[0])

df['COB SELF'] = df['COB SELF'].fillna(df['COB SELF'].mode()[0])
dft['Hispanic'] = dft['Hispanic'].fillna(dft['Hispanic'].mode()[0])

dft['COB SELF'] = dft['COB SELF'].fillna(dft['COB SELF'].mode()[0])
df.isnull().values.any()
df.isnull().sum()
dft.isnull().sum()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
df = df.drop(['Weaks'], axis = 1)

dft = dft.drop(['Weaks'], axis = 1)

df = df.drop(['Vet_Benefits'], axis = 1)

dft = dft.drop(['Vet_Benefits'], axis = 1)

df = df.drop(['NOP'], axis = 1)

dft = dft.drop(['NOP'], axis = 1)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
df = df.drop(['IC'], axis = 1)

dft = dft.drop(['IC'], axis = 1)
df.head()
dft.head()
df['Class'].unique()
y_train = df['Class']

x_train = df.drop(['Class'],axis=1)

x_train.head()
x_train = x_train.drop(['Detailed'], axis = 1)

dft = dft.drop(['Detailed'], axis = 1)

for i in ["Worker Class", "Schooling", "Married_Life", "MIC", "MOC", "Cast", "Hispanic","Sex","Full/Part","Tax Status", "Summary", "MSA","REG","MOVE","Live","Teen","COB FATHER","COB MOTHER", "COB SELF", "Citizen"]:

    print(x_train[i].nunique())
for i in ["Worker Class", "Schooling", "Married_Life", "MIC", "MOC", "Cast", "Hispanic","Sex","Full/Part","Tax Status","Summary", "MSA","REG","MOVE","Live","Teen","COB FATHER","COB MOTHER", "COB SELF", "Citizen"]:

    print(dft[i].nunique())
x_train = x_train.drop(['COB FATHER'], axis = 1)

dft = dft.drop(['COB FATHER'], axis = 1)

x_train = x_train.drop(['COB MOTHER'], axis = 1)

dft = dft.drop(['COB MOTHER'], axis = 1)

x_train = x_train.drop(['COB SELF'], axis = 1)

dft = dft.drop(['COB SELF'], axis = 1)

x_train = x_train.drop(['Married_Life'], axis = 1)

dft = dft.drop(['Married_Life'], axis = 1)
x_train.head()
dtest = pd.get_dummies(dft, columns=["Worker Class", "Schooling", "MIC", "MOC", "Cast", "Hispanic","Sex","Full/Part","Tax Status", "Summary", "MSA","REG","MOVE","Live","Teen", "Citizen"])

dtest.head()
dtrain = pd.get_dummies(x_train, columns=["Worker Class", "Schooling", "MIC", "MOC", "Cast", "Hispanic","Sex","Full/Part","Tax Status", "Summary", "MSA","REG","MOVE","Live","Teen", "Citizen"])

dtrain.head()
#Naive Bayes
np.random.seed(42)
from sklearn.naive_bayes import GaussianNB as NB
print(dtrain.shape)
# from sklearn.model_selection import train_test_split

# X_train, y_train = train_test_split(data1, y, test_size=0.0, random_state=42)
nb = NB()

nb.fit(dtrain, y_train)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(dtest)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(dtrain, y_train, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier

#RandomForestClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=12, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
#Logistic Regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
y_pred_LR = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
from sklearn.tree import DecisionTreeClassifier

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier



train_acc = []

test_acc = []

for i in range(1,15):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(X_train,y_train)

    acc_train = dTree.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = dTree.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs Max Depth')

plt.xlabel('Max Depth')

plt.ylabel('Accuracy')
from sklearn.tree import DecisionTreeClassifier



train_acc = []

test_acc = []

for i in range(2,30):

    dTree = DecisionTreeClassifier(max_depth = 7, min_samples_split=i, random_state = 42)

    dTree.fit(X_train,y_train)

    acc_train = dTree.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = dTree.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs min_samples_split')

plt.xlabel('Min samples split')

plt.ylabel('Accuracy')
dTree = DecisionTreeClassifier(max_depth=7, random_state = 42)

dTree.fit(X_train,y_train)

dTree.score(X_val,y_val)
y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred_DT)
roc_auc_score(y_val, y_pred_LR)
roc_auc_score(y_val, y_pred_RF)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(dtest)

print(y_pred_NB.tolist())
list_id=dt['ID'].tolist()

y_pred1=np.array(y_pred_NB).tolist()

d = {'ID':list_id,'Class':y_pred_NB}

dfinal = pd.DataFrame(d)

dfinal.head()
dfinal.to_csv('submit.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html ='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(dfinal)