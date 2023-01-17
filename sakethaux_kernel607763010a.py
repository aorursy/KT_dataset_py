import numpy as np

import pandas as pd

import pickle as pkl

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,confusion_matrix
df = pd.read_csv("/kaggle/input/the-ultimate-halloween-candy-power-ranking/candy-data.csv")

df.head()
df.corr()
def preprocess(df):

    df['chocolate_no_fruit'] = df['fruity'].apply(lambda x : 0 if x==1 else 1)

    dummy_cols = ['fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','pluribus']

    for x in dummy_cols:

        df[x] = df[x].apply(lambda x : 'yes' if x==1 else 'no')

    df.drop(columns=['competitorname','bar','winpercent'],inplace=True)

    df = pd.get_dummies(df,columns=dummy_cols)

    return df



def scaling(df,scaler=None):

    if scaler==None:

        sc=StandardScaler()

        sc.fit(df)

        df = sc.transform(df)

        pkl.dump(sc,open("chocolate_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return df
y = df['chocolate']

X = df.drop(columns=['chocolate'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
X_train = preprocess(X_train)

X_train = scaling(X_train)

X_test = preprocess(X_test)

X_test = scaling(X_test,pkl.load(open("chocolate_scaler.pkl",'rb')))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l1')

logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for chocolate classifier')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
auc(fpr, tpr)