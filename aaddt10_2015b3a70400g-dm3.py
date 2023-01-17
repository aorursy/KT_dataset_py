import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
mal=pd.read_csv('../input/opcode_frequency_malware.csv')

ben=pd.read_csv('../input/opcode_frequency_benign.csv')
mal.head()
ben.head()
ben["Class"]=0
mal["Class"]=1
mal.head()
df=ben.append(mal, ignore_index=True)
len(df)==len(ben)+len(mal)
len(mal)
df.head()
df['Class'].value_counts()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()
df=df.drop(columns=["FileName"])
X=df.drop(columns=['Class'])

y=df['Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)
len(X_train)/(len(X_train) + len(X_test))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
abc= AdaBoostClassifier(n_estimators=550, random_state=42)
abc.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
accuracy_score(abc.predict(X_test),y_test)
accuracy_score(abc.predict(X_train),y_train)
df_test=pd.read_csv('../input/Test_data.csv')

df_test.head()
FileName=df_test["FileName"]
df_test=df_test.drop(columns=["FileName", "Unnamed: 1809"])
df_test.head()
ys = abc.predict(df_test)
len(FileName)
dicty= {"FileName" : list(FileName), "Class" : list(ys)}
sub=pd.DataFrame.from_dict(dicty)
sub=sub[['FileName', 'Class']]
sub.tail()
sub.to_csv("subs1.csv", index=False)
#cp subs1.csv "gdrive/My Drive/DM3/Subs" 
sub["Class"].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)



create_download_link(sub)