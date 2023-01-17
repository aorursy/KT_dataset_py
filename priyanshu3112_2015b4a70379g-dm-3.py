import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
malware_data = pd.read_csv('../input/opcode_frequency_malware.csv')

malware_data.head()



benign_data= pd.read_csv('../input/opcode_frequency_benign.csv')

benign_data.head()
data1=pd.read_csv('../input/Test_data.csv')

X1_test=data1.copy()



X1_test['file']=X1_test['FileName']

X1_test.head()



col=data1.columns

X1_test.drop(columns=col,axis=1,inplace=True)

# X1_test.head()

X1_test['FileName']=X1_test['file']

X1_test.drop(['file'],axis=1,inplace=True)

X1_test.head()



data1.drop(['FileName','Unnamed: 1809'],axis=1,inplace=True)



benign_data['1809']=0

malware_data['1809']=1

final_data=pd.concat([malware_data,benign_data])

final_data.head()



final_data['1809'].value_counts()



final_data=final_data.drop_duplicates()

final_data.drop(columns=['FileName'],axis=1,inplace=True)



X=final_data.drop(columns=['1809'],axis=1)

y=final_data['1809']
l=[]

for i in range(len(final_data.columns)):

    c0=0

    for j in final_data.iloc[:,i]:

        if j==0:

            c0=c0+1

    if c0>3350:

        l.append(final_data.columns[i])

len(l)
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=42,train_size=0.8)    

model=rfc(n_estimators=156,max_features=72,criterion='gini',max_depth=16,

          random_state=42,class_weight='balanced')

# model=rfc(n_estimators=30,max_features=72,criterion='gini',max_depth=13,

#           random_state=45,class_weight='balanced')

model.fit(train_X,train_y)

val_predictions=model.predict(val_X)

roc_auc_score(val_y,val_predictions)
model.fit(X,y)
y1_pred=model.predict(data1)

y1_pred
X1_test['Class']=y1_pred

X1_test['Class'].value_counts()
X1_test.to_csv("third.csv",index=False)

X1_test
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



create_download_link(X1_test)