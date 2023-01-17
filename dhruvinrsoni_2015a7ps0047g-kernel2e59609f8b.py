import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
benign_data = pd.read_csv('../input/opcode_frequency_benign.csv')

mal_data = pd.read_csv('../input/opcode_frequency_malware.csv')
benign_data.isnull().sum().sum()
mal_data.isnull().sum().sum()
benign_data['1809']=0

mal_data['1809']=1

data=[mal_data,benign_data]

data_orig=pd.concat(data)
data_orig['1809'].value_counts()
data_orig=data_orig.drop_duplicates()
data_orig=data_orig.drop(columns=['FileName'],axis=1)
X=data_orig.drop(columns=['1809'],axis=1)

y=data_orig['1809']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
from sklearn.metrics import accuracy_score

depth=[1,3,5,7,10,12,15,20]

train_accuracy=[]

test_accuracy=[]

for i in depth:

    clf = RandomForestClassifier(max_depth=i,n_estimators=50).fit(X_train,y_train)

    train_accuracy.append(accuracy_score(clf.predict(X_train),y_train))

    test_accuracy.append(accuracy_score(clf.predict(X_test),y_test))

    print("for max_depth="+str(i))

    print("training data accuracy:"+str(accuracy_score(clf.predict(X_train),y_train)))

    print("validation accuracy:"+str(accuracy_score(clf.predict(X_test),y_test)))

    print("==============================================")
rfc=RandomForestClassifier(max_depth=20,n_estimators=50).fit(X,y)
test_data = pd.read_csv('../input/Test_data.csv')
test_df = test_data.drop(columns=['Unnamed: 1809','FileName'],axis=1)
y_test_pred = rfc.predict(test_df)
sub = pd.DataFrame()

sub['FileName'] = test_data['FileName']

sub['Class'] = y_test_pred

sub.to_csv('submission.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="submission.csv" href="data:text/csv;base64,{payload}" target="_blank">Download the Submission File</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)