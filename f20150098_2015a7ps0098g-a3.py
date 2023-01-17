import numpy as np

import pandas as pd
train_data1 = pd.read_csv('../input/opcode_frequency_benign.csv')

train_data2 = pd.read_csv('../input/opcode_frequency_malware.csv')
train_data1['Class']=0

train_data2['Class']=1
train_data1.head()

#train_data2.head()
train_data=train_data1.append(train_data2)

train_data.head()
train_data = train_data.drop(['FileName'], 1)
train_data.head()
train_data=train_data.sample(frac=1).reset_index(drop=True)

train_data.head()
label=train_data['Class']
#label
train_data = train_data.drop(['Class'], 1)
train_data.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()



Xstd = ss.fit_transform(train_data.values)

Xstd
from sklearn.decomposition import PCA

pca = PCA(n_components=1000)

pc = pca.fit_transform(Xstd)

pc
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=42)

pc,label = ros.fit_resample(pc,label)

len(label)
X_train, X_test, y_train, y_test = train_test_split(pc, label,test_size=0.08)
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=50).fit(X_train,y_train)

accuracy_dtree = dtree_model.score(X_test,y_test)

accuracy_dtree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=90, random_state = 52)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)

accuracy_knn = knn.score(X_test,y_test)

accuracy_knn
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train,y_train)

accuracy_gnb = gnb.score(X_test,y_test)

accuracy_gnb
from sklearn.svm import SVC

svm_model = SVC(kernel = 'linear',C = 1).fit(X_train, y_train)

svm_predict = svm_model.predict(X_test)



accuracy = svm_model.score(X_test,y_test)

accuracy
test_data = pd.read_csv('../input/Test_data.csv')

test_data.head()

IDs=test_data['FileName']

test_data=test_data.drop(['FileName'],axis=1)

test_data.head()
test_data=test_data.drop(['Unnamed: 1809'],axis=1)
test_data.head()
Xtstd = ss.transform(test_data.values)

Xtstd=pca.transform(Xtstd)
opDtree= rf.predict(Xtstd)

opDtreeList=opDtree.tolist()
res1 = pd.DataFrame(opDtreeList)

final = pd.concat([IDs, res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final['Class'] = final.Class.astype(int)
final.to_csv('submission.csv', index = False,  float_format='%.f')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = df.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link(final)