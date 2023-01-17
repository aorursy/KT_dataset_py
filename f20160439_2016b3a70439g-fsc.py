import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv(r'../input/data-mining-assignment-2/train.csv',sep=',')

df.head()
#df.info()
df.isnull().any()
df.describe(include='object')
categorical_features = ['col2','col11','col37','col44','col56']
from sklearn.preprocessing import LabelEncoder 



df1 = df.copy()

df1 = pd.get_dummies(df1, columns=['col2','col11','col37','col44','col56'])
df1.head()
df1=df1.drop(['col11_Yes','col37_Male','col44_No'],axis=1)
df1.shape
y = df1['Class']

X=df1.drop(['ID','Class'],axis=1)
#X_unscaled=X
corr=df1.corr()

import seaborn as sns

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = False)
abs(corr['Class']).sort_values()
X.shape
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=0)
df2=pd.read_csv(r'../input/data-mining-assignment-2/test.csv',sep=',')

df2
df2.fillna(value=df2.mean(),inplace=True)

df2.fillna(value=df2.mode().loc[0],inplace=True) 
from sklearn.preprocessing import LabelEncoder 



df3 = df2.copy()

df3 = pd.get_dummies(df2, columns=['col2','col11','col37','col44','col56'])
df3=df3.drop(['col11_Yes','col37_Male','col44_No'],axis=1)
Xtest=df3.drop(['ID'],axis=1)
Xtest_unscaled=Xtest
y.value_counts()[:10].plot(kind='barh')
from sklearn.ensemble import RandomForestClassifier

clf1=RandomForestClassifier(n_estimators =70, random_state=68,max_depth=11,min_samples_leaf=2, min_samples_split=2)

clf2=RandomForestClassifier(n_estimators =70, random_state=420,max_depth=10)

clf3=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=11, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       n_jobs=None, oob_score=False, random_state=68, verbose=0,

                       warm_start=False)

clf4=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=10, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=85,

                       n_jobs=None, oob_score=False, random_state=68, verbose=0,

                       warm_start=False)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

eclf1 = VotingClassifier(estimators=[('lr1',clf1),('lr2',clf2),('lr3',clf3),('lr4',clf4)], voting='hard' ,weights=[2,2,1,1])
b2=eclf1.fit(X_train ,y_train)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

p1=b2.predict(X_val)

acc_op1 = f1_score(y_val, p1,average='micro')*100  

print(acc_op1)
b2=eclf1.fit(X ,y)
y_pred1=b2.predict(Xtest)
#y_pred1
df4=pd.DataFrame()

df4['ID']=df2['ID']

df4['Class']=y_pred1

df4.head()
#df4.to_csv('DMsub_2016B3A70439G.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df4)