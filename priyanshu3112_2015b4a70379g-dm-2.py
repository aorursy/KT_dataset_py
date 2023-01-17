import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('../input/train.csv')

data1=pd.read_csv('../input/test.csv')

data1.head()

X1_test=data1.copy()
data.replace('?',np.NaN,inplace=True)

data1.replace('?',np.NaN,inplace=True)



data.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV',

           'Teen','Fill','COB FATHER','COB MOTHER','COB SELF','Detailed','WorkingPeriod','Weight'],axis=1,inplace=True)

data1.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV',

            'Teen','Fill','COB FATHER','COB MOTHER','COB SELF','Detailed','WorkingPeriod','Weight'],axis=1,inplace=True)



data1['Hispanic'].fillna(data1['Hispanic'].mode()[0],inplace=True)



data.dropna(inplace=True)

y=data['Class']



X1=data.copy()

X1.drop(['Class'],axis=1,inplace=True)

X1.columns



X=pd.get_dummies(X1,columns=['Married_Life','Cast','Hispanic', 'Sex', 'Full/Part','Tax Status',

                            'NOP', 'Citizen', 'Vet_Benefits','Schooling','Summary','Own/Self'])

X.head()

X11=pd.get_dummies(data1,columns=['Married_Life','Cast','Hispanic', 'Sex', 'Full/Part','Tax Status',

                            'NOP', 'Citizen', 'Vet_Benefits','Schooling','Summary','Own/Self'])

X11.head()
f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=7)

model=dtc(max_depth=20)

model.fit(train_X,train_y)

val_predictions=model.predict(val_X)

roc_auc_score(val_y,val_predictions)
from sklearn.naive_bayes import GaussianNB as NB

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=45)

model = NB()



model.fit(train_X,train_y)

val_predictions=model.predict(val_X)

roc_auc_score(val_y,val_predictions)
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=45)    

model=rfc(n_estimators=40,max_features=14,criterion='gini',max_depth=16,

          class_weight='balanced',random_state=45,min_samples_split=467)

model.fit(train_X,train_y)

val_predictions=model.predict(val_X)

roc_auc_score(val_y,val_predictions)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=45)

model1 = AdaBoostClassifier(n_estimators=3,base_estimator=model,random_state=45)

model1.fit(train_X,train_y)

val_predictions=model1.predict(val_X)

roc_auc_score(val_y,val_predictions)
model1.fit(X,y)
y1_pred=model1.predict(X11)

y1_pred
col=['Age', 'Worker Class', 'IC', 'OC', 'Schooling', 'Timely Income',

       'Enrolled', 'Married_Life', 'MIC', 'MOC', 'Cast', 'Hispanic', 'Sex',

       'MLU', 'Reason', 'Full/Part', 'Gain', 'Loss', 'Stock', 'Tax Status',

       'Area', 'State', 'Detailed', 'Summary', 'Weight', 'MSA', 'REG', 'MOVE',

       'Live', 'PREV', 'NOP', 'Teen', 'COB FATHER', 'COB MOTHER', 'COB SELF',

       'Citizen', 'Own/Self', 'Fill', 'Vet_Benefits', 'Weaks',

       'WorkingPeriod']
X1_test.drop(col,axis=1,inplace=True)

X1_test['Class']=y1_pred

X1_test['Class'].value_counts()
X1_test.to_csv("ans.csv",index=False)

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