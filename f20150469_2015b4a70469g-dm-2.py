import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

pd.options.mode.use_inf_as_na = True
data = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

indexes=data_test.copy()

data.head()
col=['Age', 'Worker Class', 'IC', 'OC', 'Schooling', 'Timely Income',

       'Enrolled', 'Married_Life', 'MIC', 'MOC', 'Cast', 'Hispanic', 'Sex',

       'MLU', 'Reason', 'Full/Part', 'Gain', 'Loss', 'Stock', 'Tax Status',

       'Area', 'State', 'Detailed', 'Summary', 'Weight', 'MSA', 'REG', 'MOVE',

       'Live', 'PREV', 'NOP', 'Teen', 'COB FATHER', 'COB MOTHER', 'COB SELF',

       'Citizen', 'Own/Self', 'Fill', 'Vet_Benefits', 'Weaks',

       'WorkingPeriod']

indexes.drop(col,axis=1,inplace=True)
df = data

df_test = data_test
df.replace('?',np.NaN,inplace=True)

df_test.replace('?',np.NaN,inplace=True)
df = df.drop(['Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','Fill','Detailed','ID'],axis=1)

df_test = df_test.drop(['Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','Fill','Detailed','ID'],axis=1)

df.head()
sns.factorplot(x='Married_Life',y='Class',data=df,kind='bar')

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(14,10))

sns.heatmap(df.corr(),annot=True,fmt='.2f')

plt.show()
du_test = pd.get_dummies(df_test,columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'])

du = pd.get_dummies(df,columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'])

df = du

df_test = du_test

df.head()
X = df.drop('Class',axis=1)

Y = df['Class']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,class_weight='balanced'),n_estimators=300,random_state=47)

ada.fit(X,Y)



val_predictions = ada.predict(X)

roc_auc_score(Y,val_predictions)
# from sklearn.ensemble import RandomForestClassifier as rfc

# model=rfc(class_weight='balanced',n_estimators=300, max_depth=1,random_state=47,min_samples_leaf=50)

# model.fit(X_train,Y_train)

# val_predictions=model.predict(X_test)

# roc_auc_score(Y_test,val_predictions)
# from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=30, metric_params=None)

# model.fit(X_train,Y_train)

# val_predictions=model.predict(X_test)

# roc_auc_score(Y_test,val_predictions)
# from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier(class_weight='balanced')

# model.fit(X_train,Y_train)

# val_predictions=model.predict(X_test)

# roc_auc_score(Y_test,val_predictions)
preds = ada.predict(df_test)

indexes['Class'] = preds

indexes.to_csv("X1.csv",index=False)
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



create_download_link(indexes)