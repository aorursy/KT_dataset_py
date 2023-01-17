import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
ben=pd.read_csv('../input/dm-assignment-3/train_benign.csv')
mal=pd.read_csv('../input/dm-assignment-3/train_malware.csv')
test = pd.read_csv('../input/dm-assignment-3/Test_data.csv')
ben['1809']=0
mal['1809']=1
file_name=test['FileName']
test = test.drop(['Unnamed: 1809','FileName'],axis=1)
file_name
data=[ben,mal]
train=pd.concat(data)
train.info()
train=train.drop_duplicates()
null_columns = train.columns[train.isnull().any()] 
null_columns
train.info()
train.head()
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)] 
train.drop(to_drop, axis=1, inplace=True)
to_drop
train=train.drop(columns=['FileName'],axis=1)
X=train.drop(columns=['1809'],axis=1)
y=train['1809']
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),n_estimators=550,random_state=42).fit(X,y)
print(roc_auc_score(ada.predict(X),y))
test.drop(to_drop, axis=1, inplace=True)
pred = ada.predict(test)
final = pd.DataFrame(columns=['FileName','Class'])
final['FileName'] = file_name
final['Class'] = pred
final.head()
from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(final)
