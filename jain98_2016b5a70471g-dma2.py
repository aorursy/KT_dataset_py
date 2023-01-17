import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import cluster 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
 
## Exploring the Data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#import data from csv to dataframe df
df = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv", sep=",",na_values='?') #dataframe object
dftest = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv", sep=",",na_values='?') #dataframe object
df.shape
df.head(10)
#data tranformation
df['col11'] = df['col11'].apply(lambda x: 1 if x=='Yes' else '0')
df['col37'] = df['col37'].apply(lambda x: 1 if x=='Male' else '0')
df['col44'] = df['col44'].apply(lambda x: 1 if x=='Yes' else '0')
df['col56'] = df['col56'].apply(lambda x: 1 if x=='Low' else 2 if x=='Medium' else 3)
#test
dftest['col11'] = dftest['col11'].apply(lambda x: 1 if x=='Yes' else '0')
dftest['col37'] = dftest['col37'].apply(lambda x: 1 if x=='Male' else '0')
dftest['col44'] = dftest['col44'].apply(lambda x: 1 if x=='Yes' else '0')
dftest['col56'] = dftest['col56'].apply(lambda x: 1 if x=='Low' else 2 if x=='Medium' else 3)

cat_columns = ['col2']

#OneHot Encoding
df_onehot = df.copy()
df_onehot = pd.get_dummies(df_onehot, columns=cat_columns)

#test
dftest_onehot = dftest.copy()
dftest_onehot = pd.get_dummies(dftest_onehot, columns=cat_columns)

#
df_onehot=df_onehot.drop(['Class','ID'],axis=1)
dftest_onehot= dftest_onehot.drop('ID',axis=1)
df_onehot.head(10)
df_onehot['col11']=df_onehot['col11'].astype('float')
df_onehot['col37']=df_onehot['col37'].astype('float')
df_onehot['col44']=df_onehot['col44'].astype('float')

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier,RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

x=df_onehot.copy()
test=dftest_onehot.copy()
y=df['Class']
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size=0.33)
[x_train.shape, y_train.shape, x.shape, y.shape, x_cv.shape, y_cv.shape]
model = RandomForestClassifier(n_estimators = 2000, max_depth=10)
model.fit(x_train,y_train)
predict_train=model.predict(x_train)
predict_cv=model.predict(x_cv)
y_train.shape, y_cv.shape
from sklearn.metrics import f1_score,accuracy_score

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)

f1_train = f1_score(y_train,predict_train,average = 'macro')
print('\naccuracy_score on train dataset : ', accuracy_train)
print('\nf1_score on train dataset : ', f1_train)

# predict the target on the test dataset
y_predict_cv = model.predict(x_cv)

# Accuracy Score on test dataset
f1_test = f1_score(y_cv,y_predict_cv,average = 'macro')

accuracy_test = accuracy_score(y_cv,y_predict_cv)

print('\naccuracy_score on test dataset : ', accuracy_test)
print('\nf1_score on test dataset : ', f1_test)
model = RandomForestClassifier(n_estimators = 2000,max_depth=10)
model.fit(x,y)
y_test=model.predict(x)
y_pred=model.predict(test)
y_test.shape, y_pred.shape
# Accuray Score on train dataset
accuracy_train = accuracy_score(y_test,y)
f1_train = f1_score(y_test,y,average='macro')
print('\naccuracy_score on train dataset : ', accuracy_train)
print('\nf1_score on train dataset : ', f1_train)
#storing y_test in reuired format
ID = dftest['ID']
#y_test= y_test.reshape(len(y_test),1)
ans = pd.concat([ID,pd.DataFrame(y_pred)],axis=1)
ans=ans.astype('int32')
ans[0].value_counts()
ans.info
#store in csv
ans.to_csv("submit.csv",index=None,header=["ID","Class"])
from IPython.display import HTML
import base64

def create_download_link(data_orig, title = "Download CSV file", filename = "data.csv"): 
    csv = data_orig.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html =  '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(ans)
