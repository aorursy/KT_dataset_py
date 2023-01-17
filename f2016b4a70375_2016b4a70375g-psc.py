# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv",sep = ',',index_col = 'ID')
test_data = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv",sep = ',',index_col = 'ID')
test_data
train_data.describe()
df1= train_data.iloc[:,:-1]
df2 = test_data
df2
obj_cols = []
for i in range(0,64):
  col = "col" + str(i);
  if(df1[col].dtypes == object):
    obj_cols.append(col)

obj_cols
df_onehot1 = pd.get_dummies(df1,columns=obj_cols,prefix=obj_cols)
df_onehot2 = pd.get_dummies(df2,columns=obj_cols,prefix=obj_cols)
df_onehot2
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_data1=scaler.fit_transform(df_onehot1)
scaled_df1=pd.DataFrame(scaled_data1,columns=df_onehot1.columns,index = df1.index)
scaled_data2=scaler.fit_transform(df_onehot2)
scaled_df2=pd.DataFrame(scaled_data2,columns=df_onehot2.columns,index = df2.index)
scaled_df2
train_df = pd.DataFrame(scaled_df1,index = scaled_df1.index)
test_df = pd.DataFrame(scaled_df2,index = scaled_df2.index)
test_df
from sklearn.model_selection import train_test_split

X_train1, X_val, Y_train1, Y_val = train_test_split(train_df, train_data['Class'], test_size=0.20, random_state=55)
X_train1
from sklearn.ensemble import ExtraTreesClassifier

score_train = []
score_val = []

for i in range(3,20,1):   
    clf = ExtraTreesClassifier(n_estimators=400,max_depth = i,min_samples_leaf = 6,random_state=55)
    clf.fit(X_train1, Y_train1)
    sc_train = clf.score(X_train1,Y_train1)
    score_train.append(sc_train)
    sc_val = clf.score(X_val,Y_val)
    score_val.append(sc_val)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(3,20,1),score_train,color='blue', linestyle='dashed', marker='o',
        markerfacecolor='green', markersize=5)
val_score,=plt.plot(range(3,20,1),score_val,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [score_train,score_val],["Train Score","Cross-Validation Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
clf = ExtraTreesClassifier(n_estimators=400,max_depth = 11,min_samples_split = 6,random_state = 55)
clf.fit(X_train1,Y_train1)
clf.score(X_val,Y_val)
clf.fit(train_df, train_data['Class'])
y_pred = clf.predict(test_df)
y_pred
res1 = pd.DataFrame(y_pred.astype(int),columns=['Class'])
res1.set_index(scaled_df2.index,inplace=True)
res1
res1.to_csv('sub11.csv')
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}<\a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(res1)
