# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# data = pd.read_csv('./dataset/train.csv')
# data2 = pd.read_csv('./dataset/test.csv')
data = pd.read_csv('../input/data-mining-assignment-2/train.csv')
data2 = pd.read_csv('../input/data-mining-assignment-2/test.csv')
## Make all data numeric so its easier to analyse
rep_dict = {'Yes':1,'No':0,
            'Male':0,'Female':1, 
            'Low':0,'Medium':1,'High':2
            ,'Silver':0,'Gold':1,'Platinum':2,'Diamond':3} # Not sure about these ratings
print(data2['col2'].value_counts())   # Just checking for discrepancies

data_numeric = data.replace(rep_dict)
data_numeric = pd.concat([data_numeric,pd.get_dummies(data_numeric['col2'], prefix='categ')],axis=1)
data_numeric = data_numeric.drop('col2',axis=1)

data2_numeric = data2.replace(rep_dict)
data2_numeric = pd.concat([data2_numeric,pd.get_dummies(data2_numeric['col2'], prefix='categ')],axis=1)
data2_numeric = data2_numeric.drop('col2',axis=1)
## Making final sets
x_train = data_numeric.iloc[:,data_numeric.columns!='Class'].values
x_test  = data2_numeric.values
y_train = data_numeric.loc[:,'Class'].values
print(np.unique(y_train,return_counts=True))

x_val = x_train[560:,:]
y_val = y_train[560:]
print(np.unique(y_val,return_counts=True))
from sklearn.ensemble import RandomForestClassifier      
from sklearn.metrics import accuracy_score,f1_score

for i in range(1,20):
    rf1 = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=i)
    rf1.fit(x_train[:560],y_train[:560])
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4,random_state=i)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    print(str(i),end=' ')
    print(f1_score(y_train[560:],rf1.predict(x_train[560:]),average='micro'))
    print(np.unique(pred,return_counts=True))

rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=1)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
finaldict = {'ID':data2['ID'],'Class':pred}
finaldf = pd.DataFrame(finaldict)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="submission.csv" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(finaldf)
## Kindly see the psc to see the full amount of effort I put in :(