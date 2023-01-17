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
data_orig = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv", sep=',')
data = data_orig
data.head()
data.info()
data.duplicated().sum()
for cols in data.columns:
    if str(data.dtypes[cols]) == 'object':
        print(cols)
print(data['col2'].unique())
print(data['col11'].unique())
print(data['col37'].unique())
print(data['col44'].unique())
print(data['col56'].unique())
edata = pd.get_dummies(data, columns=['col2', 'col11', 'col37', 'col44', 'col56'])
edata.head()
import seaborn as sns
f, ax = plt.subplots(figsize=(100, 100))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 100, as_cmap=True),
            square=True, ax=ax, annot = True);
edata = edata.drop(['ID'], 1)
edata.head()
Y = edata['Class']
X = edata.drop(['Class'], axis=1)
X.head()
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X_N = pd.DataFrame(np_scaled)
X_N.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, Y, test_size=0.20, random_state=42)
X_train.head()
best = [3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3,
        3, 3, 3, 2, 2, 3, 2, 3, 2, 0, 3, 2, 3, 0, 0, 2, 2, 3, 2, 3, 0, 3,
        3, 3, 3, 3, 3, 2, 0, 3, 0, 2, 3, 2, 0, 3, 3, 3, 3, 0, 2, 3, 0, 3,
        3, 2, 2, 3, 3, 0, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0,
        0, 3, 3, 3, 3, 3, 2, 3, 2, 0, 3, 3, 0, 0, 0, 3, 2, 3, 3, 3, 3, 3,
        3, 2, 0, 3, 0, 2, 3, 3, 3, 2, 3, 2, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3,
        2, 3, 3, 0, 3, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 2, 2, 0, 3, 3, 3, 3,
        0, 0, 3, 2, 3, 0, 3, 3, 3, 3, 3, 0, 3, 0, 2, 3, 0, 2, 0, 3, 3, 3,
        3, 3, 3, 3, 2, 0, 3, 3, 3, 2, 3, 3, 0, 3, 0, 3, 3, 2, 2, 0, 3, 2,
        3, 0, 0, 2, 2, 3, 3, 2, 3, 3, 3, 3, 0, 2, 3, 3, 3, 2, 2, 3, 0, 2,
        2, 3, 2, 3, 2, 3, 0, 3, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 0, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 2, 0, 0, 2, 2, 3,
        3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3,
        0, 2, 0, 2, 3, 3, 3, 2, 2, 2, 3, 3, 0, 2]

def CheckWithTheBest(best, op):
    l1 = len(best)
    l2 = len(op)
    if l1 != l2:
        print('Not matching outputs')
        return 0
    match = 0
    for i in range(l1):
        if best[i] == op[i]:
            match += 1
    return match/l1
tdata = pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv', sep=',')
tdata.head()
for cols in tdata.columns:
    if str(tdata.dtypes[cols]) == 'object':
        print(cols)
tdata = pd.get_dummies(tdata, columns=['col2', 'col11', 'col37', 'col44', 'col56'])
tdata.head()
tdata = tdata.drop(['ID'], 1)
tdata.head()
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(tdata)
tX_N = pd.DataFrame(np_scaled)
tX_N.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# rf = RandomForestClassifier(n_estimators = 200, max_depth=18)
# rf.fit(X_train, y_train)
# predictedVal = rf.predict(tX_N)


labelAssigned = [False for i in range(300)]
finalPrediction = [4 for i in range(300)]

#First classifying 0
cpyFor0 = Y.copy()
cpyFor0 = cpyFor0.replace(to_replace=[3,1,2], value = 4)

rf = RandomForestClassifier(n_estimators = 100, max_depth=15)
rf.fit(X_N, cpyFor0)
pred0 = rf.predict(tX_N)

for i in range(300):
    if pred0[i] == 0 and not labelAssigned[i]:
        finalPrediction[i] = 0
        labelAssigned[i] = True

#Now Classifying 1
cpyFor1 = Y.copy()
cpyFor1 = cpyFor1.replace(to_replace=[0,2,3], value = 4)

rf = RandomForestClassifier(n_estimators = 100, max_depth=15)
rf.fit(X_N, cpyFor1)
pred1 = rf.predict(tX_N)

for i in range(300):
    if pred1[i] == 1 and not labelAssigned[i]:
        finalPrediction[i] = 1
        labelAssigned[i] = True


#Now Classifying 2
cpyFor2 = Y.copy()
cpyFor2 = cpyFor2.replace(to_replace=[0,1,3], value = 4)

rf = RandomForestClassifier(n_estimators = 100, max_depth=15)
rf.fit(X_N, cpyFor2)
pred2 = rf.predict(tX_N)

for i in range(300):
    if pred2[i] == 2 and not labelAssigned[i]:
        finalPrediction[i] = 2
        labelAssigned[i] = True

# #0(25) 1(5) 2(43) 3(9)     
#Now Classifying 3
cpyFor3 = Y.copy()
cpyFor3 = cpyFor3.replace(to_replace=[0,1,2], value = 4)

rf = RandomForestClassifier(n_estimators = 100, max_depth=15)
rf.fit(X_N, cpyFor3)
pred3 = rf.predict(tX_N)

for i in range(300):
    if pred3[i] == 3 and not labelAssigned[i]:
        finalPrediction[i] = 3
        labelAssigned[i] = True

        
rf = RandomForestClassifier(n_estimators = 100, max_depth=15)
rf.fit(X_N, Y)
predVal = rf.predict(tX_N)        
# knn = KNeighborsClassifier(n_neighbors=57)
# knn.fit(X_N, Y)
# predVal = knn.predict(tX_N)

for i in range(300):
    if not labelAssigned[i]:
        finalPrediction[i] = predVal[i]
        
        
freq = {0:0, 1:0, 2:0, 3:0, 4:0}
for i in finalPrediction:
    freq[i] += 1
print(freq)
predictedVal = finalPrediction
print(predictedVal)
CheckWithTheBest(best, finalPrediction)
nd = pd.DataFrame(columns=['ID', 'Class'])
idR = np.array([i for i in range(700,1000)])
classR = predictedVal
nd['ID'] = idR
nd['Class'] = predictedVal
nd.head()
np.array(predictedVal)
CheckWithTheBest(best, predictedVal)
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
create_download_link(nd)
