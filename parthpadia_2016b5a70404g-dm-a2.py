import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df1 = pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')

df2 = pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')

answer = df1['Class']

df1.drop(columns=['Class'],inplace=True)
char1 = df1.describe()

char2 = df2.describe()



for cols in char1:

    if(abs(char1[cols]['mean']-char2[cols]['mean'])>5):

        print(cols, char1[cols]['mean'],char2[cols]['mean'],char1[cols]['std'],char2[cols]['std'])

        

df1.drop(columns=['ID','col2','col11','col37','col44','col56','col6','col19','col30','col59'],inplace=True)

df2.drop(columns=['ID','col2','col11','col37','col44','col56','col6','col19','col30','col59'],inplace=True)
df3 = pd.concat([df1, df2], ignore_index=True, sort=False)



#Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(df3)

dff = pd.DataFrame(x_scaled)
from imblearn.over_sampling import RandomOverSampler





ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(df1, answer)

df1ovs = pd.DataFrame(X_resampled)

answerovs = y_resampled





dffovs = pd.concat([df1ovs, df2], ignore_index=True, sort=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

clf = RandomForestClassifier(bootstrap='True',n_estimators=200,max_depth=90,max_features='auto',

                             min_samples_leaf= 5,min_samples_split=6,random_state=2424)

clf.fit(dff[:700],answer)

a = clf.predict(dff[700:])

ans = pd.DataFrame(a)

ans['ID'] = pd.Series(range(700,1000))

ans['ID'] = ans['ID'].astype('int64')

ans['RandomForest1'] = a

ans.drop(columns=[0],inplace=True)





clf = ExtraTreesClassifier(random_state=1069)

clf.fit(dff[:700],answer)

a = clf.predict(dff[700:])

ans['ExtraT1'] = a



clf = RandomForestClassifier(bootstrap='False',n_estimators=200,max_depth=100,max_features='auto',

                             min_samples_leaf= 1,min_samples_split=6,random_state=69)

clf.fit(dffovs[:900],answerovs)

a = clf.predict(dffovs[900:])

ans['RandomForest5'] = a



clf = ExtraTreesClassifier(bootstrap='False',n_estimators=400,max_depth=90,max_features='auto',

                             min_samples_leaf= 1,min_samples_split=2,random_state=69)

clf.fit(dffovs[:900],answerovs)

a = clf.predict(dffovs[900:])

ans['ExtraT5'] = a
ans['Class'] = ans[['RandomForest1','RandomForest5','ExtraT1',

                    'ExtraT5']].mode(axis='columns', numeric_only=True)[1]

ans['Class'].fillna(ans[['RandomForest1','RandomForest5','ExtraT1',

                    'ExtraT5']].mode(axis='columns', numeric_only=True)[0],inplace=True)

ans['Class'] = ans['Class'].astype('int64')
final = ans[['ID','Class']]
final['Class'].value_counts()
final.to_csv('final_sub.csv',index = False)
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

create_download_link(final)