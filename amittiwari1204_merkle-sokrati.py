# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_excel('../input/facebook-google-clicks/morkel.xlsx')
df.info()
for i in df:
    print(df[i].value_counts())
df.drop(['product','phase','campaign_type','communication_medium'], axis=1, inplace=True)
df.head()
df['Month'] = df['Date'].dt.month
df['Day of the month'] = df['Date'].dt.day
df["Day of the week"] = df['Date'].dt.dayofweek
df['Year'] = df['Date'].dt.year
df = df.drop(['Date'], axis=1)
from sklearn.preprocessing import LabelEncoder as LE
le=LE()
l=['campaign_platform','subchannel','audience_type','creative_type','creative_name','device','age']
for i in l:
    df[i]=le.fit_transform(df[i])
df.head()
for i in df.iterrows():
    if pd.isnull(i[1]['link_clicks']) and i[1]['clicks']==0:
        df.iloc[i[0],10]=0
df.dropna(axis=0, inplace=True)
from sklearn.model_selection import train_test_split

X = df[['campaign_platform','subchannel','audience_type','creative_type','creative_name','device','age','spends','impressions','clicks','Month','Day of the month','Day of the week','Year']]
y = df['link_clicks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model_1 = LogisticRegression(solver='lbfgs',max_iter=100)
model_1.fit(X_train, y_train)
predictions_LR = model_1.predict(X_test)

print('Logistic regression accuracy:', accuracy_score(predictions_LR, y_test))
#print('')
#print('Confusion matrix:')
#print(confusion_matrix(y_test,predictions_LR))
print(confusion_matrix(y_test,predictions_LR))