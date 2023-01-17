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
import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
df=pd.read_csv(r'/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')
df.head()
for col in df.columns:
    print(np.sum(df[col].isnull()))
df[df['NAME']=='NOTA']=df[df['NAME']=='NOTA'].replace(np.nan,'NOTA')
df['ASSETS']=df['ASSETS'].replace('Not Available','0')
df['LIABILITIES']=df['LIABILITIES'].replace('Not Available','0')
df['ASSETS']=df['ASSETS'].replace('Nil','0')
df['LIABILITIES']=df['LIABILITIES'].replace('Nil','0')
df['LIABILITIES']=df['LIABILITIES'].replace('NIL','0')
df['ASSETS']=df['ASSETS'].replace('`','0')
df['LIABILITIES']=df['LIABILITIES'].replace('`','0')

df['ASSETS']=df['ASSETS'].replace('NOTA','0')
df['LIABILITIES']=df['LIABILITIES'].replace('NOTA','0')
df['AGE']=df['AGE'].replace('NOTA',0)
df['CRIMINAL\nCASES']=df['CRIMINAL\nCASES'].replace('NOTA',0)
df['CRIMINAL\nCASES'].replace('Not Available','0',inplace=True)
df['CRIMINAL\nCASES']=df['CRIMINAL\nCASES'].astype('int')


df['ASSETS']=df['ASSETS'].replace(',','',regex=True)
df['LIABILITIES']=df['LIABILITIES'].replace(',','',regex=True)
df['ASSETS']=df['ASSETS'].replace('Nil','0',regex=True)
df['LIABILITIES']=df['LIABILITIES'].replace('Nil','0',regex=True)
df['ASSETS']=df['ASSETS'].str.extract(pat='([0-9]+)')
df['LIABILITIES']=df['LIABILITIES'].str.extract(pat='([0-9]+)') 

df['ASSETS']=df['ASSETS'].astype('float')
df['LIABILITIES']=df['LIABILITIES'].astype('float')

df.dtypes
df['Is_NOTA']=df['NAME']=='NOTA'

X=df.drop(['WINNER','NAME','SYMBOL'],axis=1)
y=df['WINNER']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.33, random_state=42)
from catboost import CatBoostClassifier

clf = CatBoostClassifier(
    iterations=500, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)


clf.fit(X_train, y_train, 
        cat_features= [0,1,2,3,4,6,7], 
         
        verbose=False
        
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
from sklearn.metrics import accuracy_score
y_pred=clf.predict(X_test)
print('ACCURACY SCORE is:' ,accuracy_score(y_pred,y_test))
