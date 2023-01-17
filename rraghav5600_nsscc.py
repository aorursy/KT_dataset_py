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
from collections import Counter
pd.set_option('display.max_columns', None)

df = pd.read_csv('/kaggle/input/Cross Sell.csv')

df.drop([32264], inplace=True)

df = df.replace('.', np.nan)
def get_mapping(ddf, col):

    temp=dict()

    t=0

    for i in ddf[col].unique():

        temp[i]=t

        t=t+1

    return temp



for col in ['Branch', 'Res']:

    ddict = get_mapping(df, col)

    df[col] = df[col].map(ddict).astype(int)

df.info()
temp = []

for col in df:

    tt = Counter(df[col].isnull())[1]

    if tt>0: temp.append([col,tt,tt/32264])

temp = sorted(temp, key=lambda x:x[1], reverse=True)

temp
draw = np.random.choice([3,4], 1, p=[1,0])[0]

list(Counter(df['Inv'].dropna()).keys()),list(Counter(df['Inv'].dropna()).values()),Counter(df['Inv'].dropna())
for col in 'HMOwn Phone POS CC CCPurc Inv'.split():

    sample = list(Counter(df[col].dropna()).keys())

    prob = list(Counter(df[col].dropna()).values())

    prob = [i/sum(prob) for i in prob]

    for i in range(len(df)):

        if df.at[i,col]==np.nan:

            df.at[i,col]=np.random.choice(sample, 1, prob)[0]
for col in 'HMOwn Phone POS CC CCPurc Inv'.split():

    print(Counter(df[col]))
remove = 'ATM Inv InvBal Res Branch MTG DDA Sav ATM CD DDA NSF IRA LOC Moved CC SDB NSFAmt MM MMCred ILS IRABal'

df.drop(remove.split(), axis=1, inplace=True)
df
for col in df:

    df[col] = pd.to_numeric(df[col])
for col in df:

    tt = len(df[col].unique())

    if tt <=10: df[col] = df[col].fillna(df[col].dropna().mode()[0])

    else: df[col] = df[col].fillna(df[col].dropna().mean())
for col in df:

    df[col] = [int(i) for i in df[col]]
for col in 'Age HMVal'.split():

    df[col] = [i//10 for i in df[col]]

df['CRScore'] = [i//100 for i in df['CRScore']]
df = df.sample(frac=1)



from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split



X = df.drop(['Ins'], axis=1)

y = df['Ins']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25)

model_names, acc, err = [],[],[]



from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier



rf = RandomForestClassifier(random_state=1)

xgb = XGBClassifier()

lr = LogisticRegression(solver='saga')

mlp = MLPClassifier()

dtc = DecisionTreeClassifier(random_state=0)



model_names = ['rf', 'xgb', 'lr', 'mlp', 'dtc']

models = [rf, xgb, lr, mlp, dtc]
for i in range(len(models)):

    models[i].fit(train_X, train_y)

    prediction = models[i].predict(val_X)

    acc.append(accuracy_score(val_y, prediction))

    err.append(mean_absolute_error(val_y, prediction))

    print(accuracy_score(val_y, prediction), mean_absolute_error(val_y, prediction))
models = pd.DataFrame({

    'Model': [i.upper() for i in model_names],

    'Accuracy': acc,

    'Mean Abs Error': err})

models.sort_values(by='Accuracy', ascending=False)