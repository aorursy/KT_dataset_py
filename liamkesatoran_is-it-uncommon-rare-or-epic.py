import os



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from random import seed

seed(42)
chest_df = pd.read_csv('../input/world-of-warcraft-items-dataset/chest.csv')

hands_df = pd.read_csv('../input/world-of-warcraft-items-dataset/hands.csv')

feet_df = pd.read_csv('../input/world-of-warcraft-items-dataset/feet.csv')

head_df = pd.read_csv('../input/world-of-warcraft-items-dataset/head.csv')

legs_df = pd.read_csv('../input/world-of-warcraft-items-dataset/legs.csv')



# One hot encoding of body part type, as this is a very important feature

chest_df['chest_type']=1

hands_df['hands_type']=1

feet_df['feet_type']=1

head_df['head_type']=1

legs_df['legs_type']=1





# ### Way to automate this:

# i = 0

# df = pd.DataFrame()

# for file in os.listdir('./world-of-warcraft-items-dataset'):

#     app = pd.read_csv(f'./world-of-warcraft-items-dataset/{file}')

#     app['item_type_'+str(i)]=1

#     df = pd.concat([df,app])

#     i+=1

    
df = pd.concat([chest_df,feet_df,hands_df,head_df,legs_df], sort=True)

df.head()
df.isnull().sum()/df.shape[0]
tooMuchNa = df.columns[df.isnull().sum()/df.shape[0] > 0.98]
df = df.drop(tooMuchNa, axis =1)
df = df.drop(['name_enus','classes','socket1','socket2','socket3'], axis =1)
df = df.rename({'quality':'target'},axis =1)
df = df.dropna(subset=['target'])
df = df.fillna(0)

df
df.max()
df.min()
df = df.drop('itemset', axis=1) 
sns.distplot(df['agi'])
sns.distplot(df['agi'].apply(lambda x: np.log(x+1)))
logNeeded = df.drop('target',axis=1).max()[df.drop('target',axis=1).max() > 500].index

for column in logNeeded:

    df[column]=df[column].apply(lambda x: np.log(x+1))
df.max()
df = df.drop(['agiint','strint'], axis=1)
df.hist(figsize=(20,20))
sns.countplot(df['target'])
df = df[df['target'].isin(['Uncommon', 'Rare', 'Epic'])]

df
#Label Encoding

LE_df = df.replace( {'target': {'Uncommon':0,'Rare':1, 'Epic':2}})
from sklearn.model_selection import train_test_split

X = LE_df.drop('target',axis =1)

y = LE_df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
import xgboost as xgb

from sklearn.metrics import accuracy_score





XGBC = xgb.XGBClassifier(max_depth=10,n_estimators=50)

XGBC.fit(X_train,y_train)

y_pre_xgb= XGBC.predict(X_test)

print('Accuracy : ',accuracy_score(y_test,y_pre_xgb))

fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(XGBC, height=0.5, ax=ax, importance_type='gain')

plt.show()