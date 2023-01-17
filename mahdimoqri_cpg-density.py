import pandas as pd 

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from scipy import signal



train='/kaggle/input/predict-dna-methylation/train.csv'

df=pd.read_csv(train,index_col=0)

df.head(1)
m=500

df['CG']=df.seq.apply(lambda x:x[1000-m:1000+m].count('CG'))

df['TG']=df.seq.apply(lambda x:x[1000-m:1000+m].count('TG'))

df['CA']=df.seq.apply(lambda x:x[1000-m:1000+m].count('CA'))

df=pd.get_dummies(df,columns=['Regulatory_Feature_Group'])

df.head(1)
X=df[['CG','Beta','TG','CA','Regulatory_Feature_Group_Promoter_Associated']]

X['mutation']=(X.TG+X.CA)/(2*X.CG)

y=X.Beta

X=X[['mutation','Regulatory_Feature_Group_Promoter_Associated']]

clf = LogisticRegression(random_state=0).fit(X, y)

y=y.values

prob=clf.predict_proba(X)

print('AUC: '+str(roc_auc_score(y,prob[:,1])))

pred=clf.predict(X)

print(classification_report(y, pred))
test='/kaggle/input/predict-dna-methylation/test.csv'

df=pd.read_csv(test,index_col=0)

df['CG']=df.seq.apply(lambda x:x[500:1500].count('CG'))

df['TG']=df.seq.apply(lambda x:x[500:1500].count('TG'))

df['CA']=df.seq.apply(lambda x:x[500:1500].count('CA'))

df['mutation']=(df.TG+df.CA)/(2*df.CG)

df=pd.get_dummies(df,columns=['Regulatory_Feature_Group'])

X=df[['mutation','Regulatory_Feature_Group_Promoter_Associated']]

pred=clf.predict(X)

df['Beta']=pred

df[['Beta']].to_csv('solution.csv')

df.head(1)