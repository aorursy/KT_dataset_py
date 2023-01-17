import pandas as pd 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

train='/kaggle/input/predict-dna-methylation/train.csv'

df=pd.read_csv(train,index_col=0)

df.head(1)
X = pd.get_dummies(df['Relation_to_UCSC_CpG_Island'])

X['Beta']=df.Beta

X.head(1)
X['Beta']=df.Beta

X.groupby('Island').mean()['Beta']
X=X.drop('Beta',1)

y=df.Beta

clf = LogisticRegression(random_state=0).fit(X, y)

y=y.values

prob=clf.predict_proba(X)

print('AUC: '+str(roc_auc_score(y,prob[:,1])))

pred=clf.predict(X)

print(classification_report(y, pred))
test='/kaggle/input/predict-dna-methylation/test.csv'

df=pd.read_csv(test,index_col=0)

df.head(1)
X = pd.get_dummies(df['Relation_to_UCSC_CpG_Island'])

X.head(1)
pred=clf.predict(X)

df['Beta']=pred

df[['Beta']].to_csv('solution.csv')

df.head(1)