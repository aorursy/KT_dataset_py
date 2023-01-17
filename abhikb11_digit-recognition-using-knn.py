import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
x=train.drop('label',1)
y=train['label']
clf= KNeighborsClassifier(n_jobs=-1)
clf.fit(x,y)
y=clf.predict(test)
ans=pd.DataFrame(y,columns=['label'])
ans.to_csv('ans.csv')