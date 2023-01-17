import pandas as pd

import numpy as np

train=pd.read_csv('../input/whiskey-reviews-ds15/train.csv')

test=pd.read_csv('../input/whiskey-reviews-ds15/test.csv')
train.head(3)
import seaborn as sns
sns.pairplot(train)
sns.lineplot(x=train['id'], y=train['ratingCategory']);
l1,l2,l0=[],[],[]

for i in range(len(train['ratingCategory'])):

  if(train['ratingCategory'][i]==2):

    l2.append(train['id'][i])

  elif (train['ratingCategory'][i]==1):

    l1.append(train['id'][i])

  else:

    l0.append(train['id'][i])  

    
print('for rating 1 -->',min(l1),max(l1))

print('for rating 2->',min(l2),max(l2))
def idf(lst):

  a=[]

  for i in lst:

    if(i>126 and i<3713):

      a.append(1)

    elif (i>5030):

      a.append(2)

    else:

      a.append(0)

  return a        
submission=pd.read_csv('../input/whiskey-reviews-ds15/sample_submission.csv')
submission['ratingCategory']=idf(test['id'])
