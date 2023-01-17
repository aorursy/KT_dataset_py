import time

t0 = time.time()



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



sns.set()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename = '/kaggle/input/pima-indians-diabetes-database/diabetes.csv'

df = pd.read_csv(filename)

df.head()
df.info()
df.describe()
df.hist(histtype='stepfilled',figsize = (20,20))

df2 = df.copy(deep = True)

exist_zero_coloumns=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']



df2[exist_zero_coloumns] = df2[exist_zero_coloumns].replace(0,np.NaN)

df2.describe()
df2.isnull().sum()
for i in exist_zero_coloumns:

    df2[i].fillna(df2[i].median(), inplace = True)

    
df2.hist(figsize = (20,20))
df2=(df2-df2.min())/(df2.max()-df2.min())
df2.describe()
idxs= list(range(len(df2)))



np.random.seed(1000000007)

np.random.shuffle(idxs)



sz = int(len(df2)*0.8)

train_idxs, test_idxs = idxs[:sz], idxs[sz:]



f = df2.values

m = len(df2.columns)

m1=m-1

m2=1



labels = np.array(df2['Outcome']).astype('int')



train_labels = labels[train_idxs].astype('int')

test_labels = labels[test_idxs].astype('int')

def l1_norm(a, b):

    s=0

    for p in range(m1):

            s= s + abs(f[a][p] - f[b][p])

    return s



def l2_norm(a, b):

    s=0

    for p in range(m1):

            s= s + (f[a][p] - f[b][p])**2

    return s



def maximum_norm(a, b):

    s=0

    for p in range(m1):

            s= max(s,abs(f[a][p] - f[b][p]))

    return s





def predict(dis,test_id, k):

    distances = [dis(test_id, train_id) for train_id in train_idxs]

    distances = np.array(distances)

    

    topk_id = np.argpartition(distances, k)[:k]

  

    topk_id = topk_id[np.argsort(distances[topk_id])]



    topk_labels = train_labels[topk_id] 

    

    t=np.zeros(shape=(k))

    

    ls=np.bincount(topk_labels[:k])



    return int(np.argmax(ls))



k = 11

def work(dis,k):

    pred = [predict(dis,i,k) for i in test_idxs]



    TN = TP = FN = FP = 0

    for i,j in zip(pred,test_labels):

        if(i==0 and j==0): TN=TN+1

        if(i==0 and j==1): FN=FN+1

        if(i==1 and j==0): FP=FP+1

        if(i==1 and j==1): TP=TP+1



    confusion_matrix=[[TN,FP],[FN,TP]]



    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    print("accuracy=(TP+TN)/(TP+TN+FP+FN)=" ,(TP+TN)/(TP+TN+FP+FN))
work(l1_norm,k)
work(l2_norm,k)
work(maximum_norm,k)
t1 = time.time()

print(t1-t0, "seconds wall time")