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

train_file = '/kaggle/input/digit-recognizer/train.csv'

test_file =  '/kaggle/input/digit-recognizer/test.csv'

sample_submission_file = '/kaggle/input/digit-recognizer/sample_submission.csv'

df = pd.read_csv(train_file)

df.head()
df.info()
df.describe()
df['label'].hist(histtype='stepfilled')



df=df.sample(frac=1).reset_index(drop=True)



n=len(df)

sz = 2500



f = np.array(df.values)



labels = f[:,0]

f = f[:,1:]



train_idxs, test_idxs = np.array(range(sz)), np.array(range(sz,int(sz*1.25)))



m = len(df.columns)





labels = np.array(labels)



train_labels = labels[train_idxs].astype('int')

test_labels = labels[test_idxs].astype('int')







def l2_norm(a, b):

    return int(sum((f[a]-f[b])**2))



def predict(dis,test_id, k):



    distances = [dis(test_id, train_id) for train_id in train_idxs]

    distances = np.array(distances)

    

    topk_id = np.argpartition(distances, k)[:k]

  

    topk_id = topk_id[np.argsort(distances[topk_id])]



    topk_labels = train_labels[topk_id] 

    

    t=np.zeros(shape=(k))

    

    for i in range(1,k+1):

        ls=np.bincount(topk_labels[:i]  )

        t[i-1]=np.argmax(ls)

    return t.astype('int')



def work(dis,k):

    pred = []

    for i in tqdm(test_idxs):

        pred.append(predict(dis,i,k))

    

    

    pred = np.vstack(pred)

    

    confusion_matrix=np.zeros(shape=(10,10)).astype('int')



    for j,i in zip(pred[:,9],test_labels):

        confusion_matrix[int(i)][int(j)]+=1

    

    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    plt.xlabel('Predicted label')

    plt.ylabel('True label')

 

    acc=np.zeros(k)

    for _k in range(k):

        TN = TP = FN = FP = 0

        for i,j in zip(pred[:,_k],test_labels):

            if(i==0 and j==0): TN=TN+1

            if(i==0 and j!=0): FN=FN+1

            if(i!=0 and j==0): FP=FP+1

            if(i!=0 and j!=0): TP=TP+1

        acc[_k] = (TP+TN)/(TP+TN+FP+FN)

        

    return acc

K=10

rn = 0

ans = work(l2_norm, K)
x = range(1,K+1)

plt.plot(x,ans,'s-',color = 'r')

plt.xlabel("k")

plt.ylabel("accuracy")

plt.legend(loc = "best")

plt.show()

t1 = time.time()

print(t1-t0, "seconds wall time")
p=0

samples_idxs = np.empty(10).astype('int')

samples_idxs.fill(-1)



for i in range(n):

    if(samples_idxs[labels[i]]==-1):

        samples_idxs[labels[i]]=i

        p+=1

        if(p>=10): break
rows = 2

cols = 5



fig=plt.figure()



for i in range(10):

    subfig = fig.add_subplot(rows, cols,i+1 )

    subfig.set_title("Label " + str(i)) 

    plt.imshow(f[samples_idxs[i]].reshape(28,28))



fig.tight_layout()    

plt.show()
