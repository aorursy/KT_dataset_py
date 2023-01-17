import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# columns of dataset
columns = ["height","lenght","area","eccen","p_black","p_and","mean_tr","blackpix","blackand","wb_trans","class"]

df = pd.read_csv("../input/pageblocks/page-blocks.data" ,sep="\s+" ,
                 names=columns,
                 header=None,
                )
df.shape  # dimension of dataset
df.head()
df.info()
df.isna().sum() #no missing value is present
df.describe().T  #statistical analysis
df['class'].plot(kind='hist') # highly imbalanced dataset
i=1
plt.figure(figsize=(20,10))

for col in df.columns:
    plt.subplot(3,4,i)
    plt.hist(df[col],bins=50)
    plt.tight_layout()
    plt.title(col,fontsize=15)
    i+=1
i=1
plt.figure(figsize=(20,10))

for col in df.drop(columns='class').columns:
    plt.subplot(3,4,i)
    plt.scatter(df['class'],df[col])
    plt.tight_layout()
    plt.title(col,fontsize=15)
    i+=1
i=1
plt.figure(figsize=(20,10))

for col in df.columns:
    plt.subplot(3,4,i)
    plt.boxplot(df[col])
    plt.tight_layout()
    plt.title(col,fontsize=15)
    i+=1
df = df[df['height']<250] 
df = df[df['area']<35000]
df = df[df['eccen']<300]
df = df[df['mean_tr']<4000]
df = df[df['blackand']<30000]
df = df[df['wb_trans']<2000]
i=1
plt.figure(figsize=(20,10))

for col in df.columns:
    plt.subplot(3,4,i)
    plt.boxplot(df[col])
    plt.tight_layout()
    plt.title(col,fontsize=15)
    i+=1
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
df = shuffle(df)
X = df.drop(columns=['class'])
Y = df['class']
X= (X-X.mean())/X.std()
GBN = GaussianNB()
scores = pd.DataFrame(columns=['MIN','MAX','AVG']) #dataframe for storing scores
score5 = cross_val_score(GBN,X,Y,cv=5,verbose=3)
print("MIN - " , score5.min())
print("AVG - " , score5.mean())
print("MAX - " , score5.max())
score10 = cross_val_score(GBN,X,Y,cv=10,verbose=3)
print("MIN - " , score10.min())
print("AVG - " , score10.mean())
print("MAX - " , score10.max())
for i in range(1,20):
    score = cross_val_score(GBN,X,Y,cv=i+1)
    scores.loc[i+1] = [score.min() , score.max() , score.mean()]
plt.figure(figsize=(20,5))

plt.plot(scores['MIN'],marker='o')
plt.plot(scores['MAX'],marker='o')
plt.plot(scores['AVG'],marker='o')

plt.xticks(np.arange(1, 22, 1.0))
plt.legend(['MIN','MAX','AVG'],fontsize=15)
plt.title("Accuracy Vs N_Split" ,fontsize=20)
scores2 = pd.DataFrame(columns=['MIN','MAX','AVG']) #dataframe for storing scores
#calculate the cross validation score on different split point in stratified k-fold and store into dataframe
for i in range(1,20):
    cv = StratifiedKFold(n_splits=i+1,shuffle=True)
    score = cross_val_score(GBN,X,Y,cv=cv)               #Crossvalidation
    scores2.loc[i+1] = [score.min() , score.max() , score.mean()]
plt.figure(figsize=(20,5))

plt.plot(scores2['MIN'],marker='o')
plt.plot(scores2['MAX'],marker='o')
plt.plot(scores2['AVG'],marker='o')

plt.xticks(np.arange(1, 22, 1.0))
plt.legend(['MIN','MAX','AVG'],fontsize=15)
plt.title("Accuracy Vs N_Split" ,fontsize=20)