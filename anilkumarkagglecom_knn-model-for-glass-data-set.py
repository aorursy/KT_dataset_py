import pandas as pd
import numpy as np
import matplotlib,pylab as plt 
g=pd.read_csv("../input/glass/glass.csv")
g.head()
df=pd.DataFrame(g)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
train,test = train_test_split(df,test_size = 0.2)
neigh = KNC(n_neighbors= 2)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
test_acc

acc = []
for i in range(2,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])

plt.plot(np.arange(2,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(2,50,2),[i[1] for i in acc],"bo-")

