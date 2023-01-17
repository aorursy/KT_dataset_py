# importing common libraries...

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data_train = pd.read_csv("../input/train.csv")  # loading the data
data_train.info()  # checking for data types
print(list(data_train.any().isnull()))   # there is no null value in our columns, which is great
data_train.describe()   # I see that in most of the cases values are distributed between -1.00 and 1.00  ...
data_train.head(20)
data_train.drop(["rn"],axis=1,inplace=True)    ## removing the rn column.
data_train.head(10)  ## As seen, problem solved!! :)
# importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30,20))
sns.heatmap(data=(data_train.corr()*data_train.corr()),cmap="BuPu",vmin=0.4)
plt.show()

# what I do here is: only showing correlation x on => ((x^2)>0.4).  By this we only see highly correlated columns. and seems like there are many of them.
# Let's see our labels.
labels = list(data_train.activity.unique())
print(labels) 
# they are strings, we should make them numerical values.  for this i will use scikit-learn

# importing and setting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# fitting each possible label
le.fit(labels)
# updating our data with LabelEncoder
data_train.activity = le.transform(data_train.activity)
y_train = data_train.activity.values.reshape(-1,1)  # scikit doesn't likes when it is like (n,). it rathers (n,m).. that's why I used reshape
x_train = data_train.values 
from sklearn.model_selection import train_test_split 
x_tr,x_tst,y_tr,y_tst = train_test_split(x_train,y_train,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1,algorithm="auto")

knn_model.fit(x_tr,y_tr.ravel())

y_head = knn_model.predict(x_tst)
knn_model.score(x_tst,y_tst)
n = range(1,30)
results = []
for i in n:
    #print(i)
    knn_tester = KNeighborsClassifier(n_neighbors=i)
    knn_tester.fit(x_tr,y_tr.ravel())
    results.append(knn_tester.score(x_tst,y_tst))
plt.clf()

plt.suptitle("SCORES",fontsize=18)

plt.figure(figsize=(20,10))
plt.plot(n,results,c="red",linewidth=4)
plt.xlabel("n neighbors")
plt.ylabel("score")
plt.show()
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_pred=y_head,y_true=y_tst)
conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(20,20))
sns.heatmap(conf,annot=True,cmap="summer")
plt.show()

## I am currently improving my skills on data science. If you have any advice or comment, make sure you show it.  Best Regards.

