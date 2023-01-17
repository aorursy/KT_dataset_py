import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns 

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score,roc_auc_score,f1_score)
df=pd.read_csv("../input/diabetes.csv")



X=df.iloc[:,0:8]

y=df[['Outcome']]



plt.style.use('ggplot')





knn = KNeighborsClassifier()



size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

results = {}

for i in size: 

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i)

    knn.fit(X_train,y_train)

    

    test=knn.score(X_test,y_test)

    train=knn.score(X_train,y_train)

    

    results[i]=train,test



size_df=pd.DataFrame(results).transpose()

size_df.columns=['Train accuracy','Test accuracy']



size_df.plot()
plt.style.use('ggplot')





k = list(range(1,50))



accuracy={}

auc={}

f1={}



for i in k: 

    

    knn = KNeighborsClassifier(n_neighbors=i)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    

    knn.fit(X_train,y_train)

    

    acc_tr=accuracy_score(knn.predict(X_train),y_train)

    acc_tst=accuracy_score(knn.predict(X_test),y_test)

    

    

    auc_tr=roc_auc_score(knn.predict(X_train),y_train)

    auc_tst=roc_auc_score(knn.predict(X_test),y_test)

    

    f1_tr=f1_score(knn.predict(X_train),y_train)

    f1_tst=f1_score(knn.predict(X_test),y_test)



    accuracy[i]=acc_tr,acc_tst

    auc[i]=auc_tr,auc_tst

    f1[i]=f1_tr,f1_tst

    

    

pd.DataFrame(accuracy).transpose().plot().set_ylabel('Accuracy')

pd.DataFrame(auc).transpose().plot().set_ylabel('AUC')

pd.DataFrame(f1).transpose().plot().set_ylabel('F1 Score')