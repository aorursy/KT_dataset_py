import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_wine   # load the data from sklearn

wine=load_wine()  # store the data in a variable 
dir(wine)  # check all the directories of wine 
wine.feature_names   
df=pd.DataFrame(wine.data,columns=wine.feature_names)  # show the data in dataframe
df.head()
wine.target
df['target']=wine.target   # add a cloumn target to store the target from wine
df.head()
wine.target_names      # check how many classes of wines are there 
df['target_names']=df.target.apply(lambda x:wine.target_names[x])# add a cloumn target_names to store the target_names from

df
df.describe()
sns.distplot(df['alcohol'])     # distrbution of alcohol content among all of the wines
# distribution of alcohol content by class

for i in df.target.unique():

    sns.distplot(df['alcohol'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()     
plt.figure(1)                       # distribution of malic_acid content by class

for i in df.target.unique():

    sns.distplot(df['malic_acid'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(2)                    # distribution of ash content by class

for i in df.target.unique():

    sns.distplot(df['ash'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(3)                     # distribution of alcalinity_of_ash content by class

for i in df.target.unique():

    sns.distplot(df['alcalinity_of_ash'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(4)                  # distribution of magnesium content by class

for i in df.target.unique():

    sns.distplot(df['magnesium'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(5)                  # distribution of total_phenols content by class

for i in df.target.unique():

    sns.distplot(df['total_phenols'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(6)                 # distribution of flavanoids content by class

for i in df.target.unique():

    sns.distplot(df['flavanoids'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(7)                # distribution of nonflavanoids_phenols content by class

for i in df.target.unique():

    sns.distplot(df['nonflavanoid_phenols'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(8)               # distribution of proanthocyanins content by class

for i in df.target.unique():

    sns.distplot(df['proanthocyanins'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(9)                # distribution of color_intensity content by class

for i in df.target.unique():

    sns.distplot(df['color_intensity'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(10)              # distribution of hue content by class

for i in df.target.unique():

    sns.distplot(df['hue'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(11)             # distribution of od280/od315_of_diluted_wines content by class

for i in df.target.unique():

    sns.distplot(df['od280/od315_of_diluted_wines'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



plt.figure(12)            # distribution of proline content by class

for i in df.target.unique():

    sns.distplot(df['proline'][df.target==i],

                 kde=1,label='{}'.format(i))



plt.legend()



from sklearn.model_selection import train_test_split
x=df.drop(['target','target_names'],axis='columns') 
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) # we use 20% data for testing
len(x_train)
len(x_test)
len(y_train)
len(y_test)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'),x,y,cv=3)
cross_val_score(SVC(gamma='auto'),x,y,cv=3)
cross_val_score(RandomForestClassifier(n_estimators=40),x,y,cv=3)
cross_val_score(DecisionTreeClassifier(max_leaf_nodes=40),x,y,cv=3)
rfc=RandomForestClassifier(n_estimators=40)   # use RandomForestClassifier for prediction and store it in a valiable
rfc.fit(x_train,y_train)      # fit the data 
rfc.predict(x_test)     # pridict the test data
rfc.predict([[14.23,1.71,2.43,15.6,127.0,2.80,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]]) # pridict the values
rfc.score(x_test,y_test)     # check the accuracy 
y_predicted=rfc.predict(x_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predicted)

cm
plt.figure(figsize=(10,6))

sns.heatmap(cm,annot=True)

plt.xlabel('predicted')

plt.ylabel('Actual')