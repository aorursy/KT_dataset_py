# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.shape
df.columns
df.isnull().values.any()
df["Amount"].head()
df["Class"].value_counts()


import seaborn as sns

sns.countplot(df['Class'])

corr = df.corr()
plt.figure(figsize =(20,20))
sns.heatmap(corr,annot = True,linecolor = 'black')

X1 = df.iloc[:,0:30]
Y1 = df.iloc[:,-1]

from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state = 42)

X1_res,y1_res = smk.fit_sample(X1,Y1)
print(X1_res.shape,y1_res.shape)
df1 = X1_res
df1.head()
type(y1_res)

#columns = df.columns.tolist()
# Filter the columns to remove data we do not want 
#columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
#target = "Class"
# Define a random state 
#state = np.random.RandomState(42)
#X = df[columns]
#Y = df[target]
# Print the shapes of X & Y
#print(X.shape)
#print(Y.shape)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X1_res,y1_res)
print(model.feature_importances_)

bestfeatures = pd.Series(model.feature_importances_,index = X1_res.columns)
print(bestfeatures)

bestfeatures.nlargest(10).plot(kind = "barh")
plt.show()

final_x_res = df1[["V18","V9","V16","V17","V3","V10","V11","V4","V12","V14"]]



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(final_x_res,y1_res,test_size  =0.3,random_state = 200)





from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators =10 ,criterion = "gini")
model = rfc.fit(x_train,y_train)

#prediction

pred = model.predict(x_test)

from sklearn.metrics import classification_report

print( classification_report(y_test,pred))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred))
from sklearn.metrics import accuracy_score

acc_RFC = accuracy_score(y_test,pred)


from sklearn.model_selection import cross_val_score

score = cross_val_score(rfc,final_x_res,y1_res,cv = 5)
print(score)
print(score.mean())
from sklearn.model_selection import StratifiedKFold


accuracy = []
skf =  StratifiedKFold(n_splits = 5,random_state = None)
skf.get_n_splits(final_x_res,y1_res)

for train_index,test_index in skf.split(final_x_res,y1_res):
    print("train:", train_index,"test:", test_index)
    x1_train,x1_test = final_x_res.iloc[train_index],final_x_res.iloc[test_index]
    y1_train,y1_test = y1_res.iloc[train_index],y1_res[test_index]
    
    
    model1 = rfc.fit(x1_train,y1_train)
    pred1 = rfc.predict(x1_test)
    acc = accuracy_score(y1_test,pred1)
    accuracy.append(acc)
    
    
print(accuracy)
acc_skf = np.array(accuracy).mean()
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()


score1 = (cross_val_score(lr,final_x_res,y1_res,cv = 5))
acc_lr = score1.mean()
from sklearn.neighbors import KNeighborsClassifier
kneigh = KNeighborsClassifier(n_neighbors = 50)

#indices,distance = mod2.kneighbors(final_x_res)
x1_train,x1_test,y1_train,y1_test = train_test_split(final_x_res,y1_res,test_size  =0.3,random_state = 200)

mod2 = kneigh.fit(x1_train,y1_train)
mod2predict = kneigh.predict(x1_test)
mod2predict[0:5]

print(classification_report(y1_test,mod2predict))
acc_knn = accuracy_score(y1_test,mod2predict)
print("acc for random forest classifier :", acc_RFC)
print("acc for random forest classifier :" , acc_skf)
print("acc for logistic regression  :" , acc_lr)
print("acc for K nearesr neighbour  :" , acc_knn)

