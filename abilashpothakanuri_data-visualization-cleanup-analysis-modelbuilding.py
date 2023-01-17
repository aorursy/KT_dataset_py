# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
Pokemon=pd.read_csv("/kaggle/input/pokemon/Pokemon.csv")
Pokemon.columns
Pokemon.shape
Pokemon.head()
D=pd.DataFrame({"Dtype":Pokemon.dtypes,"Null":Pokemon.isnull().sum(),"Percentage NUlls":Pokemon.isnull().sum()/len(Pokemon)*100,"Uniques":Pokemon.nunique()})
D
Pokemon=Pokemon.drop("#",axis=1)
Pokemon['Type 1'].value_counts(normalize=True)
Pokemon['Type 1'].value_counts().plot(kind="bar",color="green")
Pokemon['Type 2'].value_counts(normalize=True)
Pokemon['Type 2'].value_counts().plot(kind="bar",color="green")
sns.distplot(Pokemon['Attack'])
sns.distplot(Pokemon['Defense'])
Pokemon['Generation'].value_counts().plot(kind="bar")
Pokemon['Legendary'].value_counts(normalize=True)
Pokemon['Legendary'].value_counts()
Pokemon['Legendary'].value_counts().plot(kind="bar")
Pokemon[Pokemon['Legendary']==True]["Name"].head()
w=WordCloud()
legend=" "
for i in Pokemon[Pokemon['Legendary']==True]["Name"]:
  legend=legend+" "+i
plt.figure(figsize=(15,7))
plt.grid(False)
plt.imshow(w.generate(legend),interpolation='bilinear',)
plt.figure(figsize=(15,7))
sns.distplot(Pokemon['HP'])
Pokemon['HP'].describe()
sns.catplot(x="Legendary",y="Type 1",data=Pokemon,kind="bar",legend=True)
Ct=pd.crosstab(Pokemon['Type 1'],Pokemon['Legendary'])
Ct.div(Ct.sum(1),axis=0).plot.bar(stacked=True)
corr=Pokemon.corr()
corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="YlGnBu")
def fill_type2(x):
  return Pokemon[Pokemon["Type 1"]==x["Type 1"]]["Type 2"].mode()[0]
Pokemon['Type 2']=Pokemon.apply(lambda x:fill_type2 if pd.isnull(x["Type 2"]) else x["Type 2"],axis=1)
new_pokemon=pd.get_dummies(Pokemon.drop("Legendary",axis=1))
Maxmin=MinMaxScaler()
Pokemon_scaled=Maxmin.fit_transform(new_pokemon)
new_pokemon_scaled=pd.DataFrame(Pokemon_scaled,columns=new_pokemon.columns)
x=new_pokemon_scaled
y=Pokemon['Legendary'].replace({True:1,False:0})
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,train_size=0.9,shuffle=True,random_state=92)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
def model_build(model,x_train,y_train,x_test,y_test):
   k=KFold(shuffle=True,random_state=94,n_splits=4)
   a=1
   for i,j  in k.split(x_train,y_train):
               x_trainn,x_val=  x_train.iloc[i],x_train.iloc[j]
               y_trainn,y_val=  y_train.iloc[i],y_train.iloc[j]

               model.fit(x_trainn,y_trainn)
               train_scores=model.predict(x_trainn)
               test_scores=model.predict(x_val)
               val_scores=model.predict(x_test)
               print("{}.Train f1 score is {} and test f1_score is {} validation score {}".format(a,f1_score(train_scores,y_trainn),f1_score(y_val,test_scores),f1_score(y_test,val_scores)))
           

               a=a+1
   test=model.predict(x_test)
   trains=model.predict(x_train)
   return model,test,trains
X=RandomForestClassifier(max_features=0.5,random_state=88,n_estimators=500,max_depth=100)
X,test_predictions,train_predictions=model_build(X,x_train,y_train,x_test,y_test)
L=KNeighborsClassifier(n_neighbors=1)
L,l_test_score,l_train_scores=model_build(L,x_train,y_train,x_test,y_test)
A=AdaBoostClassifier(n_estimators=400,learning_rate=0.7)
A,a_test_score,a_train_scores=model_build(A,x_train,y_train,x_test,y_test)
accuracy=pd.DataFrame({"test":[f1_score(a_test_score,y_test),f1_score(test_predictions,y_test),f1_score(l_test_score,y_test)],"train":[f1_score(a_train_scores,y_train),f1_score(train_predictions,y_train),f1_score(l_train_scores,y_train)]},index=["Ada","random","Neighbors"])
accuracy.plot(kind="bar")
x=new_pokemon_scaled[ranks['columns'][:500]]
y=Pokemon['Legendary'].replace({True:1,False:0})
final=[]
for i in range(len(l_test_score)):
      final.append(mode([test_predictions[i],l_test_score[i],a_test_score[i]]))
f1_score(final,y_test)