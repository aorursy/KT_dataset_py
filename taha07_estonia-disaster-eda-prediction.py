# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import missingno as msno
import pandas_profiling
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading the data
df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
df.head()
df.info()
profile = pandas_profiling.ProfileReport(df)
profile
n = msno.bar(df,color='gold')
plt.rcParams['figure.figsize'] = (12,9)
#plt.style.use('classic')
color = plt.cm.hsv(np.linspace(0,1,20))
df['Country'].value_counts().plot.bar(color=color)
plt.show()
plt.rcParams['figure.figsize'] = (10,8)
#plt.style.use('classic')
color = plt.cm.hsv(np.linspace(0,1,20))
sns.countplot(y = 'Country',palette ='Set3',hue='Survived',orient = 'h',data = df)
plt.show()

plt.rcParams['figure.figsize']=(10,8)
#plt.style.use("classic")
color = ['yellowgreen','gold']
labels =['Not Survived','Survived']
df['Survived'].value_counts().plot.pie(y="Survived",colors=color,explode=(0,0.08),startangle=50,shadow=True,autopct="%0.1f%%")
plt.legend(labels,loc='best')
plt.axis('on');
plt.rcParams['figure.figsize'] =(9,8)
sns.catplot(x="Sex", hue="Category", col="Survived",
                data=df, kind="count",
                height=6, aspect=.7,palette='Set3')
plt.show()
facet = sns.FacetGrid(df,hue="Survived",aspect = 4)
facet.map(sns.kdeplot,"Age",shade = True)
facet.set(xlim = (0,df["Age"].max()))
facet.add_legend()
plt.show()
plt.rcParams['figure.figsize'] = (10,8)
#plt.style.use('classic')
color = plt.cm.hsv(np.linspace(0,1,20))
sns.countplot(y = 'Country',palette ='Set3',hue='Category',orient = 'h',data = df)
plt.show()
df.groupby(['Survived','Sex'])['Sex'].count()
df.groupby(['Survived','Sex'])['Sex'].count().plot();
plt.rcParams['figure.figsize']=(10,8)
plt.style.use("classic")
color = ['yellowgreen','gold','lightskyblue','coral']
df.groupby(['Survived','Sex'])['Sex'].count().plot.pie(colors=color,explode=(0,0.02,0.4,0.01),shadow=True,autopct = '%0.1f%%')
plt.legend(labels,loc='best')
plt.axis('on');
Country = pd.DataFrame(df["Country"].value_counts().reset_index().values,columns=["country","NumOfPassenger"])
Country.head()
import plotly.express as px
fig = px.choropleth(Country,   
    locationmode='country names',
    hover_name="country",
    color='country',
    locations=Country.country,
    hover_data = ['country','NumOfPassenger'],
    featureidkey="Country.NumOfPassenger",
    labels=Country["NumOfPassenger"],
    title= "Country with Number of Passengers"
)
fig.show()

df.head(2)
df.drop(['PassengerId','Firstname','Lastname'],axis=1,inplace=True)
mapping = {'Sweden': 'Sweden','Estonia':'Estonia','Latvia':'Latvia'}
df['Country'] = df['Country'].map(mapping)
df['Country'].fillna('Other',inplace=True)
df['Age'] = df['Age'].apply(lambda x: 1 if (x>=0 and x<= 20)
                                                      else (2 if x>20 and x<=40 else (3 if x>40 and x<=60 else 4)))
df = pd.get_dummies(df,drop_first=True)
from sklearn.model_selection import train_test_split
x = df.drop(['Survived'],axis=1)
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=0)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
model_params  = {
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            'criterion':["entropy","gini"],
            "max_depth":[5,8,9]
        }
    },
    
    "random_forest":{
        "model": RandomForestClassifier(),
        "params":{
            "n_estimators":[1,5,10,15,20,30],#20,30
            "max_depth":[5,8,9,3,2]#3
        }
    },
    
    "adaboost":{
        "model": AdaBoostClassifier(),
        "params":{
            "n_estimators":[1,5,10,15,20,30],
            "learning_rate":[0.01,0.02,0.001,0.03],
            "random_state":[0,100,10]
        }
    }
    
}
score=[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=8,return_train_score=False)
    clf.fit(x,y)
    score.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })
    
df2 = pd.DataFrame(score,columns=["Model","Best_Score","Best_Params"])
df2
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=20,learning_rate=0.01,max_depth=3)
lgb.fit(x_train,y_train)
scores=cross_val_score(lgb,x,y,cv=10,scoring="accuracy")
scores.mean()
knn_score=[]
for k in range(1,16):
    knn = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn,x,y,cv=10)
    knn_score.append(score.mean())
    
knn_score
%matplotlib inline
plt.figure(figsize=(26,12))
plt.style.use('ggplot')
plt.plot([k for k in range(1,16)],knn_score)
for i in range(1,16):
    plt.text(i,knn_score[i-1],(i,knn_score[i-1]))
    plt.xticks([i for i in range(1,16)])
plt.title("K Neighbor Classifiers Value for different K")
plt.xlabel("k neighbors Value")
plt.ylabel("knn Score")
plt.show()
knn = KNeighborsClassifier(n_neighbors = 10)
score=cross_val_score(knn,x,y,cv=10)
score.mean()
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix
classes_name = ['Not Survived','Survived']
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(mat,figsize=(8,6),class_names=classes_name,show_normed=True)
plt.xticks(rotation=0)
plt.show()
