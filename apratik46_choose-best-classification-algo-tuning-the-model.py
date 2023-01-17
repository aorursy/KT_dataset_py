import numpy as np

import pandas as pd

import matplotlib.pyplot as pl

%matplotlib inline
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train.info()
test.info()
test_passengerid=test.PassengerId
df=train.append(test,ignore_index=True,sort=False)
df.info()
df.isnull().sum()
male_avg_age=df[df.Sex=='male'].Age.mean()
female_avg_age=df[df.Sex=='female'].Age.mean()
df[df.Sex=='male'].Age.hist(bins=[0,10,20,30,40,50,60,70,80,90,100])

pl.xlabel('Male')

pl.text(male_avg_age, 200,male_avg_age)

pl.show()

df[df.Sex=='female'].Age.hist(bins=[0,10,20,30,40,50,60,70,80,90,100])

pl.text(female_avg_age, 100,female_avg_age)

pl.xlabel('Female')

pl.show()
df['Name']
df['title']=df.Name.apply(lambda n: n.split(',')[1].split('.')[0].strip())
df['title'].unique()
titles_cat = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royal",

    "Don":        "Royal",

    "Sir" :       "Royal",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royal",

    "Dona":       "Royal",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royal"

}
df.title=df.title.map(titles_cat)

df.title.value_counts()
ag_group=df.groupby(['Sex','Pclass','title'])

ag_group.Age.median()
df.Age=ag_group.Age.apply(lambda a:a.fillna(a.median()))
df.Cabin.unique()
df.Cabin=df.Cabin.fillna('U')
df.Embarked.value_counts()
df.Embarked=df.Embarked.fillna(df.Embarked.value_counts().index[0])
df.Fare=df.Fare.fillna(df.Fare.median())
df.isnull().sum()
df.Cabin=df.Cabin.apply(lambda x:x[0])
df.head()
df.Fare.hist()
from sklearn import preprocessing
mm_scaler=preprocessing.MinMaxScaler()
data=df['Fare'].values

scaled_fare=mm_scaler.fit_transform(pd.DataFrame(data))
scaled_fare=pd.DataFrame(scaled_fare,columns=['Nm_Fare'])
prepared_data=pd.concat([df,scaled_fare],axis=1)
prepared_data.head()
prepared_data.Sex=prepared_data.Sex.map({'male':0,'female':1})
pclass_dum=pd.get_dummies(prepared_data.Pclass,prefix='Pclass')

title_dum=pd.get_dummies(prepared_data.title,prefix='title')

cabin_dum=pd.get_dummies(prepared_data.Cabin,prefix='Cabin')

emb_dum=pd.get_dummies(prepared_data.Embarked,prefix='Embarked')
prepared_data=pd.concat([prepared_data,pclass_dum,title_dum,cabin_dum,emb_dum],axis=1)
prepared_data.drop(['Fare','Pclass','title','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)

prepared_data.head()
train_len=len(train)

train_len
train=prepared_data[ :train_len]

test=prepared_data[train_len: ]
train.Survived.isnull().any()
test.Survived.notnull().any()
train.Survived=train.Survived.astype(int)
X=train.drop('Survived',axis=1).values

y=train.Survived.values
X_test=test.drop('Survived',axis=1).values
from sklearn.model_selection import train_test_split
X_train, X_train_test, y_train, y_train_test = train_test_split(X,y,test_size=0.1,random_state=0)
print(X_train.shape,X_train_test.shape,y_train.shape,y_train_test.shape)
from sklearn.metrics import accuracy_score,fbeta_score

from time import time
def train_predict(model, sample_size, X_train,y_train, X_test, y_test):

    result={}

    start=time()

    model=model.fit(X_train[:sample_size],y_train[:sample_size])

    end=time()

    

    result['train_time']=end-start

    

    start=time()

    pred_test=model.predict(X_test)

    pred_train=model.predict(X_train)

    end=time()

    

    result['pred_time']=end-start

    

    result['acc_train'] = accuracy_score(y_train, pred_train)

    result['acc_test'] = accuracy_score(y_test, pred_test)

    

    result['f_train'] = fbeta_score(y_train, pred_train, beta = 0.5)

    result['f_test'] = fbeta_score(y_test, pred_test, beta = 0.5)

    

    return(result)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
model1=LogisticRegression(random_state=0)

model2=GradientBoostingClassifier(random_state=0)

model3=RandomForestClassifier(random_state=0)
result={}

sample_5=int(len(y_train)*0.05)

sample_25=int(len(y_train)*0.25)

sample_100=len(y_train)



for model in [model1, model2, model3]:

    model_name=model.__class__.__name__

    result[model_name]={}

    for i,sample in enumerate([sample_5,sample_25,sample_100]):

        result[model_name][i]=train_predict(model,sample,X_train, y_train,X_train_test,y_train_test)
print(result)
import matplotlib.pyplot as pl

import matplotlib.patches as mpatches

def evaluate(results):

    

    # Create figure

    fig, ax = pl.subplots(2, 3, figsize = (11,7))



    # Constants

    bar_width = 0.3

    colors = ['#A00000','#00A0A0','#00A000']

    

    # Super loop to plot four panels of data

    for k, learner in enumerate(results.keys()):

        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):

            for i in np.arange(3):

                

                # Creative plot code

                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])

                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])

                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])

                #ax[j//3, j%3].set_xlabel("Training Set Size")

                ax[j//3, j%3].set_xlim((-0.1, 3.0))

    

    # Add unique y-labels

    ax[0, 0].set_ylabel("Time (in seconds)")

    ax[0, 1].set_ylabel("Accuracy Score")

    ax[0, 2].set_ylabel("F-score")

    ax[1, 0].set_ylabel("Time (in seconds)")

    ax[1, 1].set_ylabel("Accuracy Score")

    ax[1, 2].set_ylabel("F-score")

    

    # Add titles

    ax[0, 0].set_title("Model Training")

    ax[0, 1].set_title("Accuracy Score on Training Subset")

    ax[0, 2].set_title("F-score on Training Subset")

    ax[1, 0].set_title("Model Predicting")

    ax[1, 1].set_title("Accuracy Score on Testing Set")

    ax[1, 2].set_title("F-score on Testing Set")

    

    

    # Set y-limits for score panels

    ax[0, 1].set_ylim((0, 1))

    ax[0, 2].set_ylim((0, 1))

    ax[1, 1].set_ylim((0, 1))

    ax[1, 2].set_ylim((0, 1))



    # Create patches for the legend

    patches = []

    for i, learner in enumerate(results.keys()):

        patches.append(mpatches.Patch(color = colors[i], label = learner))

    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \

               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')

    

    # Aesthetics

    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)

    pl.tight_layout()

    pl.show()
evaluate(result)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
mymodel= model1
dual=[True,False]

max_iter=[100,110,120,130,140]

penalty=['l2']

param_grid=dict(dual=dual,max_iter=max_iter,penalty=penalty)
grid=GridSearchCV(estimator=mymodel,param_grid=param_grid,cv=3,n_jobs=-1)

grid.fit(X,y)
best_model=grid.best_estimator_
print(best_model)

print(grid.best_score_)
predictions=best_model.predict(X_test)
load_kaggle=pd.DataFrame({'PassengerId':test_passengerid,'Survived':predictions})



load_kaggle.to_csv('./Titanic_Logistic_Regression.csv',index=False)
load_kaggle