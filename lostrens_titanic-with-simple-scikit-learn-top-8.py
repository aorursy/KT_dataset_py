#!pip install kaggle
#from google.colab import drive

#drive.mount('/content/gdrive')
#from google.colab import files

#files.upload() 
#!pip install -q kaggle

#!mkdir -p ~/.kaggle

#!cp kaggle.json ~/.kaggle/

#!ls ~/.kaggle

## we need to set permissions 

#!chmod 600 /root/.kaggle/kaggle.json
# Google Colab directory setting. Comment out after run this line



#import os

#os.chdir('/content/gdrive/My Drive/Competitions/kaggle/Kaggle-Titanic/nbs')  #change dir
#!pwd
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# plotly

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import matplotlib.style as style

style.available





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
df.head()
test.head()
df.groupby(by=['Pclass'])['Survived'].agg(['mean','count'])
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/2bc37b51-c9e4-402e-938e-70d3145815f2/d787jna-1b3767d2-f297-4b73-a874-7cfa6d1e8a69.png/v1/fill/w_1600,h_460,q_80,strp/r_m_s__titanic_class_system_by_monroegerman_d787jna-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NDYwIiwicGF0aCI6IlwvZlwvMmJjMzdiNTEtYzllNC00MDJlLTkzOGUtNzBkMzE0NTgxNWYyXC9kNzg3am5hLTFiMzc2N2QyLWYyOTctNGI3My1hODc0LTdjZmE2ZDFlOGE2OS5wbmciLCJ3aWR0aCI6Ijw9MTYwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.6krQcPQvsfcQ_ZJ_CGvufi9MT-PJkkg1I8-grLy7Hiw")
sex_survived= df.groupby(by=['Sex','Survived'])['Survived'].agg(['count']).reset_index()

sex_survived
plt.figure(figsize=(10, 5))

style.use('seaborn-notebook')

sns.barplot(data=sex_survived, x='Sex',y='count', hue='Survived');
# Plotly configuration function for Google Colab. We need to run this function for showing plotly graph in the Google colab

def configure_plotly_browser_state():

    

    import IPython

    display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',

            },

          });

        </script>

        '''))
male_survived=pd.DataFrame(df['Age'][(df['Sex']=='male')& (df['Survived']==1)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})

female_survived=pd.DataFrame(df['Age'][(df['Sex']=='female')& (df['Survived']==1)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})

male_not_survived=pd.DataFrame(df['Age'][(df['Sex']=='male') & (df['Survived']==0)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})

female_not_survived=pd.DataFrame(df['Age'][(df['Sex']=='female') & (df['Survived']==0)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})



from plotly import tools



#Add function here

configure_plotly_browser_state()

init_notebook_mode(connected=False)



trace1 = go.Scatter(

    x = male_survived['Age'].sort_values(ascending=False),

    y = male_survived['Number'],

    name='Survived Male',

    fill='tozeroy',

    #connectgaps=True



)

trace2 = go.Scatter(

    x = female_survived['Age'].sort_values(ascending=False),

    y = female_survived['Number'],

    name='Survived Female',

    fill='tozeroy',

    #connectgaps=True



)

trace3 = go.Scatter(

    x = male_not_survived['Age'].sort_values(ascending=False),

    y = male_not_survived['Number'],

    fill='tozeroy',

    name='Not Survived Male',

    #connectgaps=True



)

trace4 = go.Scatter(

    x = female_not_survived['Age'].sort_values(ascending=False),

    y = female_not_survived['Number'],

    fill='tozeroy',

    name = 'Not Survived Female',

    



)



fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Male', 'Female'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace3, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace4, 2, 1)



fig['layout']['xaxis2'].update(title='Age')

fig['layout'].update(height=700, width=1200,

                     title='Age Gender Survive')







iplot(fig)
# We need to make some data wrangling with both train and test data

df_all = [df,test]
for data in df_all:

    print(f"\n -------- {data.index } ------- \n")

    print(data.isnull().sum())


for data in df_all:

    data['isAlone']=1



    data['Family_No'] = data['Parch'] + data['SibSp'] + 1

        

    data['isAlone'].loc[data['Family_No']>1]=0

    

    data['Age'].fillna(round(data['Age'].mean()), inplace=True)

    

    #``df.fillna(df.mode().iloc[0])`` If you want to impute missing values with the mode in a dataframe 

    data['Embarked'].fillna(data['Embarked'].mode().iloc[0], inplace=True)

    

    # mean of each Pclass

    #data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    data['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.mean()))

    

    
df.head()
test.isAlone.value_counts()
# Drop features that will not process

for data in df_all:

    data.drop(columns=['PassengerId','Name','Cabin','Ticket','SibSp','Parch'],inplace=True,axis=1)
for data in df_all:

    print(f"\n -------- {data.index } ------- \n")

    print(data.isnull().sum())
#get_dummies() function allows us to make a column for each categorical variable in features

test = pd.get_dummies(test,columns=['Sex','Embarked'])

df = pd.get_dummies(df,columns=['Sex','Embarked'])
df.head()
y=df['Survived']

X=df.drop(columns=['Survived'],axis=1)
#Split the data for training and testing

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# Original pararamaters

DT= DecisionTreeClassifier()

DT.fit(X_train, y_train)

DT.score(X_test,y_test)
# Checking the hyperparamates of decision tree classifier

from IPython.display import HTML, IFrame

IFrame("https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier", width=1100, height=500)
# Grid CV

parameters1 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True)}]
# Grid Search for Decision Treee

Grid1 = GridSearchCV(DT, parameters1, cv=4,return_train_score=True,iid=True)



Grid1.fit(X_train,y_train)
scores = Grid1.cv_results_
for param, mean_train in zip(scores['params'],scores['mean_train_score']):

    print(f"{param} accuracy on training data is {mean_train}")
# best estimator for in Decision tree paramaters that we define. 

Grid1.best_estimator_
#Max score for above parameters

max(scores['mean_train_score'])
XGB = XGBClassifier()
#parameters2 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True),'n_estimators':[100]}]



parameters3 =[{"learning_rate": [0.05, 0.10, 0.15, 0.20] ,"max_depth": [ 3, 4, 5, 6], "min_child_weight": [3,5,7],"gamma": [ 0.0, 0.1, 0.2 ,0.3],"colsample_bytree" : [ 0.4, 0.5]}]
Grid1 = GridSearchCV(XGB, parameters3, cv=2,return_train_score=True)



Grid1.fit(X_train,y_train)
scores = Grid1.cv_results_
# best estimator for in Decision tree paramaters that we define. 

Grid1.best_estimator_
max(scores['mean_train_score'])
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.5, gamma=0.0, learning_rate=0.1,

       max_delta_step=0, max_depth=3, min_child_weight=5, missing=None,

       n_estimators=100, n_jobs=1, nthread=None,

       objective='binary:logistic', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

       subsample=1)
XGB.fit(X_train, y_train)
XGB.score(X_test,y_test)
#pred = XGB.predict(test)
#result = pd.DataFrame(pred,columns=['Survived'])
#test1  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
#submission = result.join(test1['PassengerId']).iloc[:,::-1]
#submission.to_csv('submission.csv', index=False)
df = pd.read_csv('../input/train.csv' , header = 0,dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0,dtype={'Age': np.float64})
df.info()


for data in [df,test]:

    data['isAlone']=1



    data['Family_No'] = data['Parch'] + data['SibSp'] + 1

        

    data['isAlone'].loc[data['Family_No']>1]=0

    

    data['Age'].fillna(data['Age'].mean(), inplace=True)

    

    #``df.fillna(df.mode().iloc[0])`` If you want to impute missing values with the mode in a dataframe 

    data['Embarked'].fillna(data['Embarked'].mode().iloc[0], inplace=True)

    

    # mean of each Pclass

    #data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    data['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.mean()))
#import re

# We have two types tickets first only number and the second one letter and number. We are going to have letters and create a feature.

#trial_addFeature['Ticket_name'] =[]

#test_addFeature['Ticket_name'] =[]

# for data in df_all:

#     for i,k in enumerate(data['Ticket']):

#         try:

#             x=k.split(" ")[1]

#             data['Ticket'].replace(data['Ticket'][i],k.split(" ")[0],inplace=True)

#         except IndexError:

#             data['Ticket'].replace(data['Ticket'][i],"No_letter",inplace=True)





#     data['Ticket'] =data['Ticket'].map(lambda x: re.sub('[./]', '', x))

#     data['Ticket'] =data['Ticket'].map(lambda x: x.upper())





#df['Ticket_name'] =df['Ticket_name'].map(lambda x: re.sub('[./]', '', x))

#df['Ticket_name'] =df['Ticket_name'].map(lambda x: x.upper())



#set(Ticket_name)            
#set(data['Ticket'])
#test['Ticket'] = test_addFeature['Ticket']

#df['Ticket'] = df_addFeature['Ticket']

test.head()
from sklearn.preprocessing import LabelEncoder



LE = LabelEncoder()

#Upper limit is 100 but the oldest person is 80 years old

for data in [test,df]:

    bins = [-1,0,5,10, 15, 25, 50,100]

    labels = ['Unknown','Baby','Child','Young','Teen','Adult','Old']

    data['Age'] = pd.cut(data['Age'], bins=bins,labels=labels)

    data['Age'] = data['Age'].astype(str)

test['Age'] = LE.fit_transform(test['Age'])  

df['Age'] = LE.fit_transform(df['Age'])  



#data['Age'] = data['Age'].astype(int)

for data in [test,df]:

    for i,k in enumerate(data['Name']):

        x=k.split(",")[1]

        data['Name'].replace(data['Name'][i],x.split(" ")[1],inplace=True)

        
df['Name'].value_counts()
all_data = [df,test]

Known = ['Mr.','Miss.','Mrs.','Master.','Ms.','Mlle.','Mme.']

for k in (all_data):

    for i,data in enumerate(k['Name']):

        if (data) in Known:

            if(data=='Mlle.'):

                k['Name'] = k['Name'].replace('Mlle.','Miss.')

            elif(data=='Ms.'):

                k['Name'] = k['Name'].replace('Ms.','Miss.')

            elif(data=='Mme.'):

                k['Name'] = k['Name'].replace('Mme.','Mrs.')

            else:

                continue

        else:

            k['Name'] = k['Name'].replace(data,'not_known')

        

            

            

        

        

    
# Survived difference between people who had different title

df['Name'][df['Survived']==1].value_counts()/df['Name'].value_counts()
df.info()

#columns = ['Embarked','Age','Sex','Name']

#

# for data in [df,test]:

#     for i in columns:

#         data[i] = data[i].astype(str)

#         data[i] = LE.fit_transform(data[i])



# Create feature for each categories

test=pd.get_dummies(test,columns=['Embarked','Name'])

df=pd.get_dummies(df,columns=['Embarked','Name'])

test['Sex'] = LE.fit_transform(test['Sex'])

df['Sex'] = LE.fit_transform(df['Sex'])
for data in [df,test]:

    data.drop(columns=['Ticket','Cabin','SibSp','Parch','PassengerId'], inplace=True, axis=1)
df.drop(columns=['Embarked_Q'],axis=1,inplace=True)

test.drop(columns=['Embarked_Q'],axis=1,inplace=True)
for data in [df,test]:



    scale = StandardScaler().fit(data[['Fare']])

    data[['Fare']] = scale.transform(data[['Fare']])

y=df['Survived']

X=df.drop(columns=['Survived'],axis=1)

#Split the data for training and testing

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


df.head()
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,  VotingClassifier
parameters_DC = [{'max_depth':[50,100],'min_samples_split': [0.1,0.2,0.5,0.8,0.9]}]



paramaters_RF = [{'max_depth':[2,5,10,15,20,50],'min_samples_split': [0.1,0.2,0.5,0.8],'n_estimators':[100]}]



parameters_XGB =[{"learning_rate": [0.2,0.5,0.8,0.9] ,"max_depth": [1, 3,5, 10], "min_child_weight": [3,5,7,10,20],"gamma": [0.1, 0.2 ,0.4,0.7],'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],'n_estimator':[100,1000,2000,4000]}]



parameters_GBC =[{"learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01] ,"max_depth": [ 3, 4, 5, 6], "min_samples_leaf" :[50,100,150],"n_estimators" : [16, 32, 64, 128]}]



parameters_ADA =[{'algorithm':['SAMME'],"base_estimator__criterion" : ["gini"],"base_estimator__splitter" :   ["best", "random"],"n_estimators": [500,1000],"learning_rate":  [ 0.01, 0.1, 1.0]}] 



DC = DecisionTreeClassifier()







Grid_DC = GridSearchCV(DC, parameters_DC, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)

#Fit the model

Grid_DC.fit(X_train,y_train)



# Best estimator parameters

DC_best = Grid_DC.best_estimator_



# Best score for the model with the paramaters

Grid_DC.best_score_
RF = RandomForestClassifier()



Grid_RF = GridSearchCV(RF, paramaters_RF, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)

#Fit the model

Grid_RF.fit(X_train,y_train)



# Best estimator parameters

RF_best = Grid_RF.best_estimator_



# Best score for the model with the paramaters

Grid_RF.best_score_

XGB = XGBClassifier()



Grid_XGB = GridSearchCV(XGB, parameters_XGB, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)

#Fit the model

Grid_XGB.fit(X_train,y_train)



# Best estimator parameters

XGB_best = Grid_XGB.best_estimator_



# Best score for the model with the paramaters

Grid_XGB.best_score_
GBC = GradientBoostingClassifier()





Grid_GBC = GridSearchCV(GBC,parameters_GBC, cv=4, scoring="accuracy", n_jobs= 4, return_train_score=True,verbose = 1)



Grid_GBC.fit(X_train,y_train)



GBC_best = Grid_GBC.best_estimator_



# Best score

Grid_GBC.best_score_
ADA = AdaBoostClassifier(DC_best)





Grid_ADA = GridSearchCV(ADA,parameters_ADA, cv=4, scoring="accuracy", n_jobs= 4, return_train_score=True,verbose = 1)



Grid_ADA.fit(X_train,y_train)



ADA_best = Grid_ADA.best_estimator_



# Best score

Grid_ADA.best_score_
parameters_SVM = {'C': [0.1, 1, 10,50,100], 'gamma' : [0.001, 0.01, 0.1, 1,10]}



from sklearn.svm import SVC

SVMC =SVC(probability=True)

Grid_SVC = GridSearchCV(SVMC, parameters_SVM, scoring="accuracy", return_train_score=True,verbose = 1,cv=2)



Grid_SVC.fit(X_train, y_train)



SVM_best = Grid_SVC.best_estimator_



# Best score

Grid_SVC.best_score_
voting = VotingClassifier(estimators=[('ADA', ADA_best),('DC', DC_best),('RF', RF_best),('GBC',GBC_best),('XGB',XGB_best),('SVC',SVM_best)],weights=[3,0,0,1,3,3], voting='hard', n_jobs=4)



voting_result = voting.fit(X_train, y_train)
voting.score(X_test,y_test)
pred = voting.predict(test)
test_2 = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
result = pd.DataFrame(pred,columns=['Survived'])

submission13 = result.join(test_2['PassengerId']).iloc[:,::-1]
submission13.to_csv('submission13.csv', index=False)
#!kaggle kernels push