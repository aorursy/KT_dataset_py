import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Ignore warning

import warnings

warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')

test_df = pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')
train_df.head()
df1 = train_df.copy()
#Missing Values

plt.figure(figsize=(20,10))

sns.heatmap(df1.isnull(),yticklabels=False,cbar=False)
#Need to handle missing value of age after EDA as it have more missing value and mean will not be a good option

#Filling missing value of Embarked

df1['Embarked'].unique()
df2 = df1.dropna(subset=['Embarked'])
#Drop Cabin as it have more than 70% missing value

df3 = df2.drop('Cabin',axis=1)
#Make new feature as title

df3['Title'] = df3['Name'].str.replace(r'(.*, )|(\..*)','')
#Drop Name colums

df4 = df3.drop(['Name','Ticket'],axis=1)
df4.head()

#Comment: looks good
df4.info()

#Comment: Everything looks good here
cat_col = ['Pclass','Sex','Embarked','Title']

num_col = ['Age','SibSp','Parch','Fare']
for col in cat_col:

  plt.figure(figsize=(20,10))

  sns.countplot(df4[col])

  plt.title(col)

  plt.show()
#Converting rare values to others in Title

df5 = df4.copy()

df5['Title'] = df5['Title'].apply(lambda x: 'others' if x not in ['Mr','Mrs','Miss','Master'] else x)

df5.describe()
for col in num_col:

  plt.figure(figsize=(20,10))

  sns.distplot(df5[col],kde=False)

  plt.title(col)

  plt.show()
df6 = df5.copy()

#SibSP

df6['SibSp'] = df6['SibSp'].apply(lambda x: 3 if x>2 else x)



#Parch

df6['Parch'] = df6['Parch'].apply(lambda x: 3 if x>2 else x)
for col in cat_col:

  print(df6.groupby(col)['Survived'].mean())
for col in num_col:

  plt.figure(figsize=(20,10))

  sns.swarmplot(x=df6['Survived'],y=df6[col])

  plt.title(col)

  plt.show()
for col in num_col:

  plt.figure(figsize=(20,10))

  sns.boxplot(x=df6['Survived'],y=df6[col])

  plt.title(col)

  plt.show()
sns.heatmap(df6.corr(),annot=True)
sns.boxplot(x=df6['Pclass'],y=df6['Age'])
df6.groupby('Pclass')['Age'].mean()
#Handleing missing value of Age

df7 = df6.copy()



def age_miss(df):

  if np.isnan(df['Age']):

    if df['Pclass']==1:

      return 38

    elif df['Pclass']==2:

      return 29

    elif df['Pclass']==3:

      return 25

  else:    

    return df['Age']  



df7['Age'] = df6.apply(age_miss,axis=1)       
final_df = df7.copy()
test_df = test_df.dropna(subset=['Embarked'])

test_df = test_df.drop('Cabin',axis=1)

test_df['Title'] = test_df['Name'].str.replace(r'(.*, )|(\..*)','')

test_df = test_df.drop(['Name','Ticket'],axis=1)

test_df['Title'] = test_df['Title'].apply(lambda x: 'others' if x not in ['Mr','Mrs','Miss','Master'] else x)

def age_miss(df):

  if np.isnan(df['Age']):

    if df['Pclass']==1:

      return 38

    elif df['Pclass']==2:

      return 29

    elif df['Pclass']==3:

      return 25

  else:    

    return df['Age']  



test_df['Age'] = test_df.apply(age_miss,axis=1) 
final_df.columns
features = ['Pclass', 'Sex','Age','Embarked', 'Title']

X = final_df[features]

y = final_df.Survived

test = test_df[features]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.head(3)
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.compose import make_column_transformer



from sklearn.preprocessing import OneHotEncoder



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import GridSearchCV
#Define preprocessing variable

ohe = OneHotEncoder()



#Define model and there parameters

params = [{'classifier':[RandomForestClassifier()],

           'classifier__n_estimators':[10,25,50,75,100],

           'classifier__max_depth':[5,10,15,20]},

          

          {'classifier':[LogisticRegression()],

           'classifier__penalty':['l1','l2'],

           'classifier__C':[0.01,0.1,1,10,100,1000],

           'classifier__max_iter':[10,100,1000]},

          

          {'classifier':[KNeighborsClassifier()],

           'classifier__n_neighbors':[2,5,10]}]



#Preprocessing

preprocessing = make_column_transformer((ohe,['Pclass', 'Sex','Embarked', 'Title']),remainder='passthrough')



#Pipeline

pipe = Pipeline([('preprocessing',preprocessing),

                 ('classifier',RandomForestClassifier())])
gr = GridSearchCV(pipe,param_grid=params,cv=5).fit(X_train,y_train)
gr.best_params_
submit = pd.DataFrame(gr.predict(test),index=test_df.index,columns=['Survived'])
submit.reset_index(inplace=True)
submit.to_csv('Submission.csv',index=False)