# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

import pandas as pd 

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

## ignore warnings

import warnings; warnings.simplefilter('ignore')

pd.set_option('display.width',1000000)

pd.set_option('display.max_columns', 500)
### some more libraries



import numpy as np # linear algebra

import seaborn as sns

import os

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score,confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from scipy import stats

import statistics as s

from sklearn.linear_model import LogisticRegression


#print(os.getcwd())

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



#### setting Index of the test data as Passenger Id 

#### which acts as a unique identifier for the data

train.set_index('PassengerId',inplace=True)

test.set_index('PassengerId', inplace=True)



### Data frame to hold scores

score_df = pd.DataFrame(columns={'Model_Name','Score'})
# check the columns in the data set

train.dtypes
train.isnull().any()
test.isnull().any()
## lets look at the initial few rows to see how the data looks 

train.head(5)
test.head(5)
## using describe functions give some usefull stats for the columns in the dataset

train.describe()
######################## Column Embarked - Train data 



## Only 2 values are NUll in the data for this column 

print(train[train['Embarked'].isnull()])

### check if any relation between where a person has embarked and the ticket fare 

sns.catplot(x='Embarked',y='Fare',hue ='Pclass',data=train,kind='swarm')
## check how Embarked is distirbuted

print(train['Embarked'].value_counts())
### for now don't see any relation between Embarked so makring this as 'S' which is the 

train['Embarked'].fillna('S',inplace=True)

#train[train['Embarked'].isnull()]
##### Column Cabin 

##### Cabin has a lot of distinct values this is of no use

##### for building a prediction model rather we can use a bit of feature engineering

print(train['Cabin'].value_counts())
#### to make this column usefull , let's only use the Cabin Name (first Alphabet)

train.loc[:,'Cabin'] = train.loc[:,'Cabin'].str[0]

test.loc[:,'Cabin'] = test.loc[:,'Cabin'].str[0]

train[['Cabin','Fare']].groupby(['Cabin'],as_index=False).mean()


sns.catplot(x='Cabin',y='Fare',data=train,kind='bar',ci=None)
sns.catplot(x='Cabin',y='Fare',data=test,kind='swarm')
def calc_cabin_by_fare(df_train):

    # sns.catplot(x='Cabin',y='Fare',data = df_train,kind='bar')

    # plt.show()

    def calculate_cabin(row):

        if row['Fare'] <= 15:

            return 'G'

        elif row['Fare'] <= 19:

            return 'F'

        elif row['Fare'] <= 35:

            return 'T'

        elif row['Fare'] <= 40:

            return 'A'

        elif row['Fare'] <= 45:

            return 'E'

        elif row['Fare'] <= 57:

            return 'D'

        elif row['Fare'] <= 100:

            return 'C'

        else:

            return 'B'





    df_train.loc[df_train['Cabin'].isnull(), 'Cabin'] = df_train[df_train['Cabin'].isnull()].apply(calculate_cabin,

                                                                                                       axis=1)

    return df_train



train = calc_cabin_by_fare(train)

test = calc_cabin_by_fare(test)
### Column Age 

sns.distplot(train.loc[train['Survived']==1,'Age'].dropna(),color='blue',bins=40)

sns.distplot(train.loc[train['Survived']==0,'Age'].dropna(),color='yellow',bins=40)


temp = train[['Sex','Cabin','Age']].groupby(['Sex','Cabin'],as_index=False).mean()



def find_mean_age(Sex,Cabin):

    return temp.loc[(temp['Sex']==Sex)&(temp['Cabin']==Cabin),'Age'].tolist()[0] 





train.loc[train['Age'].isnull(),['Age']] = train.apply(lambda row:find_mean_age(row['Sex']

                                                                                ,row['Cabin']),

                                                       axis=1)



test.loc[test['Age'].isnull(),['Age']] = test.apply(lambda row:find_mean_age(row['Sex']

                                                                                ,row['Cabin']),

                                                       axis=1)
## column - Fare 

## use mean to fill the Fare



test.loc[test['Fare'].isnull(),]



test.loc[test['Fare'].isnull(),['Fare']] = np.mean(train['Fare'])

# Embarked

emb = {'S':1,'C':2,'Q':3}

train['Embarked']=train['Embarked'].replace(emb)

test['Embarked']=test['Embarked'].replace(emb)

### Extract the title from Name and store this in a new column

train['Title']= train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



train['Title'].value_counts()
## titles with very few frequency being renamed as Rare

train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', \

                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', \

                                             'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')



test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
# convert titles into numbers

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train['Title'] = train['Title'].map(titles)

# filling NaN with 0, to get safe

train['Title'] = train['Title'].fillna(0)





test['Title'] = test['Title'].map(titles)

# filling NaN with 0, to get safe

test['Title'] = test['Title'].fillna(0)
## convert sex variable to Numberic



train['Sex_var'] = np.where(train['Sex'] == 'male', 1, 0)

test['Sex_var'] = np.where(test['Sex'] == 'male', 1, 0)
## convert Cabin to Numeric



encode = LabelEncoder()

train['Cabin'] = encode.fit_transform(train['Cabin'])

test['Cabin'] = encode.transform(test['Cabin'])
plt.figure(figsize=(14,12))

plt.title('Correlation of Features')

cor = train.corr()

sns.heatmap(cor, cmap="YlOrRd", annot=True)

plt.show()
### create family count

train['fmly_count'] = train['SibSp']+train['Parch']

test['fmly_count'] = test['SibSp']+test['Parch']



##create is_alone



train['is_alone'] = np.where((train['SibSp'] + train['Parch']) == 0, 1, 0)

test['is_alone'] = np.where((test['SibSp'] + test['Parch']) == 0, 1, 0)







### let's draw the co-realtion matrix again to include new features



cor = train.corr()

plt.figure(figsize=(14,12))

plt.title('Correlation of Features')

sns.heatmap(cor, cmap="YlOrRd", annot_kws={'fontsize':8},annot=True,linewidths=0.1,vmax=1.0)

plt.show()

#print(train.dtypes)

#col_list=['Pclass','Sex_var','Age','Fare','Cabin','fmly_count','is_alone']

## selecting the column names that we would use for prediction modelling 

col_list = ['Pclass', 'Sex_var', 'Age', 'Fare', 'Cabin', 'fmly_count','Title']

X = train[col_list].copy()

y = train['Survived'].copy()



test_orig = test.copy()

test = test[col_list]

#print(test.describe())
## by deafult 75% train data and 25% test data divison 

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100)
## repeated import here 

## just to increase readability



from sklearn.neighbors import KNeighborsClassifier



scale = MinMaxScaler()

X_scaled=pd.DataFrame(scale.fit_transform(X),columns=X.columns)

test_scaled = pd.DataFrame(scale.fit_transform(test), columns=test.columns)

test_scaled.index=test.index



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,random_state=100)







## fit the model 

knn_model = KNeighborsClassifier(n_neighbors=6).fit(X_train,y_train)

knn_score = knn_model.score(X_test,y_test)

print('Score of fitted model is: ')

print(knn_score)



### let's get the scores now

predict_df = pd.DataFrame()

predict_df['PassengerId'] = test.index

predict_df['Survived'] = knn_model.predict(test_scaled)

#predict_df.to_csv('submission100_knn_nvalue_6.csv', index=False)

score_df = score_df.append({'Model_Name':'K Nearest Neighbour','Score':knn_score},ignore_index=True)

temp = test.copy()

temp['Survived'] = predict_df['Survived'].tolist() 

print(temp)

print('Percentage of People who Survived - %3f'%(temp['Survived'].mean()*100))
from sklearn.model_selection import validation_curve



### this is how you do a validation curve

train_score,test_score = validation_curve(KNeighborsClassifier(),X,y,param_name='n_neighbors',param_range=range(1,11),cv=5)

train_score_mean = np.mean(train_score,axis=1)

test_score_mean = np.mean(test_score, axis=1)

plt.plot(range(1,11),train_score_mean,'-o',label='train score')

plt.plot(range(1, 11), test_score_mean, '-o', label='train score')

plt.xlabel('N_neighbours values')

plt.ylabel('Accuracy')

plt.title('Variation of Accuracy with input parameter n_neighbour')

plt.legend(loc='best')

plt.show()

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV



lr_model = LogisticRegression(random_state=100)

grid_values = {'penalty':['l1', 'l2'],'C':[0.01, 0.1, 1, 10, 100]}

grid_search_model = GridSearchCV(lr_model,param_grid=grid_values,cv=3)

grid_search_model.fit(X_scaled,y)

print(grid_search_model.best_estimator_)

print('Model Accuracy')

print(grid_search_model.best_score_)

print(grid_search_model.best_params_)

predict_df = pd.DataFrame()

predict_df['PassengerId'] = test_scaled.index

predict_df['Survived'] = grid_search_model.predict(test_scaled)

predict_df.to_csv('submission101_lr_gsearch_opt.csv', index=False)

score_df = score_df.append({'Model_Name':'LR - with Grid Search','Score':grid_search_model.best_score_},ignore_index=True)

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV





svc_model = SVC(C=100,random_state=100).fit(X_train,y_train)

scv_score = svc_model.score(X_test,y_test) 



print('SVC Model score is ')

print(scv_score)



predict_df = pd.DataFrame()

predict_df['PassengerId'] = test_scaled.index

predict_df['Survived'] = svc_model.predict(test_scaled)

#predict_df.to_csv('submission101_lr_gsearch_opt.csv', index=False)

score_df = score_df.append({'Model_Name':'SVC Model - C 100','Score':scv_score},ignore_index=True)

from sklearn.naive_bayes import GaussianNB



X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100)

gaussnb_model = GaussianNB().fit(X_train,y_train)

print('Naive Bayes Accuracy')

print(gaussnb_model.score(X_test,y_test))

gb_model_pred = gaussnb_model.predict(test)

score_df = score_df.append({'Model_Name':'Gaussian Naive Bayes','Score':scv_score},ignore_index=True)



from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=3,random_state=100).fit(X_train,y_train)

dt_model_score = dt_model.score(X_test,y_test)

print('Decision Tree Accuracy')

print(dt_model_score)

score_df = score_df.append({'Model_Name':'Decision Tree','Score':dt_model_score},ignore_index=True)

from sklearn.ensemble import GradientBoostingClassifier



## fitting the model on complete data set 

gbf_model = GradientBoostingClassifier(learning_rate=0.001,n_estimators=3000).fit(X,y)

gbf_score = gbf_model.score(X_test,y_test)

print('GBF score -')

print(gbf_score)

gbf_model_pred = gbf_model.predict(test)

score_df = score_df.append({'Model_Name':'GradientBoostingClassifier','Score':gbf_score},ignore_index=True)





predict_df = pd.DataFrame()

predict_df['PassengerId'] = test.index

predict_df['Survived'] = gbf_model.predict(test)

predict_df.to_csv('submission_temp.csv', index=False,line_terminator="")

    

### some more file clearning

file_data = open('submission_temp.csv', 'rb').read()

open('submission_gbfc_model.csv', 'wb').write(file_data[:-1])
from sklearn.ensemble import RandomForestClassifier



rf_model= RandomForestClassifier(n_estimators=1000,max_depth=7)

rf_model.fit(X,y)

rf_score = rf_model.score(X_test,y_test)

print('RF Model Score')

print(rf_score)

score_df = score_df.append({'Model_Name':'RandomForestClassifier','Score':rf_score},ignore_index=True)

rf_model_pred = rf_model.predict(test)







predict_df = pd.DataFrame()

predict_df['PassengerId'] = test.index

predict_df['Survived'] = rf_model.predict(test)

predict_df.to_csv('submission_temp.csv', index=False,line_terminator="")

    

### some more file clearning

file_data = open('submission_temp.csv', 'rb').read()

open('submission_rf_model.csv', 'wb').write(file_data[:-1])
print(score_df[['Model_Name','Score']].sort_values(by='Score'))
final_pred = []

for i in range(0,len(test)):

   final_pred.append(s.mode(np.array([rf_model_pred[i],gb_model_pred[i],gbf_model_pred[i]])))



predict_df = pd.DataFrame()

predict_df['PassengerId'] = test.index

predict_df['Survived'] = final_pred

predict_df.to_csv('submission_temp.csv', index=False,line_terminator="")

    

### some more file clearning

file_data = open('submission_temp.csv', 'rb').read()

open('submission_final.csv', 'wb').write(file_data[:-1])