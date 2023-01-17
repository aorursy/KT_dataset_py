# Pandas library in python to read the csv file.

import pandas as pd



# for numerical computaions use numpy library

import numpy as np



# data visualization

import missingno as msno

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

 

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix
# Create a pandas dataframe and assign it to variable.

titanic = pd.read_csv('../input/titanic/train.csv')

titanic_test = pd.read_csv('../input/titanic/test.csv')
# Print first 5 rows of the dataframe.

titanic.head()
# Print Last 5 rows of the dataframe.

titanic_test.tail() 
# gives shape of datase in (rows,columns)

titanic.shape
# Describe gives us statistical information about numerical columns in the dataset

titanic.describe()
# unique values or range for feature set

print('Genders:', titanic['Sex'].unique())

print('Embarked:', titanic['Embarked'].unique())

print('Pclass:', titanic['Pclass'].unique())

print('Survived:', titanic['Survived'].unique())

print('SibSp Range:', titanic['SibSp'].min(),'-',titanic['SibSp'].max())

print('Parch Range:', titanic['Parch'].min(),'-',titanic['Parch'].max())

print('Family size range:', (titanic['Parch']+titanic['SibSp']).min(),'-',(titanic['Parch']+titanic['SibSp']).max())

print('Fare Range:', titanic['Fare'].min(),'-',titanic['Fare'].max())
# info method provides information about dataset like 

# total values in each column, null/not null, datatype, memory occupied etc

titanic.info()
msno.matrix(titanic)
msno.matrix(titanic_test)
# Let's write a function to print the total percentage of the missing values.

# (This can be a good excercise for beginers to try to write sample function like this)



# This function takes a Dataframe (df) as input and returns two columns,total missing values and total missing alues percentage

def missing_data(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)

    return pd.concat([total,percent], axis = 1 ,keys = ['total','percent'])
missing_data(titanic)
# check missing values in test dataset

missing_data(titanic_test)
# COMPLETING: complete or delete missing values in train and test dataset

dataset = [titanic,titanic_test]



for data in dataset:

    # coplete missing age with median

    data['Age'].fillna(data['Age'].median(),inplace = True)

    

    # complete Embarked with mode

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    

    # complete missing Fare with median

    data['Fare'].fillna(data['Fare'].median(),inplace = True)
missing_data(titanic)
titanic.drop(['Cabin'], axis=1, inplace = True)

titanic_test.drop(['Cabin'],axis=1,inplace=True)
titanic.head()
titanic_test.head()
missing_data(titanic)
net_Survived=titanic['Survived'].value_counts().to_frame().reset_index().rename(columns={'index':'Survived','Survived':'count'})
fig = go.Figure([go.Pie(labels=net_Survived['Survived'], values=net_Survived['count'])])



fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')



fig.update_layout(title="Travellers survived on titanic",title_x=0.5)

fig.show()
age_analysis=titanic[titanic['Survived']==1]['Sex'].value_counts().reset_index().rename(columns={'index':'Sex','Sex':'count'})
fig = go.Figure(go.Bar(x=age_analysis['Sex'],y=age_analysis['count']))

fig.update_layout(autosize=False,width=400,height=500,title_text='Analysis of Survived travellers by gender',xaxis_title="sex",yaxis_title="count",paper_bgcolor="lightsteelblue")

fig.show()
def draw(graph):

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
sns.set(style="darkgrid")

plt.figure(figsize = (5, 6))

x = sns.countplot(titanic['Sex'])

draw(x)
plt.figure(figsize = (10, 6))

graph  = sns.countplot(y = "Embarked", hue ="Survived", data = titanic)

for p in graph.patches:

        Total = '{:,.0f}'.format(p.get_width())

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        graph.annotate(Total, (x, y))
FGrid = sns.FacetGrid(titanic, row='Pclass', aspect=2)

FGrid.map(sns.pointplot, 'Embarked', 'Survived', 'Sex', palette=None,  order=None, hue_order=None)

FGrid.add_legend()
titanic.drop(['Embarked'], axis=1, inplace = True)

titanic_test.drop(['Embarked'],axis=1,inplace=True)
titanic=titanic.dropna()

titanic['age_category']=np.where((titanic['Age']<19),"below 19",

                                 np.where((titanic['Age']>18)&(titanic['Age']<=30),"19-30",

                                    np.where((titanic['Age']>30)&(titanic['Age']<=50),"31-50",

                                                np.where(titanic['Age']>50,"Above 50","NULL"))))

age=titanic['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})
titanic_age=titanic['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'count'})
colors=['pink','teal','orange','green']

fig = go.Figure([go.Pie(labels=titanic_age['age_category'], values=titanic_age['count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=15,

                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title="Titanic Age Categories",title_x=0.5)

fig.show()
titanic['survived_or_not']=np.where(titanic['Survived']==1,"Survived",np.where(titanic['Survived']==0,"Died","null")) # .head(2)'



sun_df=titanic[['Sex','survived_or_not','age_category','Fare']].groupby(['Sex','survived_or_not','age_category']).agg('sum').reset_index()
fig = px.sunburst(sun_df, path=['Sex','survived_or_not','age_category'], values='Fare')

fig.update_layout(title="Titanic dataset distribution by Drilldown (Sex, Survived, Age Categories)",title_x=0.5)

fig.show()
sur_age=titanic[titanic['Survived']==1]['Age']

un_age=titanic[titanic['Survived']==0]['Age']
fig = go.Figure(go.Box(y=sur_age,name="Age")) 

fig.update_layout(title="Distribution of Age by Survived travellers", autosize=False, width=600, height=700)

fig.show()
fig = go.Figure(go.Box(y=un_age,name="Age")) 

fig.update_layout(title="Distribution of Age By Unsurvived tarvellers", autosize=False, width=600, height=700)

fig.show()
ax = sns.countplot(y="Pclass", hue="Survived", data=titanic, palette="Set1")

for p in ax.patches:

        Total = '{:,.0f}'.format(p.get_width())

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(Total, (x, y))
# combine test and train as single to apply some function, we will use it again in Data Preprocessing

all_data=[titanic,titanic_test]



for dataset in all_data:

    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
sns.set(style="darkgrid")

plt.figure(figsize = (7, 6))

x = sns.countplot(titanic['Family'])

draw(x)
surfamily_size = titanic[titanic['Survived'] == 1]
fig = go.Figure(data=go.Violin(y=surfamily_size['Family'],

                               marker_color="blue",

                               x0='Family size'))



fig.update_layout(title="Survived travellers family size")

fig.show()
unfamily_size = titanic[titanic['Survived'] == 0]
fig = go.Figure(data=go.Violin(y=unfamily_size['Family'],

                               marker_color="blue",

                               x0='Family size'))



fig.update_layout(title="Unsurvived travellers family size")

fig.show()
axes = sns.factorplot('Family','Age','Survived',

                      data=titanic, aspect = 2,kind='bar', orient='v',palette="Set2")
# create bin for age features. 

for dataset in all_data:

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
plt.figure(figsize = (8, 5))

bin = sns.countplot(x='Age_bin', hue='Survived', data=titanic,palette="Set1")

draw(bin)
for dataset in all_data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','medium_fare','Average_fare','high_fare'])
plt.figure(figsize = (8, 5))

sns.countplot(x='Pclass', hue='Fare_bin', data=titanic)
pd.DataFrame(abs(titanic.corr()['Survived']).sort_values(ascending = False))
# Generate a mask for the upper triangle (taken from seaborn example gallery)

corr=titanic.corr()  #['Survived']



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (14,8))

sns.heatmap(corr, 

            annot=True,

            mask = mask,

            cmap = 'Blues',

            linewidths=.9, 

            linecolor='white',

            vmax = 0.3,

            fmt='.2f',

            center = 0,

            square=True)

plt.yticks(rotation = 0)

plt.title("Correlation Matrix", y = 1,fontsize = 25, pad = 20);
titanic.info()
drop_col= ["survived_or_not","age_category"]

titanic.drop(drop_col,axis=1,inplace=True)
# Convert ‘Sex’ feature into numeric.

genders = {"male": 0, "female": 1}



for dataset in all_data:

    dataset['Sex'] = dataset['Sex'].map(genders)

titanic['Sex'].value_counts()
for dataset in all_data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 28), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 35), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5

    dataset.loc[ dataset['Age'] > 45, 'Age'] = 6

titanic['Age'].value_counts()
for dataset in all_data:

    drop_column = ['Age_bin','Fare','Name','Ticket', 'PassengerId','SibSp','Parch','Fare_bin']

    dataset.drop(drop_column, axis=1, inplace = True)
all_features = titanic.drop("Survived",axis=1)

Targete = titanic["Survived"]

X_train,X_test,y_train,y_test = train_test_split(all_features,Targete,test_size=0.3,random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
titanic.head()
model = LogisticRegression()

model.fit(X_train,y_train)

prediction_lr=model.predict(X_test)

Log_acc = round(accuracy_score(prediction_lr,y_test)*100,2)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

Log_cv_acc=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Logistic Regression is',Log_acc)

print('The cross validated score for Logistic REgression is:',round(Log_cv_acc.mean()*100,2))
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

kfold = KFold(n_splits=10, random_state=22) 

result_knn=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the K Nearst Neighbors Classifier is',round(accuracy_score(Y_pred,y_test)*100,2))

print('The cross validated score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))
from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(X_train,y_train)

prediction_gnb=model.predict(X_test) 

nb_acc = round(accuracy_score(prediction_gnb,y_test)*100,2)

kfold = KFold(n_splits=12, random_state=22)

result_gnb=cross_val_score(model,all_features,Targete,cv=12,scoring='accuracy')



print('The accuracy of the Gaussian Naive Bayes Classifier is',nb_acc)

print('The cross validated score for Gaussian Naive Bayes classifier is:',round(result_gnb.mean()*100,2))
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

kfold = KFold(n_splits=5, random_state=22)

result_svm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Support Vector Machines Classifier is',acc_linear_svc)

print('The cross validated score for Support Vector Machines Classifier is:',round(result_svm.mean()*100,2))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_rm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Random Forest Classifier is',acc_random_forest)

print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test) 

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_rm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Random Forest Classifier is',acc_decision_tree)

print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, Log_acc, 

              acc_random_forest, nb_acc, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Model')

result_df.head(9)
predictions = cross_val_predict(random_forest, X_train, y_train, cv=3)

c_mat = confusion_matrix(y_train, predictions)

print(c_mat)
# we will see our confusion matrix in percentage.

sns.heatmap(c_mat/np.sum(c_mat), annot=True, 

            fmt='.2%', cmap='Blues')
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))
from sklearn.metrics import f1_score

f1_score(y_train, predictions)