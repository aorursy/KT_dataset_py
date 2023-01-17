# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



import seaborn as sns

from collections import Counter



import warnings



warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_PassengerId = train_df['PassengerId']



test_PassengerId = test_df["PassengerId"]

train_df.head()
train_df.describe().T
train_df.columns
train_df.index
train_df.shape
train_df.info()
def bar_plot(variable):

    """ input : variable (sex,survived ..)

        output : bar plot & value counts

    """

    # get feature

    var = train_df[variable]

    

    # number of categorical variable

    val = var.value_counts()

    

    # visualise of variable

    plt.figure(figsize = (8,3))

    plt.bar(val.index,val)

    #plt.xticks(val.index,val.values)

    plt.title(variable)

    plt.ylabel('Frequency')

    plt.show()

    # print variable and values

    print('{} : \n {}'.format(variable,val))

    

    

    
variables = ['Survived','Sex','Pclass','Embarked','SibSp','Parch']



for var in variables:

    bar_plot(var)
variables2 = ['Cabin','Name','Ticket']



for var2 in variables2:

    print('{} \n'.format(train_df[var2].value_counts()))
def hist_plot(numeric):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[numeric])

    plt.xlabel(numeric)

    plt.ylabel('Frequency')

    plt.title('{} distribution with histogram plot'.format(numeric))

    plt.show()
numericalvar = ['Fare','Age']

for num in numericalvar:

    hist_plot(num)
# Pclass - Survived



# Dataframe of Pclass vs Survived

Pclass_Survived = train_df[['Pclass','Survived']]



# Group by survived mean with pclass and set new index(0,1,...)

P_S=Pclass_Survived.groupby(['Pclass'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)



P_S
plt.bar(P_S.Pclass,P_S.Survived);

plt.xlabel('Pclass')

plt.ylabel('Survived rate ');

plt.title('Survived rate of Passengers with Pclass');
# Sex - Survived



# Dataframe of Sex vs Survived

Sex_Survived = train_df[['Sex','Survived']]



# Group by survived mean with pclass and set new index(0,1,...)

S_S = Sex_Survived.groupby(['Sex'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)



S_S
plt.bar(S_S.Sex,S_S.Survived);

plt.xlabel('Gender')

plt.ylabel('Survived rate')

plt.title('Survived rate of Passengers with Gender');
# SibSp - Survived



# Dataframe of Pclass vs Survived

SibSp_Survived = train_df[['SibSp','Survived']]



# Group by survived mean with pclass and set new index(0,1,...)

sib_sp = SibSp_Survived.groupby(['SibSp'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)



sib_sp
plt.bar(sib_sp.SibSp,sib_sp.Survived);

plt.xlabel('SibSp')

plt.ylabel('Survived rate')

plt.title('Survived rate of Passengers with SibSp');
# Parch - Survived



# Dataframe of Pclass vs Survived

Parch_Survived = train_df[['Parch','Survived']]



# Group by survived mean with pclass and set new index(0,1,...)

p_s = Parch_Survived.groupby(['Parch'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)



p_s
plt.bar(p_s.Parch,p_s.Survived);

plt.xlabel('Parch')

plt.ylabel('Survived rate')

plt.title('Survived rate of Passengers with Parch');
# Fare - Survived



# Dataframe of Pclass vs Survived

Fare_Survived = train_df[['Fare','Survived']]



# Group by survived mean with pclass and set new index(0,1,...)

f_s = Fare_Survived.groupby(['Fare'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)



f_s
plt.plot(f_s.Survived,f_s.Fare);

plt.xlabel('Tickets Pay')

plt.ylabel('Survived rate')

plt.title('Survived rate of Passengers with Values of Tickets');
# Detect the outliers values and drop the values



def outliers(data,features):

    

    outlier_values = []

    

    for i in features:

    # Q1 Outliers

        Q1 = np.percentile(data[i],25)

    # Q3 Outliers

        Q3 = np.percentile(data[i],75)

    # IQR 

        IQR = (Q3 - Q1)

    # outliers step

        step = (IQR * 1.5)

    # outliers values

        outlier = data[(data[i] < (Q1-step)) | (data[i] > (Q3+step))].index

        

        outlier_values.extend(outlier)

        

    outlier_values = Counter(outlier_values)

    clear_outlier = list(i for i,j in outlier_values.items() if j>2)

    

    return clear_outlier  

        

    

features = ['Age','SibSp','Parch','Fare']



train_df.loc[outliers(train_df,features)]
# now drop wir the outliers and reset index from 0,1,2,3....



train_df = train_df.drop(outliers(train_df,features),axis = 0).reset_index(drop = True)



train_df
# Find Missing Values





data = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)



data.info()
data.isnull().any()
data.columns[data.isnull().any()]
data.isnull().sum()
# Fill Missing Values 



"""  Fill Embarked and Fare"""



# Embarked data where nan is

data[data['Embarked'].isnull()]
# Analyse with Pclass

data[data['Pclass']==1].Embarked.value_counts()
# Analyse with Fare

data[ (data['Fare'] > 79 ) & (data['Fare'] < 81)].Embarked.value_counts()
# of the analyse fill wir with C (6)



data['Embarked'] = data['Embarked'].fillna('C')



data['Embarked'].isnull().sum()
""" Fill Fare """



data[data['Fare'].isnull()]
# Analyse with Pclass



data[data['Pclass'] == 3].Fare.value_counts()
# Anlayse with Embarked



data[data['Embarked'] == 'S'].Fare.value_counts()
data[(data['Pclass'] == 3) & (data['Embarked'] == 'S')].Fare.value_counts().head(50)
# the first 50 Passengers of Pclass (3) and Embarked(S) is more then other Passengers



mean_fare = np.mean(data[(data['Pclass'] == 3) & (data['Embarked'] == 'S')].Fare.value_counts().head(50))



mean_fare
data['Fare'] = data['Fare'].fillna(mean_fare)



data['Fare'].isnull().sum()
list1 = ["SibSp", "Parch", "Age" ,"Fare","Survived"]



plt.figure(figsize = (10,9))

sns.heatmap(train_df[list1].corr(),annot = True,fmt = ".2f");
sns.factorplot('SibSp','Survived',data = data,kind = 'bar',size = 5);
sns.factorplot('Parch','Survived',data = train_df,kind = 'bar');
sns.factorplot('Pclass','Survived',data = train_df,kind = 'bar');
g = sns.FacetGrid(data = train_df,col = 'Survived')

g.map(sns.distplot,'Age');
# Age distrubitions

sns.distplot(train_df['Age']);
# make one facegrid



g = sns.FacetGrid(data = train_df, col = 'Survived', row = 'Pclass')



# mapping with plotting



g.map(plt.hist,'Age');
g = sns.FacetGrid(train_df,row = 'Embarked')



g.map(sns.pointplot,'Pclass','Survived','Sex')

g.add_legend();
g = sns.FacetGrid(train_df, col = 'Survived',row = 'Embarked')

g.add_legend()

g.map(sns.barplot,'Sex','Fare');

# age nan values



data[data['Age'].isnull()]
# analysing age with sex



sns.boxplot('Sex','Age',data = data);
# Analysing age with SibSp



sns.boxplot('SibSp','Age',data=data);
# age of sex with sibsp

s_s_a = data[['Sex','SibSp','Age']]

s_s_a
df = s_s_a.set_index(['SibSp','Sex'])

medians_age = df[['Age']].groupby(df.index).median()

medians_age
df
for i in df[['Age']].groupby(df.index).median()['Age']:

    print(i)
for i,j in medians_age.index:

    print(i,j)
arrays= [[i for i,j in medians_age.index],[j for i,j in medians_age.index]]

index = pd.MultiIndex.from_arrays(arrays,names = ('SibSp','Sex'))

df2 = pd.DataFrame({'Age_median': [i for i in df[['Age']].groupby(df.index).median()['Age']]},index = index)

df2
(s_s_a['Sex'].groupby(s_s_a['SibSp']).value_counts())
# Analysing age with Parch

sns.boxplot('Parch','Age',data = data);
s_p_a = train_df[['Sex','Parch','Age']].set_index(['Parch','Sex'])



s_p_a
medians_age1 = s_p_a[['Age']].groupby(s_p_a.index).median()

medians_age1
arrays1= [[i for i,j in medians_age1.index],[j for i,j in medians_age1.index]]

index1 = pd.MultiIndex.from_arrays(arrays1,names = ('Parch','Sex'))

df3 = pd.DataFrame({'Age_median': [i for i in s_p_a[['Age']].groupby(s_p_a.index).median()['Age']]},index = index1)

df3
# Analyise age with Embarked



sns.boxplot('Embarked','Age',hue = 'Sex',data = data);
data[data['Age'].isnull()]
# Analyse age with ['SibSp','Parch','Embarked']

data2 = data[['Sex','SibSp','Pclass','Parch','Embarked','Age']].dropna().set_index(['SibSp','Pclass','Parch','Sex','Embarked'])

data2
medians_age2 = data2[['Age']].groupby(data2.index).median()

medians_age2.head(10)
# Group the features with age



arrays2= ([[i for i,j,k,x,a in medians_age2.index],[j for i,j,k,x,a in medians_age2.index],

           [k for i,j,k,x,a in medians_age2.index],[x for i,j,k,x,a in medians_age2.index],

            [a for i,j,k,x,a in medians_age2.index]])

index2 = pd.MultiIndex.from_arrays(arrays2,names = ('SibSp','Pclass','Parch','Sex','Embarked'))

df4 = pd.DataFrame({'Age_median': [i for i in data2[['Age']].groupby(data2.index).median()['Age']]},index = index2)

df4.head(10)
for i,j,k,x,a in df4.index:

    print(i,j,k,x,a)
for i,j,k,x in data[data['Age'].isnull()][['SibSp','Parch','Sex','Embarked']].values:

    print(i,j,k,x)
data[data['Age'].isnull()]
# Nan's index number

index_nan_age = list(data["Age"][data["Age"].isnull()].index)



# Nans index numbers Sibsp,Parch,Pclass,Embarked,Sex is equals with df4 or others features

# than giv we the median of age



for i in index_nan_age:

    age_pred = (data["Age"][((data["SibSp"] == data.iloc[i]["SibSp"]) &

                                 (data["Parch"] == data.iloc[i]["Parch"])& 

                                 (data["Pclass"] == data.iloc[i]["Pclass"])&

                                 (data["Embarked"] == data.iloc[i]["Embarked"])&

                                  (data["Sex"] == data.iloc[i]["Sex"]))].median())

    age_med = data["Age"].median()

    if not np.isnan(age_pred):

        data["Age"].iloc[i] = age_pred

    else:

        data["Age"].iloc[i] = age_med

age_med
data[data['Ticket']=='W./C. 6607']


data.iloc[index_nan_age].head(50)
# Names of train data

data['Name'].head()
# Names analyse with Mr. , Miss , Mrs, ....

names_titles = [j[0].split() for j in [i.split('.') for i in data['Name']]]



titles = [title[1] for title in names_titles]



titles[:10]
# count plot of titles

plt.figure(figsize=(15,8))

g = sns.countplot(titles)

g.set_xticklabels(g.get_xticklabels(), rotation=45);
# titles anlayse as first four(Mr,Mrs,Miss,Master) and others



others = list(set([e for e in titles if e not in ('Mr','Mrs','Miss','Master' )]))



others
# give the data title values as title



data['title'] = titles



data.head()
# replace others in data

def replace(data):

    for i in data['title']:

        if i in others:

            data['title']=data['title'].replace(i,'other')

            

replace(data)
# barplot of title



sns.barplot(data['title'],data['Survived']);
# Now we can the name-columns delete



data.drop(['Name'],axis=1,inplace=True)



data.head()
# Now can we titles encode

data = pd.get_dummies(data,columns=['title'])



data.head()
# Now count the persons together woth family and set as family size

# Self person is one family with 1 count



self_person = 1



data['family_size'] = data['SibSp'] + data['Parch'] + self_person



data.head()
# Family size with Survived



sns.barplot(data['family_size'],data['Survived']);
# Embarked encode



sns.countplot(data['Embarked']);
data = pd.get_dummies(data,columns=['Embarked'])



data.head()
# Value count of Ticket



data['Ticket'].value_counts()
# Number values of tickets delete



tickets = []



for i in data['Ticket']:

    if not i.isdigit():

        tickets.append(i.replace('.',"").replace('/','').strip().split(' ')[0])

    else:

        tickets.append('x')

        
data['Ticket'] = tickets



data['Ticket'].head(10)
data.head()
# get dimmues of ticket



data = pd.get_dummies(data,columns=['Ticket'])



data.head()
# Pclass  visualisation



print(data['Pclass'].value_counts())



sns.countplot(data['Pclass']);
# Get dummies

data = pd.get_dummies(data,columns=['Pclass'])



data.head()
# Get dummies of Sex columns



data = pd.get_dummies(data,columns=['Sex'])



data.head()
# We can drop the id and cabin



data.drop(['PassengerId','Cabin'],axis=1,inplace = True)



data.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# our data

data.head()
data.shape
test = data[881:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = data[:881]
test.shape
# Dependet feature

y = train[['Survived']]



# Independet features

x = train.drop(['Survived'],axis = 1)
# Now train tes split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42)


print("X_train",len(x_train))

print("X_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
# parameters for cros validation

random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
# Visualisation

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores");
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test),y_test))
test
test_survived = pd.Series(votingC.predict(test), name = "Survived")

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic_data.csv", index = False)