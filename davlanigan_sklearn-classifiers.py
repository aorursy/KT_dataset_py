import numpy as np # linear algebra

from sklearn import preprocessing as prepr

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pylab import rcParams

rcParams['figure.figsize'] = 40, 40



df=pd.read_csv("/kaggle/input/titanic/train.csv")



## get titles

titles=[]

for name in df["Name"]:

    titles.append( name[name.find(",")+2:name.find(".")] )



df["Cabin"] = df["Cabin"].fillna("Z")

cab=[]

for el in df["Cabin"]:

    cab.append(el[0])



df["Title"]=titles

df["Cabin_Section"]=cab



print(" ")

print(" ")

df.head()

sns.catplot(x="Survived", y="Age",hue="Embarked", kind="swarm", data=df)

sns.catplot(x="Survived", y="Age",hue="Title", kind="swarm", data=df)

sns.catplot(x="Survived", y="Fare",hue="Sex", kind="swarm", data=df)

sns.catplot(x="Survived", y="Fare",hue="Cabin_Section", kind="swarm", data=df)
df=pd.read_csv("/kaggle/input/titanic/train.csv")

len_df=len(df)

num_females = len(df.where(df["Sex"]=="female",inplace=False).dropna())



df=df.where((df["Survived"]==1),inplace=False).dropna()

total_survived=len(df)

df=df.where(df["Sex"]=="female").dropna()

female_survived=len(df)





print( "total survived {}, female survived {}".format(total_survived,female_survived) )

print( "percent survivors that are female {}".format( round(female_survived/total_survived,2)*100 ) )

print( "percent of females who survived {}".format( round(female_survived/num_females,2)*100 ) )

print( "percent people who were female survivors {}".format( round(female_survived/len_df,2)*100 ) )

print( "percent people who were female {}".format( round(num_females/len_df,2)*100 ) )
#functions





def create_integer_labels(seq):

    uniq_ids=list(set(seq))

    uniq_ids.sort()

    for i,el in enumerate(seq): 

        seq[i]= uniq_ids.index(el)

    return seq



def prep_data(in_df):

    df=in_df.copy()

    ##get average age

    avg_age = round(df["Age"].mean(),0)

    avg_fare = round(df["Fare"].mean(),2)





    #Fill in missing data with appropriate values

    #For age add the average age



    df["Cabin"] = df["Cabin"].fillna("Z")

    df["Embarked"] = df["Embarked"].fillna("P")

    df["Age"] = df["Age"].fillna(avg_age)

    df["Fare"] = df["Fare"].fillna(avg_fare)



    null_data = df[df.isnull().any(axis=1)]

    print(null_data)



    df = df.dropna()



    #get cab sections

    cab=[]

    for el in df["Cabin"]:

        cab.append(el[0])



    ## get titles

    titles=[]

    for name in df["Name"]:

        titles.append( name[name.find(",")+2:name.find(".")] )



    titles=create_integer_labels(titles)

    cab=create_integer_labels(cab)

    embarked=create_integer_labels( list(df["Embarked"]) )

    sex=create_integer_labels( list(df["Sex"]) )



    df["Cabin_Section"]=cab

    df["Title"]=titles

    df["Embarked"]=embarked

    df["Sex"]=sex

    

    print(len(in_df))

    print(len(df))

    return df
from sklearn import svm, datasets



train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")



train_df=prep_data(train_df)

test_df=prep_data(test_df)

y=train_df["Survived"].values

#train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin"])

copy_test_df=test_df.copy()

#test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin"])



print(test_df.head())



X = train_df.values

Xt = test_df.values



X = prepr.normalize(X)

Xt = prepr.normalize(Xt) 



n=len(X)

c=[]

for x in range(100):

    split=n-267

    np.random.shuffle(X)

    X_train, X_test, y_train,y_test = X[:split],X[split:],y[:split],y[split:]





    ### linear

    svc=svm.SVC(kernel="rbf",C=97)

    svc.fit(X_train,y_train)



    out=svc.predict(X_test)



    correct=0

    for i,el in enumerate(out):

        if el==y_test[i]:

            correct+=1

    c.append(1-round(correct/n,4))

            

print(c)



sns.distplot(c,bins=5)

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")



train_df=prep_data(train_df)

test_df=prep_data(test_df)

y=train_df["Survived"].values

#train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin"])

copy_test_df=test_df.copy()

#test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin"])



print(train_df.head())

print(train_df.values)



X=prepr.normalize(train_df.values)



print(X)



split=len(X)-267

X_train, X_test, y_train,y_test = X[:split],X[split:],y[:split],y[split:]



clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train,y_train)

out=clf.predict(X_test)



out

y_test



correct=0

for i,el in enumerate(out):

    if el==y_test[i]:

        correct+=1



print("correct: {}".format( round( (correct/float(len(out)))*100,0 ) ) )
from sklearn.linear_model import LogisticRegression



train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")



train_df=prep_data(train_df)

test_df=prep_data(test_df)

y=train_df["Survived"].values

#train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

train_df=train_df.drop(columns=["PassengerId","Survived","Name","Ticket","Parch","Cabin"])

copy_test_df=test_df.copy()

#test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin","Cabin_Section","Title","SibSp","Age","Embarked","Fare"])

test_df=test_df.drop(columns=["PassengerId","Name","Ticket","Parch","Cabin"])



X = train_df.values

Xt = test_df.values



X = prepr.normalize(X)

Xt = prepr.normalize(Xt)



split=len(X)-267

X_train, X_test, y_train,y_test = X[:split],X[split:],y[:split],y[split:]



lg=LogisticRegression(C=1e5)



lg.fit(X_train,y_train)



out=lg.predict(X_test)





correct=0

for i,el in enumerate(out):

    if el==y_test[i]:

        correct+=1



print("correct: {}".format( round( (correct/float(len(out)))*100,0 ) ) )
#out=svc.predict(Xt)

#out=clf.predict(Xt)

out=lg.predict(Xt)



copy_test_df=copy_test_df.drop(columns=["Pclass","Name","Sex","Age","SibSp","Title","Parch","Cabin_Section","Ticket","Fare","Cabin","Embarked"])

copy_test_df["Survived"]=out



print(copy_test_df.head()) 

print(len(copy_test_df))

copy_test_df.to_csv("submission.csv",index=False)

from IPython.display import FileLink

FileLink(r'submission.csv')