# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from matplotlib.pyplot import plot

from sklearn.covariance import EllipticEnvelope







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



%matplotlib inline



# Any results you write to the current directory are saved as output.
def processing(df,Y):



    #Dealing with Nan Values

    for l in df[df.Age.isnull()].index:

        p_class = df[df.index==l].Pclass.values

        new_age = np.mean(df[df.Pclass==p_class[0]].Age)

        df.at[l,'Age']=new_age

        

    for l in df[df.Fare.isnull()].index:

        df.at[l,'Fare']=np.mean(df.Fare)

        

    #Adding features

    df['Is_alone']=1*((df['SibSp']==0)& (df['Parch']==0))

    

    #Name processing

    title_list=[]

    for name in df.Name.values:

        i=0

        title=''

        while name[i]!=',':

            i=i+1

        j=i+2

        while name[j]!='.':

            title=title+name[j]

            j=j+1

        title_list.append(title)

    df['Title']=title_list

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    df['Title'] = df['Title'].map(title_mapping)

    df['Title'] = df['Title'].fillna(0)

    

    #Cabins processing

    cabin_list=[]

    for k in df.index:

        if df.Cabin.isnull()[k]:

            cabin_list.append('X')

        else:

            cabin_list.append(df.Cabin[k][0])

    df['Cabin']=cabin_list

    #Mapping Cabins

    cabin_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F" : 6, "G" : 7 , "T" :8 , "X":9 }

    df['Cabin'] = df['Cabin'].map(cabin_mapping)

    

    #Mapping Sex

    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping Embarked

    df['Embarked'] = df['Embarked'].fillna('S')

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Feature selection

    df=df.drop(['Ticket','Name'],axis=1)

    

    df_train=df[0:891]

    df_test=df[891:]

    

    #Outlier detection

    envelope=EllipticEnvelope(contamination=0.01)

    predict=envelope.fit_predict(df_train[['Fare','Age','SibSp','Parch']].values,Y)

    index_outlier = np.where(predict == -1)[0]+1

    df_train=df_train.drop(index=index_outlier, axis=0)

    Y=Y.drop(index=index_outlier,axis=0)

    

    return(df_train,df_test,Y)
df_train = pd.read_csv('../input/train.csv',index_col='PassengerId')

df_test = pd.read_csv('../input/test.csv',index_col='PassengerId')

X=df_train.drop(['Survived'],axis=1)

Y=df_train.Survived

ID = df_test.index



df=pd.concat([X,df_test],axis=0)

df_train,df_test,Y =processing(df,Y)

df_train.head()
X = scale(df_train)
c_list=np.geomspace(start=0.01,stop=5,num=25)    

Train_score=np.zeros(len(c_list))

Test_score=np.zeros(len(c_list))

n=50



for i in range(n):

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

    for j in range(len(c_list)):

        LogReg = LogisticRegression(penalty='l2',C=c_list[j],solver='liblinear')

        LogReg.fit(X_train,Y_train)

        Train_score[j] += LogReg.score(X_train,Y_train)

        Test_score[j] += LogReg.score(X_test,Y_test)

Train_score=Train_score/n

Test_score=Test_score/n

        

plt.plot(c_list,Train_score,label='train')

plt.plot(c_list,Test_score,label='test')

plt.legend()
c_list=np.geomspace(start=0.01,stop=5,num=25)    

Train_score=np.zeros(len(c_list))

Test_score=np.zeros(len(c_list))

n=75



for i in range(n):

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

    for j in range(len(c_list)):

        SVM = SVC(C=c_list[j],gamma='auto',kernel='rbf')

        SVM.fit(X_train,Y_train)

        Train_score[j] += SVM.score(X_train,Y_train)

        Test_score[j] += SVM.score(X_test,Y_test)

Train_score=Train_score/n

Test_score=Test_score/n

        

plt.plot(c_list,Train_score,label='train')

plt.plot(c_list,Test_score,label='test')

plt.legend()
print(max(Test_score))

print(c_list[np.argmax(Test_score)])
X2 = scale(df_test)



model = SVC(kernel='rbf',C=0.17,gamma='auto')

#model= LogisticRegression(C=1,solver='liblinear',penalty='l2')

#model = GradientBoostingClassifier(learning_rate=0.05,n_estimators=200,max_depth=2,min_samples_split=3)

model.fit(X,Y)

pred=model.predict(X2)
test_output = pd.DataFrame({"PassengerId" : ID,"Survived": pred})

test_output.set_index("PassengerId", inplace=True)

test_output.to_csv("prediction16.csv")