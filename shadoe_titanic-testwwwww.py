

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import time    

from sklearn import metrics    

import pickle as pickle    

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import preprocessing

from sklearn import cross_validation
import re

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    return np.nan



def substrings_in_bool(big_string):

    if big_string.find('(') != -1:

            return 1

    return 0



def dropdup(big_string):

    if len(big_string) > 3:

            return big_string[0:2]

    return big_string





def str_to_int(big_string):

    try:

        return int(big_string)

    except:

        return 0

        

def clean_and_munge_data(df):

    #处理一下名字，生成Title字段

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    

    def replace_titles(x):

        title=x['Title']

        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

            return 'Mr'

        elif title in ['Master']:

            return 'Master'

        elif title in ['Countess', 'Mme','Mrs']:

            return 'Mrs'

        elif title in ['Mlle', 'Ms','Miss']:

            return 'Miss'

        elif title =='Dr':

            if x['Sex']=='Male':

                return 'Mr'

            else:

                return 'Mrs'

        elif title =='':

            if x['Sex']=='Male':

                return 'Master'

            else:

                return 'Miss'

        else:

            return title



    df['Title']=df.apply(replace_titles, axis=1)

    df['Lucky_baby'] =  np.where((df['Sex'] == 'male')&(df['Parch'] == 1),1,0)

    df[' Unlucky_baby'] =  np.where((df['Sex'] == 'male')&(df['Parch'] >= 2),1,0)

    df[' Mother'] =  np.where((df['Sex'] == 'female')&(df['Parch'] >= 2) ,1,0)

    df[' Unlucky_class3_women'] =  np.where((df['Sex'] == 'female')&(df['Fare'] >= 26)&(df['Pclass'] == 3) ,1,0)

    

    Cabin_list=['A', 'B', 'C', 'D', 'E', 'F','G']

    df['Cabin'] = df['Cabin'].astype(str)

    df['Cabin_class']=df['Cabin'].map(lambda x: substrings_in_string(x, Cabin_list))

    df['Cabin_num']=df['Cabin'].map(lambda s:re.findall(r"\d+\.?\d*",s) )

    df['Cabin_num']=df['Cabin_num'].astype(str).map(lambda s:s.replace('[','') )

    df['Cabin_num']=df['Cabin_num'].map(lambda s:s.replace('\'','') )

    df['Cabin_num']=df['Cabin_num'].map(lambda s:s.replace(']','') )

    #df['Cabin_num']=df['Cabin_num'].map(lambda s:s.fillna )

    df['Cabin_num']=df['Cabin_num'].map(lambda s:dropdup(s) )

    df['Cabin_num']=df['Cabin_num'].map(lambda s:str_to_int(s) )



    df['contain()']=df['Name'].map(lambda x: substrings_in_bool(x))

    #df['contain()']=df['contain()'].map(lambda x: substrings_in_bool(x))

    

    df['Family_Size']=df['SibSp']+df['Parch']





    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())

    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    





    df['AgeFill']=df['Age']

    mean_ages = np.zeros(4)

    mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())

    mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())

    mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())

    mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())

    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]

    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]



    df['AgeCat']=df['AgeFill']

    df.loc[ (df.AgeFill<=16) ,'AgeCat'] = 'child'

    df.loc[ (df.AgeFill>60),'AgeCat'] = 'elder'

    df.loc[ (df.AgeFill>16) & (df.AgeFill <=40) ,'AgeCat'] = 'adult'

    df.loc[ (df.AgeFill>40) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'



    df.Embarked = df.Embarked.fillna('S')





    df['Men_class']='Female'

    df.loc[ (df.Fare <= 8.05) & (df.Sex == "male") ,'Men_class'] = 'Poor'

    df.loc[ (df.Fare>8.05) & (df.Fare <=13) & (df.Sex == "male")  ,'Men_class'] = 'Low'

    df.loc[ (df.Fare>13) & (df.Fare <=26) & (df.Sex == "male")  ,'Men_class'] = 'Senior'

    df.loc[ (df.Fare>26) & (df.Fare <=52) & (df.Sex == "male")  ,'Men_class'] = 'High'

    df.loc[ (df.Fare >= 52)& (df.Sex == "male")  ,'Men_class'] = 'Noble'





    df['Dangerous_zone']= 0

    df.loc[ (df.Fare>7.5) & (df.Fare <=8.5)  & (df.Sex == "female") ,'Dangerous_zone'] = 1

    df.loc[ (df.Fare>11) & (df.Fare <=15)  & (df.Sex == "female") ,'Dangerous_zone'] = 1

    df.loc[ (df.Fare>24) & (df.Fare <=28) & (df.Sex == "female")  ,'Dangerous_zone'] = 1

    df.loc[ (df.Fare>50) & (df.Fare <=54) & (df.Sex == "female")  ,'Dangerous_zone'] = 1



    df = df.drop(['Name','Age','Cabin','Ticket'], axis=1) #remove Name,Age and PassengerId





    return df

if __name__ == '__main__':

    origin_data_train = pd.read_csv("../input/test.csv")

    test=clean_and_munge_data(origin_data_train)
df = df.fillna("N")

df['Pclass'] = df['Pclass'].astype(str)

df = pd.get_dummies(df)

df.describe()
test = test.fillna("N")

test['Pclass'] = test['Pclass'].astype(str)

test = pd.get_dummies(test)

test.describe()
from sklearn.model_selection import train_test_split

df1 = df.drop(['Survived'],axis = 1)

df2 = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(df1,df2, test_size=0.2, random_state=123)
df1
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.regularizers import l2, l1

from sklearn.preprocessing import StandardScaler



stdScaler = StandardScaler()

X_train_scaled = stdScaler.fit_transform(X_train)

X_test_scaled = stdScaler.transform(X_test)

model = Sequential()

#model.add(Dense(700, input_dim=7, init='normal', activation='relu'))

#model.add(Dropout(0.5))

model.add(Dense(2000, input_dim=42, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, init='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train_scaled, y_train, nb_epoch=100, batch_size=32)

result = model.predict(X_test_scaled)

rightnum = 0

for i in range(0, result.shape[0]):

    if result[i] >= 0.5:

        result[i] = 1

    else:

        result[i] = 0

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])
train_scaled = stdScaler.fit_transform(df1)

test_scaled = stdScaler.transform(test)

model.fit(train_scaled, df['Survived'], nb_epoch=100, batch_size=32, verbose=0)

predict_NN = model.predict(test_scaled)

print(predict_NN.shape)

for i in range(0, predict_NN.shape[0]):

    if predict_NN[i] >= 0.5:

        predict_NN[i] = 1

    else:

        predict_NN[i] = 0

        

predict_NN = predict_NN.reshape((predict_NN.shape[0]))

predict_NN = predict_NN.astype('int')

print(predict_NN.shape)

submission = pd.DataFrame({

        "PassengerId": origin_data_train["PassengerId"],

        "Survived": predict_NN

    })

submission
submission.to_csv("titanic_predict_NN.csv", index=False)