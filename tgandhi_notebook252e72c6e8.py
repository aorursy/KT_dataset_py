import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

test["Survived"]="unk"

cols=test.columns.tolist()

cols=cols[:1]+cols[-1:]+cols[1:-1]

test=test[cols]

df=df.append(test,ignore_index=True)
df["Gender"]=df['Sex'].map( {'female':0, 'male':1} ).astype(int)
median_ages = np.zeros((2,3))

for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df["Age Fill"]=df["Age"]

for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'Age Fill'] = median_ages[i,j]

df['FamilySize'] = df['SibSp'] + df['Parch']

df['StrName']=df['Name'].str.split(", ").apply(lambda x: x[1])

df['Title']=df['StrName'].str.split(".").apply(lambda x: x[0])

df['HighClass']=np.where(df.Title.isin(['Mr','Mrs','Miss','Master','Mme','Ms','Mlle']),1,0)

cols=['Survived','Pclass','Gender', 'Age Fill']

train=df.ix[df['Survived']!='unk']

train=train[cols].astype('float64')

test_data=df.ix[df['Survived']=='unk']

del test_data['Survived']

cols=['Pclass','Gender']

test_data=test_data[cols].astype('float64')
from sklearn.ensemble import RandomForestClassifier 
forest = RandomForestClassifier(n_estimators = 50)

features=['Pclass', 'Gender']

forest = forest.fit(X=train[features],y=train['Survived'])

output = forest.predict(test_data)

final=pd.DataFrame({"PassengerId":pd.Series(test['PassengerId']),

      "Survived":pd.Series(output)})

final.to_csv("RandomForestOne",index=False)
final