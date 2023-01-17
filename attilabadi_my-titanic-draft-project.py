import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import joblib

from sklearn import tree



#Preparing model and train data

data = pd.read_csv("../input/titanic/train.csv", sep=',', header=0, encoding='UTF-8', engine='python')

# Replace missing values with a number

data['Embarked'].fillna("C", inplace=True)

# Replace age gaps using median 

median = data['Age'].median()

data['Age'].fillna(median, inplace=True)

# Detecting float values in Age column

cnt=0

for row in data['Age']:

    try:

        data.loc[cnt, 'Age']=float(row)

    except ValueError:

        print("Age Fuckup" + str(row))

    cnt+=1

# Detecting float values in Fare column

cnt=0

for row in data['Fare']:

    try:

        data.loc[cnt, 'Fare']=float(row)

    except ValueError:

        print("Fare Fuckup" + str(row))

    cnt+=1

# Translate gender values in Sex column (male = 1, female = 0)

cnt=0

for row in data['Sex']:

    try:

        data.loc[cnt, 'Sex']= 1 if row=="male" else 0

    except ValueError:

        print("Sex Fuckup" + str(row))

    cnt+=1

# Translate embarked values in Embarked column (Q = 2, S = 1, C = 0)

cnt=0

for row in data['Embarked']:

    try:

        if row=="Q" :

            data.loc[cnt, 'Embarked'] = 2

        elif row =="S":

            data.loc[cnt, 'Embarked'] = 1

        elif row =="C":

            data.loc[cnt, 'Embarked'] = 0

    except ValueError:

        print("Embarked Fuckup" + str(row))

    cnt+=1

    

X = data.drop(columns=["Name","Ticket","Cabin","Survived"])

y = data["Survived"]



model = DecisionTreeClassifier()

model.fit(X,y)



# Preparing test data

test_data = pd.read_csv("../input/titanic/test.csv", sep=',', header=0, encoding='UTF-8', engine='python')

# Replace missing values with a number

test_data['Embarked'].fillna("C", inplace=True)

# Replace age gaps using median 

median = test_data['Age'].median()

test_data['Age'].fillna(median, inplace=True)

# Detecting float values in Age column

cnt=0

for row in test_data['Age']:

    try:

        test_data.loc[cnt, 'Age']=float(row)

    except ValueError:

        print("Age Fuckup" + str(row))

    cnt+=1

# Replace fare gaps using median 

median_fare = test_data['Fare'].median()

test_data['Fare'].fillna(median_fare, inplace=True)    

# Detecting float values in Fare column

cnt=0

for row in test_data['Fare']:

    try:

        test_data.loc[cnt, 'Fare']=float(row)

    except ValueError:

        print("Fare Fuckup" + str(row))

    cnt+=1

# Translate gender values in Sex column (male = 1, female = 0)

cnt=0

for row in test_data['Sex']:

    try:

        test_data.loc[cnt, 'Sex']= 1 if row=="male" else 0

    except ValueError:

        print("Sex Fuckup" + str(row))

    cnt+=1

# Translate embarked values in Embarked column (Q = 2, S = 1, C = 0)

cnt=0

for row in test_data['Embarked']:

    try:

        if row=="Q" :

            test_data.loc[cnt, 'Embarked'] = 2

        elif row =="S":

            test_data.loc[cnt, 'Embarked'] = 1

        elif row =="C":

            test_data.loc[cnt, 'Embarked'] = 0

    except ValueError:

        print("Embarked Fuckup" + str(row))

    cnt+=1

    

X_test = test_data.drop(columns=["Name","Ticket","Cabin"])



# print(test_data.isnull().sum())
X_test

# X_test.describe()
X

# X.describe()
predictions = model.predict(X_test)

predictions
output=pd.DataFrame(data={"PassengerId":X_test["PassengerId"],"Survived":predictions}) 

output.to_csv(path_or_buf="./titanic_results.csv",index=False,quoting=3,sep=',')