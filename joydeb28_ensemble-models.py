import pandas as pd



# Data visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Scalers

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle



#Common Model Helpers

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



def load_data():

    train = pd.read_csv("/kaggle/input/titanic/train.csv")

    test = pd.read_csv("/kaggle/input/titanic/test.csv")

    return train,test
def preprocessing(train,test):

    

    train_len = train.shape[0]

    comb = train.append(test)

    #comb.reset_index(inplace=True, drop=True)



    title_dictionary = {'Capt': 'Dr/Clergy/Mil','Col': 'Dr/Clergy/Mil','Major': 'Dr/Clergy/Mil','Jonkheer': 'Honorific','Don': 'Honorific',

                        'Dona': 'Honorific','Sir': 'Honorific','Dr': 'Dr/Clergy/Mil','Rev': 'Dr/Clergy/Mil','the Countess': 'Honorific',

                        'Mme': 'Mrs','Mlle': 'Miss','Ms': 'Mrs','Mr': 'Mr','Mrs': 'Mrs','Miss': 'Miss','Master': 'Master','Lady': 'Honorific'

    }

    comb['Title'] = comb['Name'].map(

        lambda name: name.split(',')[1].split('.')[0].strip())

    comb['Title'] = comb.Title.map(title_dictionary)

    keys = list(title_dictionary.keys())

    title_dict = {keys[item]:item+1 for item in range(len(keys))}

    comb['Title'] = comb.Title.map(title_dict)

    comb['Title'] = comb['Title'].fillna(0)

    train['Title'] = comb['Title'][:train_len]

    test['Title'] = comb['Title'][train_len:]

    

    comb['Embarked'].fillna(comb['Embarked'].mode()[0], inplace=True)

    embarked_mapping = {"S": 1, "C": 2, "Q": 3}

    comb['Embarked'] = comb['Embarked'].map(embarked_mapping)

    train['Embarked'] = comb['Embarked'][:train_len]

    test['Embarked'] = comb['Embarked'][train_len:]

    

    sex_mapping = {"male": 0, "female": 1}

    comb['Sex'] = comb['Sex'].map(sex_mapping)

    train['Sex'] = comb['Sex'][:train_len]

    test['Sex'] = comb['Sex'][train_len:]

    

    names = ['less2', '2-18', '18-35', '35-65', '65plus']

    comb['Age'].fillna(comb['Age'].mean(), inplace=True)

    comb['AgeBin'] = pd.qcut(comb['Age'],q = 5, labels = names)

    age_dummies = pd.get_dummies(comb['AgeBin'], prefix='AgeBin')

    comb = pd.concat([comb, age_dummies], axis=1)

    comb['AgeBin'] = comb['AgeBin'].map({'less2':0, '2-18':1, '18-35':2, '35-65':3, '65plus':4})

    train['AgeBin'] = comb['AgeBin'][:train_len]

    test['AgeBin'] = comb['AgeBin'][train_len:]

    

    

    comb['Fare'].fillna(comb['Fare'].mean(), inplace=True)

    comb['FareBin'] = pd.qcut(comb['Fare'], 5)

    label = LabelEncoder()

    comb['FareBin_Code'] = label.fit_transform(comb['FareBin'])

    train['FareBin_Code'] = comb['FareBin_Code'][:train_len]

    test['FareBin_Code'] = comb['FareBin_Code'][train_len:]    



    train.drop(['Cabin','Ticket','Age','Fare','Name'], inplace=True, axis=1)

    test.drop(['Cabin','Ticket','Age','Fare','Name'], inplace=True, axis=1)

        

    print(train.isna().sum())

    print(test.isna().sum())

    return train,test
train,test = load_data()

train,test = preprocessing(train,test)
train.head()
test.head()


xtrain = train.drop(['Survived', 'PassengerId'], axis=1)

ytrain = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size = 0.22, random_state = 0)
model = DecisionTreeClassifier()

# model = BaggingClassifier()

# model = GradientBoostingClassifier()

# model = AdaBoostClassifier()

# model = KNeighborsClassifier()

# model = MLPClassifier()

# model = RandomForestClassifier()

# model = ExtraTreesClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

acc = round(metrics.accuracy_score(y_pred, y_val) * 100, 2)

print(acc)
ids = test['PassengerId']

test = test.drop(['PassengerId'], axis=1)

pred = model.predict(test)

submission = pd.DataFrame({'PassengerId': ids,'Survived': pred})

submission.to_csv("/kaggle/working/submission.csv",index=False)