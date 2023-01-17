# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(3)
train.describe()
# View group by survived

train.groupby(train['Survived']).mean()
# cabin transform

def cabin_value(cabin):

    if str(cabin) == 'nan':

        return str('0')

    else:

        return ord(cabin[0])-ord('A')+1



# transform gender to numeric

def sex_value(Sex):

    if Sex == 'male':

        return 1

    else:

        return 0



# Port

def embark_value(em):

    if em == 'C':

        return 1

    elif em == 'S':

        return 2

    elif em == 'Q':

        return 3

    else:

        return 0



# features function

def titan_feature(df):

    # fill NA age with mean value

    avg_age=df['Age'].mean()

    avg_fare=df['Fare'].mean()

    df['Age']=df['Age'].fillna(0)

    df['Fare']=df['Fare'].fillna(0)

    df['Child']=df['Age'].apply(lambda x:1 if x<18 else 0)

    df['Sex']=df['Sex'].apply(lambda x:sex_value(x))

    df['Cab']=df['Cabin'].apply(lambda x:cabin_value(x))

    df['Embark']=df['Embarked'].apply(lambda x:embark_value(x))

    df_features = df[['Pclass','Age','Child','SibSp','Parch','Fare','Sex','Cab','Embark']]

    return df_features
train_features = titan_feature(train)

train_label = train['Survived']
# split into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(titan_feature(train), train_label, test_size=0.3, random_state=0)
# Random Forest Modeling

clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=1)

clf = clf.fit(X_train, y_train)

# See training results of model

label_pred = clf.predict(X_test)

print(classification_report(y_test, label_pred))

print(confusion_matrix(y_test,label_pred))
test_features = titan_feature(test)

test_label = clf.predict(test_features)
#

#print("Checkinf for NaN and Inf")

#print("np.inf=", np.where(np.isnan(test['Fare'])))

#print("is.inf=", np.where(np.isinf(test['Fare'])))

#print("np.max=", np.max(abs(test['Fare'])))
test_id = test['PassengerId']

test_result = np.c_[test_id,test_label]
pd.DataFrame(test_result).to_csv('gender_submission.csv',header=(['PassengerId','Survived']), index=False)
pd.read_csv('gender_submission.csv')