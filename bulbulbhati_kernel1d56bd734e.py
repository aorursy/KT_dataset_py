# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print('yolo')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print('fine')

    for filename in filenames:

        

        print(os.path.join(dirname, filename))

        print('work')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('hello world')
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data['Parch']
train_data.isnull().sum()
train_data['Parch'].value_counts()
women_alive = train_data.loc[train_data['Sex']=='female']['Survived']

print(sum(women_alive))

print(len(women_alive))

print(sum(women_alive)/len(women_alive))
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
train_data = train_data.drop(['Name','Ticket','Cabin'],axis=1)
import seaborn as sb
sb.boxplot(x='SibSp', y ='Age',data=train_data )
Parch_group = train_data.groupby(train_data['Parch'])

Parch_group.mean()

train_data.info()
def approx_age(cols):

    age =cols[0]

    parch = cols[1]

    if pd.isnull(age):

        if parch ==0:

            age = 32

        if parch ==1:

            age = 24

        if parch ==2:

            age = 17

        if parch ==3:

            age = 33

        if parch ==4:

            age = 44

        if parch ==5:

            age = 39

        if parch ==6:

            age = 43

    return age
train_data['Age'] = train_data[['Age','Parch']].apply(approx_age,axis=1)

train_data.info()
train_data.dropna(inplace=True)

train_data.reset_index(inplace = True, drop = True)

train_data.info()
train_data.head()

#covert object type to binary

from sklearn.preprocessing import LabelEncoder

label_encoder= LabelEncoder()

gender_encoded = label_encoder.fit_transform(train_data['Sex'])



gender_df = pd.DataFrame(gender_encoded,columns=['male_gender'])

gender_df.head()
#convert object type to different binary columns

embark_encoded = label_encoder.fit_transform(train_data['Embarked'])

embark_encoded[0:10]
train_data = train_data.drop(['Embarked'], axis=1)

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

x_test = test_data_new.drop(['Survived'], axis=1)



from sklearn.preprocessing import OneHotEncoder

binary_encoder = OneHotEncoder(categories='auto')

embarked_1hot = binary_encoder.fit_transform(embark_encoded.reshape(-1,1))

embarked_1hot_mat = embarked_1hot.toarray()

embarked_DF = pd.DataFrame(embarked_1hot_mat, columns = ['C', 'Q', 'S'])



train_data_new= pd.concat([train_data, gender_df, embarked_DF], axis=1).astype(float)

train_data_new.info()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict
LogReg = LogisticRegression(solver = 'liblinear')
X_train, X_test, y_train, y_test = train_test_split(train_data_new.drop('Survived', axis=1),

                                                   train_data_new['Survived'], test_size=0.2,

                                                   random_state=200)
print(X_train.shape)

print(y_train.shape)
LogReg.fit(X_train,y_train)
y_pred = LogReg.predict(X_test)