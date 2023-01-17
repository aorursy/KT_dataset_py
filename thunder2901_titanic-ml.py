import pandas as pd

df = pd.read_csv("../input/titanic/train.csv")

df.head()
df.columns
df_new = df.drop(['PassengerId','Name','Ticket','Cabin'],axis='columns')
df_new.head()
for col in df_new.columns:

    print(col,df_new[col].isnull().values.any())
age_median = df['Age'].median()

embarked_mode = df['Embarked'].mode()

print(embarked_mode)
df_new['Age'].fillna(age_median,inplace = True)

df_new['Embarked'].fillna('S', inplace=True)
df_new['Age'].isnull().values.any()
df_new['Embarked'].isnull().values.any()
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()

df_new['sex_en'] = le_sex.fit_transform(df_new['Sex'])

df_new.head()
X = df_new.drop('Sex',axis='columns')

X.head()
embarked_dummies = pd.get_dummies(X['Embarked'])
X = pd.concat([X,embarked_dummies],axis='columns')

X = X.drop('Embarked',axis='columns')
X.head()
features = ['Pclass','Age','SibSp','Parch','Fare','sex_en','C','Q','S']
y = X['Survived']
X = X[features]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

nb.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1)

lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)
lr.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

cm_lr = confusion_matrix(y_test,pred_lr)

sns.heatmap(cm_lr,annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')

print(cm_lr)
from sklearn.neighbors import KNeighborsClassifier

for n in range(4,20):

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train,y_train)

    #pred_knn = knn.predict(X_test)

    score  = knn.score(X_test,y_test)

    print("n =",n,", score =",score)
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(random_state=1)

dct.fit(X_train,y_train)

dct.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)
from sklearn.svm import SVC

sv = SVC(gamma='auto',random_state=1)

sv.fit(X_train,y_train)

sv.score(X_test,y_test)
from sklearn.ensemble import VotingClassifier



estimators = [('lrE',lr),('knnE',knn),('nbE',nb),('svE',sv),('rfE',rf)]



ens = VotingClassifier(estimators,voting='hard')

ens.fit(X_train,y_train)

ens.score(X_test,y_test)
ens_full_data = VotingClassifier(estimators,voting='hard')

ens_full_data.fit(X,y)
path = '../input/titanic/test.csv'

test_data = pd.read_csv(path)

test = test_data.copy()
test_data.head()
test_data = test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis='columns')
test_data.head(3)
test_data['sex_en'] = le_sex.fit_transform(test_data['Sex'])

test_data.head()
test_data = test_data.drop('Sex',axis='columns')
embarked_dummies = pd.get_dummies(test_data['Embarked'])
test_data = pd.concat([test_data,embarked_dummies],axis='columns')

test_data = test_data.drop('Embarked',axis='columns')

test_data.head(3)
for col in test_data.columns:

    print(col,test_data[col].isnull().values.any())
test_data['Age'].fillna(age_median,inplace = True)
fare_median = df_new['Fare'].median()

test_data['Fare'].fillna(fare_median,inplace=True)
predictions = ens_full_data.predict(test_data)
print(predictions)
output = pd.DataFrame({'PassengerId':test["PassengerId"],'Survived':predictions})

output.to_csv('submission.csv',index=False)