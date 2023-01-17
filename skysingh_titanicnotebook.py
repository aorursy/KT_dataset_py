import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve


df_train = pd.read_csv("../input/titanic/train.csv")
df_test    = pd.read_csv("../input/titanic/test.csv")
df_train.head()
df_train.info()
#Seeing the percentage of null values in each column
ls = []
for c in df_train.columns :
    na = df_train[c].isna().sum(axis = 0) / df_train.shape[0]
    ls.append((c , na*100))
per_df = pd.DataFrame(ls , columns = ['column','percentage_of_nullValue'])

per_df = per_df.sort_values(ascending=False , by ='percentage_of_nullValue' )
print(per_df)
sns.set()
(df_train.isnull()).sum(axis = 0).sort_values(axis = 0, ascending=True).plot(kind='barh',figsize=(10,7))
df_train.drop(columns = ['Cabin'],inplace = True)
df_train['Age'].describe()
ax = df_train['Age'].hist(bins = 15 ,color = '#F39FAA',density = True , alpha=0.9, figsize=(7, 5))
df_train["Age"].plot(kind='density', color='#0A561C')
plt.xlabel('Age')
plt.xlim(-5,85)
plt.figure()
print("Meadian of Age : {}".format(df_train['Age'].median()))
df_train['Age'].fillna( df_train['Age'].median() , inplace = True )
df_train['Embarked'].describe()
print("Ports of Embarkation are :\nS : {} ,C : {} , Q : {}".format('Southampton','Cherbourg','Queenstown'))
ax = sns.countplot(x="Embarked", data=df_train)
df_train['Embarked'].value_counts().idxmax()
df_train['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(),inplace = True)
df_train.drop(columns = ['Ticket','Name','PassengerId' ],inplace = True)
df_train['Family']=np.where((df_train["SibSp"]+df_train["Parch"])>0, 0, 1)
df_train.drop(columns = ['SibSp' , 'Parch' ] , inplace = True)
df_train.head()
sns.barplot('Family', 'Survived', data=df_train, color="#F695D1")
plt.show()
sns.pairplot(data = df_train , corner=True)
sns.barplot('Embarked', 'Survived', data=df_train, color="#F39FAA")
plt.show()
sns.barplot('Sex', 'Survived', data=df_train, color="#771953")
plt.show()
sns.barplot('Pclass', 'Survived', data=df_train, color="#73C7F0")
plt.show()
pclass_dum_train = pd.get_dummies(df_train['Pclass'])
pclass_dum_train.columns = ['Upper_Class','Middle_Class','Lower_Class']
pclass_dum_train.drop(columns = ['Lower_Class'],inplace = True)

df_train.drop(columns = ['Pclass'],inplace = True)
df_train = df_train.join(pclass_dum_train)

df_train.head()
df_train.Sex = df_train.Sex.replace('male',1)
df_train.Sex = df_train.Sex.replace('female',2)
df_train['Embarked'] = df_train['Embarked'].map({'S': 1, 'Q': 2, 'C': 3})
df_train.head()
df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
cols = [col for col in df_train.columns if col not in ["Survived"]]
X = df_train[cols]
X_train, X_test, y_train, y_test = train_test_split(X, df_train['Survived'], test_size=0.25)
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_train, y_train)
model.score(X_test,y_test)
train_y_pred = model.predict(X_train)
print("f1_score of training data : {}",format(f1_score(y_train,train_y_pred)))
test_y_pred = model.predict(X_test)
print("f1_score of training data : {}",format(f1_score(y_test,test_y_pred)))
from sklearn import metrics

fpr, tpr, thresholds = roc_curve(y_test, test_y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate(FP)')
plt.ylabel('True Positive Rate(TP)')
plt.title('ROC curve')
plt.show()

# Measuring the area under the curve  
print("AUC of the predictions: {}".format(metrics.auc(fpr, tpr)))

# Measuring the Accuracy Score
print("Accuracy score of the predictions: {}".format(metrics.accuracy_score(test_y_pred, y_test)))
df_test.drop(columns =['Cabin','Name','Ticket'] , inplace = True)
ls = []
for c in df_test.columns :
    na = df_test[c].isna().sum(axis = 0) / df_test.shape[0]
    ls.append((c , na*100))
per_df = pd.DataFrame(ls , columns = ['column','percentage_of_nullValue'])

per_df = per_df.sort_values(ascending=False , by ='percentage_of_nullValue' )
print(per_df)
df_test['Fare'].fillna( df_test['Fare'].median(),inplace = True )
df_test['Age'].fillna(df_test['Age'].median(skipna=True),inplace = True)
df_test['Family']=np.where((df_test["SibSp"]+df_test["Parch"])>0, 0, 1)
df_test.drop(columns = ['SibSp' , 'Parch' ] , inplace = True)
pclass_dum_test = pd.get_dummies(df_test['Pclass'])
pclass_dum_test.columns = ['Upper_Class','Middle_Class','Lower_Class']
pclass_dum_test.drop(columns = ['Lower_Class'],inplace = True)

df_test.drop(columns = ['Pclass'],inplace = True)
df_test = df_test.join(pclass_dum_train)

df_test.Sex = df_test.Sex.replace('male',1)
df_test.Sex = df_test.Sex.replace('female',2)

df_test['Embarked'] = df_test['Embarked'].map({'S': 1, 'Q': 2, 'C': 3})
cols = [col for col in df_test.columns if col not in ["PassengerId"]]
X = df_test[cols]
X = (X - X.min()) / (X.max() - X.min())
y_pred_test = model.predict(X)
final_test = pd.DataFrame()
final_test['Survived'] = y_pred_test
final_test['PassengerId'] = df_test['PassengerId']
final_test = final_test.astype({"Survived":int})

submission = final_test[['PassengerId','Survived']]
submission.to_csv('./submission.csv', index=False)


