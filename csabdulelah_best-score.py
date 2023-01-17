import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head(10)
test.head(10)
train.tail()
test.tail()
train.dtypes
train.isnull().sum()
test.isnull().sum()
fig,ax = plt.subplots(ncols=2,figsize=(15,7))

# train heatmap
sns.heatmap(data=train.isnull(),ax=ax[0])
ax[0].set_title('Train null Data')

#test heatmap
sns.heatmap(data=test.isnull(),ax=ax[1])
ax[1].set_title('Test null Data')
pd.DataFrame(train.groupby('Pclass')['Age'].mean())
def fill_age(df):
    Age = df[0]
    
    Pclass = df[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass ==2:
            return 29
        else:
            return 25
    return Age
train.Age = train.apply(lambda row: fill_age(row[['Age','Pclass']]), axis=1)
test.Age = test.apply(lambda row: fill_age(row[['Age','Pclass']]), axis=1)

def fill_cabin(ser):
    if pd.isnull(ser):
        return 0
    else:
        return 1
train.Cabin = train.apply(lambda row :fill_cabin(row.Cabin ) , axis = 1)
test.Cabin = test.apply(lambda row :fill_cabin(row.Cabin ) , axis = 1)

train.Embarked.fillna(train.Embarked.mode()[0],inplace=True)
test.Fare.fillna(test[test['Pclass'] ==3 ].Fare.mean(),inplace=True)
train.isnull().sum()
test.isnull().sum()
# we want to see the number of Survivors for males, females and children

sns.catplot(x='Survived', data=train, hue="Sex", kind='count', legend_out=False)


plt.tight_layout()
plt.show()

# we want to see the relationship between sale_price and year_built

sns.catplot(x='Sex', data=train, hue="Pclass", kind='count', legend_out=False)



plt.tight_layout()
plt.show()
# The rate of surviveal in each gender and in each class 
# we see alwayes  women live more than man lol :)
g = sns.catplot(x="Sex", y="Survived", col="Pclass",
                data=train, saturation=.5,
                kind="bar", ci=None, aspect=1, height=5)
(g.set_axis_labels("", "Survival Rate")
  .set_xticklabels(["Men", "Women", "Children"])
  .set_titles("{col_var} {col_name}")
  .set(ylim=(0, 1))
  .despine(left=True))
plt.tight_layout()
plt.show()
# Comparing between survived people by ages
grid = sns.FacetGrid(train, col='Survived',hue="Sex", height=6);
grid.map(plt.hist, 'Age', bins=20);
plt.legend(loc = 'best')
plt.show()
train = pd.get_dummies(train, columns=['Sex','Embarked'] ,drop_first=True)
test  = pd.get_dummies(test, columns=['Sex','Embarked'] ,drop_first=True)
train.corr()
features = ['Pclass','Age','SibSp','Parch','Fare','Cabin','Sex_male','Embarked_Q','Embarked_S']
X_train = train[features]
y_train = train['Survived']
X_test= test[features]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_test_s = ss.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10.0)
logreg.fit(X_train_s,y_train)
logreg.score(X_train_s,y_train)
log_pred = logreg.predict(X_test_s)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(random_state=2)
svm.fit(X_train_s,y_train)
svm.score(X_train_s,y_train)
svm_pred = svm.predict(X_test_s)
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = svm_pred
submission.to_csv('../input/titanic/submission_best_score.csv')
