#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
%matplotlib inline
#mount drive
from google.colab import drive
drive.mount('/content/drive')
#import datasets
train_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Titanic/train.csv')
test_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Titanic/test.csv')
#for the purpose of running some operations on both datasets simultaneously
combine = [train_df,test_df]
#a peek at the data
train_df.head()
test_df.head()
train_df.info()
print('_'*40)
test_df.info()
train_df.shape
train_df.describe()
print(train_df.isnull().sum())
sns.heatmap(train_df.isnull(), yticklabels=False,cbar=False, cmap ='viridis')
sns.heatmap(test_df.isnull(), yticklabels=False,cbar=False, cmap ='viridis')
for df in combine:
  df.drop(['Cabin','Ticket','Name'], axis=1, inplace=True)
train_df.head()
#exploring the age distribution of passengers
plt.figure(figsize=(9,6))
ax = sns.distplot(train_df['Age'].dropna(),kde=False,bins=30)
ax.grid(False)
#lets drop the rows missing the Age data and take a closer look at the column
train1 = train_df.drop(train_df[train_df['Age'].isnull()].index)
train1.shape
#group the passengers ages by their classes.
grouped = train1['Age'].groupby(train1['Pclass'])
grouped.describe()
#compare the mean age of each class using one-way ANOVA
stats.f_oneway(grouped.get_group(1),
               grouped.get_group(2),
               grouped.get_group(3))
#a closer look at the distribution within the passenger classes
plt.figure(figsize=(9,6))
sns.set_style('whitegrid')
sns.boxplot(x='Pclass', y='Age', data=train_df, palette="Set1")
def input_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else: 
            return 24
    else:
        return Age
for df in combine: 
  df['Age'] = df[['Age','Pclass']].apply(input_age, axis =1)
train_df.dropna(axis=0, inplace=True)
print(train_df.isnull().sum())
train_df.shape
sns.set_style('whitegrid')
sns.factorplot('Sex',data=train_df,kind='count', palette='Set1')
def sex_age(passenger):
    age=passenger['Age']
    sex=passenger['Sex']
    return 'child' if age < 16 else sex

for df in combine: 
  df['Person'] = df.apply(sex_age,axis=1)
sns.factorplot('Person',data=train_df,kind='count', palette='Set1')
sns.factorplot('Pclass',data=train_df,hue= 'Person',kind='count', palette="Set1")
for df in combine:
  df['FamilySize'] = df.SibSp + df.Parch
def IsAlone(passenger):
  is_alone = passenger['FamilySize']
  return "Yes" if is_alone ==0 else "No"

for df in combine: 
  df['Alone'] = df.apply(IsAlone,axis=1)
train_df.head()
for df in combine:
  df.drop(['SibSp','Parch',], axis=1, inplace=True)
sns.catplot('Alone',data=train_df,hue = 'Person', kind='count',palette="Set1")
sns.catplot('Pclass',data=train_df,hue = 'Alone', kind='count',palette="Set2")
sns.catplot('Embarked',data=train_df,hue='Pclass',kind='count', palette="Set1")
sns.factorplot('Pclass','Survived',hue='Person',data=train_df, palette='Set1')
sns.factorplot('Alone','Survived',data=train_df)
sns.lmplot('Age','Survived',hue='Pclass',data=train_df, palette="Set1")
sns.factorplot('Survived', 'Embarked', data=train_df)
sns.factorplot('Survived', 'Fare', data=train_df)
# Creating different datasets for survivors and non-survivors to enable comparison
survivors_df = train_df[train_df['Survived'] == 1]
nonsurvivors_df = train_df[train_df['Survived'] == 0]

print('The number of survivors is:',len(survivors_df))
print('The number of non-survivors is:',len(nonsurvivors_df))
#survivors
women = train_df.loc[train_df.Person == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_df.loc[train_df.Person == 'male']["Survived"]
rate_men = sum(men)/len(men)

children = train_df.loc[train_df.Person == 'child']["Survived"]
rate_child = sum(children)/len(children)
print("% of women who survived:", rate_women*100, "%")
print("% of men who survived:", rate_men*100, "%")
print("% of children who survived:", rate_child*100, "%")
print("% of survivors that were women:",(sum(women)/len(survivors_df))*100,"%")
print("% of survivors that were men:",(sum(men)/len(survivors_df))*100,"%")
print("% of survivors that were children:",(sum(children)/len(survivors_df))*100,"%")
sns.catplot('Person',data=survivors_df,kind='count',order = ['male','female','child'], palette='Set1')
sns.catplot('Person',data=nonsurvivors_df,kind='count', order = ['female','male','child'], palette='Set1')
test_df.head()
print(test_df.isnull().sum())
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for df in combine:
    df.loc[df['Fare'] <= 7.896, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.896) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
    
train_df.head(5)
test_df.head()
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Alone'] = test_df['Alone'].map( {'No': 0, 'Yes': 1} ).astype(int)
test_df['Person'] = test_df['Person'].map( {'male': 0, 'female': 1, 'child': 2} ).astype(int)

train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df['Alone'] = train_df['Alone'].map( {'No': 0, 'Yes': 1} ).astype(int)
train_df['Person'] = train_df['Person'].map( {'male': 0, 'female': 1, 'child': 2} ).astype(int)

train_df.tail()
test_df.head()
y_train = train_df["Survived"]
features = ["Pclass", "Person", "Alone", "Embarked", "Fare","Age"]
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
X_train = train_df[features]
X_test = test_df[features]

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


#X_train = std_scale.transform(X_train)
#X_test = std_scale.transform(X_test)
classifiers = []

model1 = LogisticRegression()
classifiers.append(model1)

model2 = svm.SVC()
classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()
classifiers.append(model3)

model4 = RandomForestClassifier()
classifiers.append(model4)

model5 = KNeighborsClassifier()
classifiers.append(model5)
# make predictions and score the performance of each model using the training data
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = round(clf.score(X_train, y_train) * 100, 2)
    print("Accuracy of %s is %s"%(clf, acc))
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('/content/drive/My Drive/Colab Notebooks/Titanic/titanic_submission.csv', index=False)
print("Your submission was successfully saved!")