#import libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import training file

train = pd.read_csv('../input/train.csv')
train.head(3)
#deal with missing data

train.isnull().sum()
#there are null values in age and embarked; fill in with mean & mode

train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

#too many null values in Cabin to be useful, so will drop
train.drop('Cabin', axis = 1, inplace = True)

#Passenger ID and Ticket don't have relevant data to this problem, so will drop as well
train.drop('PassengerId', axis = 1, inplace = True)
train.drop('Ticket', axis = 1, inplace = True)
#we should combine redundant columns when possible

train['Family'] = train['SibSp'] + train['Parch'] + 1
train.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
#pull out any relevant content that is hiding in the data
#looks like all names are formatted in the same way, with the title after last name and a comma
#titles could give us more information on the individual, so we'll pull it out

def split_name(x):
    return x.split(",")[1].split(" ")[1]

train['Title'] = train['Name'].apply(split_name)

#group rare names together
train.loc[train.groupby('Title').Title.transform('count').lt(45), 'Title'] = 'Rare'
train.drop('Name', axis = 1, inplace = True)
train.head()
plt.style.use('ggplot')
plt.title('Survival Rates by Sex')
avg_sex = train.groupby('Sex').Survived.mean().plot(kind = 'bar')
plt.title('Survival Rates by Embarked')
avg_class = train.groupby('Embarked').Survived.mean().plot(kind = 'bar')
plt.title('Survival Rates by Sex & Embarked')
avg_class = train.groupby(['Embarked', 'Sex']).Survived.mean().plot(kind = 'bar')
m = sns.FacetGrid(train, col="Pclass", row="Sex", hue = "Survived", margin_titles=True)
m = m.map(plt.hist, "Family").add_legend()

axes = m.axes.flatten()
axes[0].set_title("First Class")
axes[1].set_title("Second Class")
axes[2].set_title("Third Class")

sns.violinplot(x="Sex", y="Family", hue = 'Survived', split = True, data= train, linewidth = 1)
def combine_age(x):
    if x < 22:
        return "0 - 21"
    elif x < 29:
        return "22 - 28"
    elif x < 35:
        return "29 - 34"
    elif x < 50:
        return "35 - 50"
    else:
        return "51 - 80"

train['Age1'] = train['Age'].apply(combine_age)
plt.title('Survival Rates by Age Buckets')

#determined age buckets from age.describe()
avg_age = train.groupby('Age1').Survived.mean().plot(kind = 'bar')
sns.stripplot(x="Pclass", y="Age", data=train)

#Looks like class 1 and 2 were slightly higher ages than class 3 - might factor into influence of age on survival
avg_age = train.groupby(['Age1', 'Pclass']).Survived.mean().plot(kind = 'bar')
plt.title('Survival Rates by Age')
ungrouped_age = train.groupby('Age').Survived.mean().plot(kind = 'line', lw = 2, color = "purple")
plt.figure(figsize=(10,6))
train[train['Survived'] == 1]['Age'].hist(alpha = 0.5, color = 'blue', label = 'Survived')
train[train['Survived'] == 0]['Age'].hist(alpha = 0.5, color = 'red', label = 'Did not Survive')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Number of Survivors/Deaths')
def combine_fare(x):
    if x < 8:
        return "0 - 8"
    elif x < 14:
        return "8 - 14"
    elif x < 31:
        return "14 - 31"
    elif x < 150:
        return "31 - 150"
    else:
        return "151 - 520"

train['Fare1'] = train['Fare'].apply(combine_fare)
plt.title('Survival Rates by Fare Buckets')
avg_fare = train.groupby('Fare1').Survived.mean().plot(kind = 'bar')
sns.lmplot(x='Age',y='Fare',data=train,hue='Survived')
plt.title('test')
plt.title('Survival Rates by Title & Gender')
titles_by_gender = train.groupby(['Title', 'Sex']).Survived.mean().plot(kind = 'bar')
g = sns.factorplot("Title", "Age", "Survived", col="Pclass", data=train, kind="swarm")
train.head()
#need to transform categorical variables 

def transform_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2
    
train['Embarked'] = train['Embarked'].apply(transform_embarked)

def transform_fare(x):
    if x == '0 - 8':
        return 0
    elif x == '14 - 31':
        return 1
    elif x == '31 - 150':
        return 2
    elif x == '8 - 14':
        return 3
    else:
        return 4

train['Fare1'] = train['Fare1'].apply(transform_fare)

def transform_sex(x):
    if x == 'male':
        return 0
    else:
        return 1

train['Sex'] = train['Sex'].apply(transform_sex)

def transform_age(x):
    if x == '29 - 34':
        return 0
    elif x == '0 - 21':
        return 1
    elif x == '35 - 50':
        return 2
    elif x == '22 - 28':
        return 3
    else:
        return 4

train['Age1'] = train['Age1'].apply(transform_age)

final_train = pd.get_dummies(train,columns=['Title'],drop_first=True)
del final_train['Age']
del final_train['Fare']
final_train.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(final_train.drop('Survived',axis=1), 
                                                    final_train['Survived'], test_size=0.30)
# Logistic Regression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)

log_class = classification_report(y_test,log_predictions)
log_conf = confusion_matrix(y_test,log_predictions)
print(log_class, log_conf)
#Decision Tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dec_predictions = dtree.predict(X_test)

dectree_class = classification_report(y_test,dec_predictions)
dectree_conf = confusion_matrix(y_test,dec_predictions)
print(dectree_class, dectree_conf)
#Random Forest

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

rfc_class = classification_report(y_test,rfc_pred)
rfc_conf = confusion_matrix(y_test,rfc_pred)
print(rfc_class, rfc_conf)
# K Nearest Neighbors

#first, picking the best K value

error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure() 
plt.plot(range(1,50),error_rate,color='blue', marker='o', linewidth = 1)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

knn_class = classification_report(y_test,knn_pred)
knn_conf = confusion_matrix(y_test,knn_pred)
print(knn_class, knn_conf)
print("Logistic Regression \n \n", log_class, log_conf)
print("\n")
print("Decision Tree \n \n", dectree_class, dectree_conf)
print("\n")
print("Random Forest \n \n", rfc_class, rfc_conf)
print("\n")
print("K Nearest Neighbors \n \n", knn_class, knn_conf)

test_csv = pd.read_csv('../input/test.csv')
final_test = test_csv.drop(['Ticket', 'Cabin'], axis = 1)
final_test['Family'] = final_test['SibSp'] + final_test['Parch'] + 1
final_test.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

final_test['Title'] = final_test['Name'].apply(split_name)
final_test.loc[final_test.groupby('Title').Title.transform('count').lt(45), 'Title'] = 'Rare'
final_test.drop('Name', axis = 1, inplace = True)
final_test = pd.get_dummies(final_test,columns=['Title'],drop_first=True)

final_test['Embarked'] = final_test['Embarked'].apply(transform_embarked)

final_test['Fare1'] = final_test['Fare'].apply(combine_fare)
final_test['Fare1'] = final_test['Fare1'].apply(transform_fare)

final_test['Sex'] = final_test['Sex'].apply(transform_sex)

final_test['Age1'] = final_test['Age'].apply(combine_age)
final_test['Age1'] = final_test['Age1'].apply(transform_age)

del final_test['Age']
del final_test['Fare']
final_test.head()
#Logistic Regression has the best performance

dropped = final_test.drop('PassengerId', axis = 1)
log_predictions_final = logmodel.predict(dropped)

submission = pd.DataFrame({
        "PassengerId": final_test["PassengerId"],
        "Survived": log_predictions_final
    })
submission.to_csv('titanic_csv_to_submit.csv', index = False)