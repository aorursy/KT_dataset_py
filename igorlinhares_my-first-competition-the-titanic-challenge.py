# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
# Loading the data set
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data = pd.concat([train, test])
data.isna().sum()
# Creating features has_age and has_cabin
data['age_na'] = data['Age'].isna().replace([True, False], [1, 0])
data['cabin_na'] = data['Cabin'].isna().replace([True, False], [1, 0])
# Visualizing the features with missing values

fare = data.loc[data['Fare'].notna(), 'Fare']
age = data.loc[data['Age'].notna(), 'Age']
embarked = data.loc[data['Embarked'].notna(), 'Embarked'].value_counts()

# Plotting the informations
fig, ax = plt.subplots(1,3, figsize = (12,3))
ax[0].boxplot(age, labels = ['Age'])
ax[1].boxplot(fare, notch= True, labels = ['Fare'])
ax[2].bar(embarked.index, embarked.values, color = ['r', 'g', 'b'], alpha=0.5)

plt.show()
# Filling the missing values except the ones at the Cabin column...
data_filled = data.copy()

data_filled.loc[data['Age'].isna(), 'Age'] = round(data_filled['Age'].mean())
data_filled.loc[data['Fare'].isna(), 'Fare'] = data_filled['Fare'].median()
data_filled.loc[data['Embarked'].isna(), 'Embarked'] = 'S'
survived = data_filled['Survived'] == 1
died = data_filled['Survived'] == 0
train = data_filled['Survived'].notna()
fare_100 = data_filled['Fare']<=100
bins = list(range(0,100,7))
fig, ax = plt.subplots(1, 2, figsize=(18,4), sharey=True)

sns.distplot(data_filled[train]['Fare'], hist = False, color = 'green', ax = ax[0], label= 'Total')
sns.distplot(data_filled[survived]['Fare'], hist = False, color = 'blue', ax = ax[0], label= 'Survived')
sns.distplot(data_filled[died]['Fare'], hist = False, color = 'red', ax = ax[0], label= 'Died')
ax[0].set_title('Total fare distribution')

sns.distplot(data_filled[train & fare_100]['Fare'], hist = False, color = 'green', ax = ax[1], label= 'Total')
sns.distplot(data_filled[survived & fare_100]['Fare'], hist = False, color = 'blue', ax = ax[1], label= 'Survived')
sns.distplot(data_filled[died & fare_100]['Fare'], hist = False, color = 'red', ax = ax[1], label = 'Died')
ax[1].set_title('Fare distribution (Less than 100)')

fares = data_filled[train]['Fare'].apply(lambda x: x//10).round().value_counts()
fare_suvived = data_filled[survived]['Fare'].apply(lambda x: x//10).round().value_counts().rename("Survived")
fares = pd.concat([fares, fare_suvived], axis = 1).replace(np.nan, 0)
fares['Percentage'] = 100*fares['Survived']/fares['Fare']
fig2, ax2 = plt.subplots(1, figsize=(18, 3))

ax2.bar(fares.index, fares.Percentage)
ax2.set_title("Rate of survival for each increase of 10 in Fare")
def categorize_fare(fare):
    if fare < 10:
        return 0
    elif fare < 50:
        return 1
    elif fare < 100:
        return 2
    else:
        return 3
    
data_filled['Fare_Categorical'] = data_filled['Fare'].round().apply(categorize_fare)
survived = data_filled['Survived'] == 1
died = data_filled['Survived'] == 0
men = data_filled['Sex'] == 'male'
women = data_filled['Sex'] == 'female'

# Histogram per of ages, based on the sex and the survival

bins = list(range(0,90,9))

fig, ax = plt.subplots(1, 3, figsize = (15,4))
ax[0].hist(data_filled[survived]['Age'], histtype='step', color='blue', bins = bins, label='Survived',linewidth=3)
ax[0].hist(data_filled[died]['Age'], histtype='step', color='red', bins = bins, label='Died', linewidth=3)
ax[0].legend()
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Number of passengers')
ax[0].set_title('Histogram of ages')


ax[1].hist(data_filled[women & survived]['Age'], histtype = 'step', color='red', bins = bins, label = 'Women', linewidth=3)
ax[1].hist(data_filled[men & survived]['Age'], histtype = 'step', color='blue',  bins = bins, label = 'Men', linewidth=3)
ax[1].legend()
ax[1].set_xlabel("Age")
ax[1].set_title("Survivors per age and Sex")

ax[2].hist(data_filled[women & died]['Age'], histtype = 'step', color='red', bins = bins, label = 'Woman', linewidth=3)
ax[2].hist(data_filled[men & died]['Age'], histtype = 'step', color='blue', bins = bins, label = 'Men', linewidth=3)
ax[2].legend()
ax[2].set_xlabel("Age")
ax[2].set_title("Passengers died per age and sex")

def categorize_age(age):
    if age < 10:   # Child
        return 0
    elif age < 20: # Mid Age
        return 1
    elif age < 55: # Adult
        return 2
    else:          # Elderly
        return 3
    
data_filled['Age_Categorical'] = data_filled['Age'].round().apply(categorize_age)
first_c = data_filled[data_filled['Pclass'] == 1]
second_c = data_filled[data_filled['Pclass'] == 2]
third_c = data_filled[data_filled['Pclass'] == 3]
print(first_c.shape, second_c.shape, third_c.shape)
# Percentage of survival of each class
first_c = 100*first_c['Survived'].value_counts()/first_c['Survived'].count()
second_c = 100*second_c['Survived'].value_counts()/second_c['Survived'].count()
third_c = 100*third_c['Survived'].value_counts()/third_c['Survived'].count()

percentages = pd.concat([first_c, second_c, third_c], axis=1, keys=['First', 'Second', 'Third']).transpose()

fig, ax = plt.subplots(figsize = (6,3))
ax.bar(percentages.index, percentages[1], label = 'Survived', color = 'blue', alpha = 0.6)
ax.set_xlabel("Class")
ax.set_ylabel("Percentage")
ax.set_title("Percentage of survival in each class")
ax.legend()
embarked = data_filled[data_filled['Survived'].notna()]['Embarked'].value_counts()
emb_survived = data_filled[survived]['Embarked'].value_counts()
emb_survived_men = data_filled[survived & men]['Embarked'].value_counts()
emb_survived_women = data_filled[survived & women]['Embarked'].value_counts()
percentages = 100*emb_survived/embarked
percentages_men = 100*emb_survived_men/embarked.values
percentages_women = 100*emb_survived_women/embarked.values

fig, ax = plt.subplots(1, figsize=(6,4))
ax.bar(percentages.index, percentages.values, label='Survived')
ax.set_title('Percentage of survival per boarding place')
ax.set_ylabel('Percentage')
ax.legend()
data_filled['Embarked'] = data_filled['Embarked'].replace({'C': 0, 'Q': 1, 'S':2})
# Adding the 'Family' Feature
data_filled['Family'] = data_filled['SibSp'] + data_filled['Parch']

# Lets see famiily
family = data_filled[data_filled['Survived'].notna()]['Family'].value_counts()
family_survived = data_filled[survived]['Family'].value_counts().rename("family_survived")
family_all = pd.concat([family, family_survived], axis = 1).replace(np.nan, 0)

fig, ax = plt.subplots(1, figsize=(7,5))
ax.bar(family_all.index, 100*family_all.family_survived/family_all.Family)
ax.set_title('Percentage of survival due the number of family members')

# 3 categories to Family (No Family, 1-3, 4 or more)

def categorize_family(members):
    if members == 0:    # No Family
        return 0
    elif members < 4:   # 1 - 3 members
        return 1
    else:               # 4 or more members
        return 2
    
data_filled['Family'] = data_filled['Family'].apply(categorize_family)

# Analyzing the feature 'Name'

cv = CountVectorizer()
count_names = cv.fit_transform(data_filled.Name)
word_count = pd.DataFrame(cv.get_feature_names(), columns = ['word'])
word_count['count'] = count_names.sum(axis=0).tolist()[0]
word_count = word_count.sort_values("count", ascending = False).reset_index(drop=True)

#word_count[0:50]
word_count[0:5]
def extract_title(name):
    name = name.lower().replace(".", "")

    titles = ['mr', 'miss', 'mrs', 'master', 'dr', 'rev']
    others = ['don', 'dona', 'sir','mme', 'mlle', 'ms', 'major', 'capt', 'lady', 'col', 'countess', 'jonkheer']
    
    for word in name.split():
        if word in titles:
            return word
        elif word in others:
            return 'other'
        
data_filled['Title'] = data_filled['Name'].apply(extract_title)
data_filled['Title'].value_counts()
titles = data_filled[data_filled['Survived'].notna()]['Title'].value_counts()
titles_survived = data_filled[survived]['Title'].value_counts().rename('titles_survived')
titles_all = pd.concat([titles, titles_survived], axis = 1).replace(np.nan, 0)
titles_all['Percentage'] = 100*titles_all.titles_survived/titles_all.Title
titles_all.sort_values(by='Percentage', inplace = True)
plt.bar(titles_all.index, titles_all.Percentage)
def categorize_title(title):
    if title == 'rev':      # 0 - Reverend
        return 0
    elif title == 'mr':     # 1 - Mister
        return 1
    elif title == 'dr':     # 2 - Doctor or Master
        return 2
    elif title == 'master': 
        return 2
    elif title == 'other':  # 3 - Other
        return 3
    elif title == 'miss':   # 4 - Miss or Misses
        return 4
    else:
        return 4
    
    
data_filled['Title'] = data_filled['Title'].apply(categorize_title)
# Sex 
data_filled['Sex'] = data_filled['Sex'].replace(['male', 'female'], [0,1])
data_filled.drop(['Name', 'Parch', 'Age', 'Fare', 'SibSp', 'Ticket', 'Cabin'], axis='columns', inplace = True)
data_filled.head()
train = data_filled[data_filled['Survived'].notna()]
test = data_filled[data_filled['Survived'].isna()]
fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(train.corr(), center = 0, annot = True, square = True, cmap = 'YlGn', ax = ax, fmt='.2g', linewidths=3)
# Required libraries
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
X = train.loc[:, 'Pclass':'Family']
y = train.loc[:, 'Survived']
knn = KNeighborsClassifier()
logreg = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SVC()

classifiers = [knn, logreg, dt, rf, svm]

results = []
for classifier in classifiers:
    results.append(cross_val_score(classifier, X,y, cv=5).mean())

results
X_submission = test.loc[:, 'Pclass':'Family']
submission = test.loc[:, ['PassengerId', 'Survived']]
svm.fit(X, y)
y_submission = svm.predict(X_submission).astype('int')
submission['Survived'] = y_submission
submission.head()
submission.to_csv('submission.csv', index=False)