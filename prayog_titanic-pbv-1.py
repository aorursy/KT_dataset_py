# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt



# Algorithms
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under 
#the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data.describe()
#Drp these column, which are not important
train_data = train_data.drop(['PassengerId','Cabin','Ticket'], axis=1)
test_data = test_data.drop(['PassengerId','Cabin','Ticket'], axis=1)
train_data.head()
print(train_data.shape, test_data.shape)

# Train having 891 row with 9 column in that survival is output, same way 418 rows in test data and no label column
df_train_corr = train_data.corr().abs()
print(df_train_corr)
## By this Pclass and survived rate are correlated,, pclass - age , pcalss - fare, age-sibsip, sibsip-parc are little high correlated
sns.heatmap(df_train_corr ,annot=True)
df_test_corr = test_data.corr().abs()
print(df_test_corr)
## By this Pclass and survived rate are correlated,, pclass - age , pcalss - fare, age-sibsip, sibsip-parc are little high correlated
sns.heatmap(df_test_corr ,annot=True)
## Graphical analysis
def bar_chart(feature):
    survived=train_data[train_data['Survived']==1][feature].value_counts()
    dead=train_data[train_data['Survived']==0][feature].value_counts()
    
    df=pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    print(df)
    # Here index becomes 2 different bars,  coumns values becomes stacked bar(No of columns= no of stacks)
    df.plot(kind='bar',stacked=True, figsize=(10,5)) # Stacked = True gives stack one above the other
bar_chart('Sex')
#More men Died compared to Female, more female survived

train_data[['Sex', 'Survived']].groupby('Sex', as_index=False).mean()

female = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_female = sum(female)/len(female)

print("% of female who survived:", rate_female)
male = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_male = sum(male)/len(male)

print("% of male who survived:", rate_male)
bar_chart('Pclass')
#More people died in Class 3 compared cass 1, class 1 more survived

train_data[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()
bar_chart('Embarked')
#Embarkation= S died more, C survived more
train_data[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values('Survived', ascending=False)
100.0*train_data['Embarked'].value_counts() / len(train_data)
#  72% of the passengers on board embarked from port S  (72 perc people of titanic are embarked from 'S')
#  Port S also has the highest number of survivors, 55%
bar_chart('SibSp')
train_data[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values('Survived', ascending=False)

# Here we can see that you had a high probabilty of survival with 1 to 2 realitves, 
#but a lower one if you had less than 1 or more than 3).
bar_chart('Parch')
train_data[['Parch','Survived']].groupby('Parch', as_index=False).mean().sort_values('Survived', ascending=False)
# Here we can see that you had a high probabilty of survival with 1 to 3 total members(with parents), 
#but a lower one if you had less than 1 or more than 3).

g = sns.FacetGrid(train_data, col='Survived')  
# Here FacetGrid provides so many plots based on col value, here col=Survived which has 2 value so 2 plots

g.map(plt.hist, 'Age', bins=15)
#Here Histogram, age wise count ploted x axis age range and y axis count with bin= 15
g = sns.FacetGrid(train_data, col='Survived', row='Pclass')
# Here FacetGrid provides so many plots based on col value, here col=Survived which has 2 value so 2 plots
# Here row wil define other extra graphs based  on ro=Pclass value in each row
g.map(plt.hist, 'Age')
#Check any null values in columns and sumns only true value/isnull true value
print(train_data.isnull().sum())
print(train_data['Age'].isnull().sum())
print(train_data['Embarked'].isnull().sum())
print(train_data['Fare'].isnull().sum())
print(test_data['Fare'].isnull().sum())
## add Title instead of Name,SibSp, and Parch.
# Rename Title's value to 'Misc', if title are rare value (occured less than `10 times)

#Pasing Train set as one set in for loop 1st iteration and 2nd set test as 2nd iteration

train_data.head()

for dataset in [train_data,test_data]:
   
    #Ex: Braund, Mr. Owen Harris
    #In this we need to take only Title = Mr , so split whole name based , and take 2nd location[1], 
    # we gets Mr., then split based on . and take 1st location value[0]. thats  Mr
    
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    print(dataset.head())
        
    # take only top 10 titles

    ## Value_counts() will provide each values of column 'titles' total count respect to each value
    # Ex: Mr              517
    #     Miss            182
    #     Mrs             125
    #     Master           40
    #     Dr                7
    
    #this will create distinct  true false series with title name as index if value_counts() < 10 as True
    # Ex: Mr              False
    #     Miss            False
    #     Mrs             False
    #     Master          False
    #     Dr              True

    title_names = (dataset['Title'].value_counts() < 10) 

    # Here title count<10 are replaced my Misc
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code:
    #https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    
    # Here apply will pass title value(Mr etc) as in row vise to lambda function, 
    #if title_names.loc['Mr'] == True, then it assigns 'Misc' as Title else it will keep its existing value
    
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(dataset['Title'].value_counts())
    
    
train_data.head()



survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_data[train_data['Sex']=='female']  # Whole train data with only female
men = train_data[train_data['Sex']=='male']  # Whole train data with only male

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

#complete missing age with overall median, but it wont give accurate age, 
#train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
#test_data['Age'].fillna(test_data['Age'].median(), inplace = True)
## Filling missing values for age based on mean age per Title
print('Number of age entries missing for title Miss:', train_data[train_data['Title'] == 'Miss']['Age'].isnull().sum())
print('Number of age entries missing for title Mr:', train_data[train_data['Title'] == 'Mr']['Age'].isnull().sum())
print('Number of age entries missing for title Mrs:', train_data[train_data['Title'] == 'Mrs']['Age'].isnull().sum())
print('Number of age entries missing for title Misc:', train_data[train_data['Title'] == 'Misc']['Age'].isnull().sum())
print('Number of age entries missing for title Master:', train_data[train_data['Title'] == 'Master']['Age'].isnull().sum())
print('Mean age for title Miss:', train_data[train_data['Title'] == 'Miss']['Age'].mean())
print('Mean age for title Mr:', train_data[train_data['Title'] == 'Mr']['Age'].mean())
print('Mean age for title Mrs:', train_data[train_data['Title'] == 'Mrs']['Age'].mean())
print('Mean age for title Misc:', train_data[train_data['Title'] == 'Misc']['Age'].mean())
print('Mean age for title Master:', train_data[train_data['Title'] == 'Master']['Age'].mean())
# Here loc used in left side only filter are used to assign values only for filtered rows column
train_data.loc[(train_data['Title']== 'Miss') & (train_data['Age'].isnull()), 'Age'] = 22
train_data.loc[(train_data['Title']== 'Mr') & (train_data['Age'].isnull()), 'Age'] = 32
train_data.loc[(train_data['Title']== 'Mrs') & (train_data['Age'].isnull()), 'Age'] = 36
train_data.loc[(train_data['Title']== 'Misc') & (train_data['Age'].isnull()), 'Age'] = 46
train_data.loc[(train_data['Title']== 'Master') & (train_data['Age'].isnull()), 'Age'] = 5
train_data.isnull().sum().sort_values(ascending=False)

# Repeating the steps for test set

print('Number of age entries missing for title Miss:', test_data[test_data['Title'] == 'Miss']['Age'].isnull().sum())
print('Number of age entries missing for title Mr:', test_data[test_data['Title'] == 'Mr']['Age'].isnull().sum())
print('Number of age entries missing for title Mrs:', test_data[test_data['Title'] == 'Mrs']['Age'].isnull().sum())
print('Number of age entries missing for title Misc:', test_data[test_data['Title'] == 'Misc']['Age'].isnull().sum())
print('Number of age entries missing for title Master:', test_data[test_data['Title'] == 'Master']['Age'].isnull().sum())
print('Mean age for title Miss:', test_data[test_data['Title'] == 'Miss']['Age'].mean())
print('Mean age for title Mr:', test_data[test_data['Title'] == 'Mr']['Age'].mean())
print('Mean age for title Mrs:', test_data[test_data['Title'] == 'Mrs']['Age'].mean())
print('Mean age for title Misc:', test_data[test_data['Title'] == 'Misc']['Age'].mean())
print('Mean age for title Master:', test_data[test_data['Title'] == 'Master']['Age'].mean())
test_data.loc[(test_data['Title']== 'Miss') & (test_data['Age'].isnull()), 'Age'] = 22
test_data.loc[(test_data['Title']== 'Mr') & (test_data['Age'].isnull()), 'Age'] = 32
test_data.loc[(test_data['Title']== 'Mrs') & (test_data['Age'].isnull()), 'Age'] = 39
test_data.loc[(test_data['Title']== 'Misc') & (test_data['Age'].isnull()), 'Age'] = 44
test_data.loc[(test_data['Title']== 'Master') & (test_data['Age'].isnull()), 'Age'] = 7
test_data.isnull().sum().sort_values(ascending=False)
#complete embarked with mode, if different value are having same frequency 
#then we may have multiple mode so take 1st one as [0]
# Filling in missing values with most_frequent or mode


train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
#test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)


### Only test set has 0 fare, that to one record, so we can follow this or 2nd method
#test_data['Fare'].fillna(test_data['Fare'].mode()[0], inplace = True)
test_data.loc[test_data['Fare'].isnull()]
# Here only one row with 0 fare, see the record. here pclass =3, so check pclass price and assign
# Finding out the mean Fare for Pclass=3

test_data[test_data['Pclass']==3]['Fare'].mean()
# Assign above value
test_data.loc[test_data['Fare'].isnull() , 'Fare'] = 12.46
# Or
# test_df['Fare'] = test_df['Fare'].fillna(12.46)

test_data.head(153).tail(1)
# Age:
# Now we need to convert the ‘age’ feature. First we will convert it from float into integer. 
# we will create the new ‘AgeGroup” variable, by categorizing every age into a group.
# Note that it is important to place attention on how you form these groups, since you don’t want for example that 80% of your data falls into group 1.

data = [train_data, test_data]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int) # Convert it to integer
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed train_df['Age'].value_counts()
# let's see how it's distributed 
train_data['Age'].value_counts()
data = [train_data, test_data]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
print(train_data['Age'].isnull().sum())
print(train_data['Embarked'].isnull().sum())
#Fare=0 its some dummy value
print(test_data['Fare'].isnull().sum())
train_data.head()
#Conver categorial to integer
train_data['Sex'] = train_data['Sex'].map({'male':0,'female':1})
test_data['Sex'] = test_data['Sex'].map({'male':0,'female':1})

train_data.head()

#Pasing Train set as one set in for loop 1st iteration and 2nd set test as 2nd iteration


for dataset in [train_data,test_data]:
    
    # Creating a categorical variable to tell if the passenger is alone
    # Add new colum IsAlone
    dataset['IsAlone'] = ''
    # Assign value to new column IsAlone with condition during assign
    # If condition True inside loc then it assign value to IsAlone then 
    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) > 0)] = 1
    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) == 0)] = 0
    


print(train_data.head())

# Drop unnecessary fields from train and test
train_data.drop(['Name','SibSp','Parch'],axis=1,inplace=True)
test_data.drop(['Name','SibSp','Parch'],axis=1,inplace=True)
print(train_data.head())
print(test_data.head())
#Apply one hot dog for categorial value to make it Numeric,separate column
#Here Embarked, IsAlone,Title will be changed

print(train_data.select_dtypes(include = ['category', 'object']).columns)
train_data.dtypes
    #define x and y variables for dummy features original
train_dummy = pd.get_dummies(train_data,drop_first=True)
test_dummy = pd.get_dummies(test_data,drop_first=True)

train_dummy

#features = ["Pclass", "Sex", "SibSp", "Parch"]
#X = pd.get_dummies(train_data[features])
#X_test = pd.get_dummies(test_data[features])
#print(X_test.head())
# Creating an empty dataframe to add model predictions for comparison

pred_df = pd.DataFrame()
from sklearn.ensemble import RandomForestClassifier

y = train_dummy["Survived"]
X = train_dummy.iloc[:,1:]
print(X.head())
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X, y)
Y_pred = sgd.predict(test_dummy)

sgd.score(X, y)

acc_sgd = round(sgd.score(X, y) * 100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

Y_prediction = random_forest.predict(test_dummy)

random_forest.score(X, y)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
logreg = LogisticRegression()
logreg.fit(X, y)

Y_pred = logreg.predict(test_dummy)

acc_log = round(logreg.score(X, y) * 100, 2)
# KNN 
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X, y)
Y_pred = knn.predict(test_dummy)  

acc_knn = round(knn.score(X, y) * 100, 2)
gaussian = GaussianNB() 
gaussian.fit(X, y)
Y_pred = gaussian.predict(test_dummy)  

acc_gaussian = round(gaussian.score(X, y) * 100, 2)
perceptron = Perceptron(max_iter=5)
perceptron.fit(X, y)

Y_pred = perceptron.predict(test_dummy)

acc_perceptron = round(perceptron.score(X, y) * 100, 2)
linear_svc = LinearSVC()
linear_svc.fit(X, y)

Y_pred = linear_svc.predict(test_dummy)

acc_linear_svc = round(linear_svc.score(X, y) * 100, 2)
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X, y)
Y_pred = decision_tree.predict(test_dummy)  

acc_decision_tree = round(decision_tree.score(X, y) * 100, 2)
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
##print(results)
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X, y, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X.columns,
                            'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()
X  = X.drop("Title_Misc", axis=1)
test_dummy  = test_dummy.drop("Title_Misc", axis=1)

X.head()
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X, y)
Y_prediction = random_forest.predict(test_dummy)

random_forest.score(X, y)

acc_random_forest = round(random_forest.score(X, y) * 100, 2)
print(round(acc_random_forest,2,), "%")
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import GridSearchCV, cross_val_score

param_grid = [{ "criterion" : ["gini", "entropy"], 
              "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
              "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
              "n_estimators": [50, 100,  200, 300 ,400], 'bootstrap': [True, False]}]


clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, n_jobs=-1,  
                   cv=3,scoring='accuracy',  verbose=True)

#rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)


clf.fit(X, y)


#param_grid = [{'n_estimators': [50, 100, 200, 300], 'max_depth': [10,20,30], 'bootstrap': [True, False]}]
#grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=3, verbose=True)
#grid_search.fit(X, y)


clf.best_params_
#Assign best parameter to model
model = clf.best_estimator_
## Final Output which needs to submit
Y_prediction = model.predict(test_dummy)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


predictions = cross_val_predict(model, X, y, cv=3)
print(confusion_matrix(y, predictions))



from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y, predictions))
print("Recall:",recall_score(y, predictions))
# Another way is to plot the precision and recall against each other:
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = model.predict_proba(X)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y, y_scores)

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()



from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score


y_train_pred = model.predict(X)


accuracy = accuracy_score(y, y_train_pred,)
r_a_score = roc_auc_score(y, y_scores)
print("ACCURACY:", accuracy)
print("ROC-AUC-Score:", r_a_score)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X)
y_scores = y_scores[:,1]
aucs = []

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate):
    roc_auc = auc(false_positive_rate, true_positive_rate)   #AUC function to find auc
    #aucs.append(roc_auc)
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)  #To show label

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

#To append passenger id read again
test_data1 = pd.read_csv("/kaggle/input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test_data1.PassengerId, 'Survived': Y_prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")