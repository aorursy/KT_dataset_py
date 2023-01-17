import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
#import the datasets
dataset = pd.read_csv('../input/train.csv')
dataset.head()
dataset.shape
#Split into train and test data
train_ratio = 0.7
dataset_size = 891

train_data = dataset.sample(int(dataset_size*train_ratio))
test_data = dataset.sample(int(dataset_size*(1-train_ratio)))

train_data.head()
train_data.shape
train_data.Survived.value_counts()
train_data.Pclass.value_counts()
plt.title("Survival by Ticket Class")
sns.countplot(x='Pclass',hue='Survived',data=train_data)
plt.title("Survival Count by class")
sns.countplot(x="Pclass",data=train_data[train_data.Survived==1])
train_data.Name.describe()
#Going to split Name into First_Name and Last_Name, and create column for last name
def SplitName(name):
    name = name.split(',')
    last_name = name[0]
    return last_name
train_data['Last_Name'] = train_data.Name.apply(SplitName)
train_data.Last_Name.describe()
#Counts number of same last names and return true if more than 1
def CheckForFamily(Last_Name):
    last_name_count = train_data.Last_Name[train_data.Last_Name == Last_Name].count()
    if last_name_count > 1:
        return 1
    return 0
train_data['Has_Family'] = train_data.Last_Name.apply(CheckForFamily)
train_data.Has_Family.value_counts()
plt.title("Survival with Family")
sns.countplot(x='Has_Family',hue='Survived',data=train_data)
family_survival_corr = train_data[['Survived','Has_Family']].corr()
plt.figure(figsize=(10,10))
plt.title("Correlation between having family aboard and surviving")
sns.heatmap(family_survival_corr,annot=True,square=True)
train_data.head()
plt.title("Sex Ratio")
sns.countplot(train_data.Sex)
plt.title("Sex/Survival Ratio")
sns.countplot(x='Sex',hue='Survived',data=train_data)
plt.title('Survival by Sex and Age')
sns.swarmplot(x='Sex',y='Age',hue='Survived',data=train_data)
train_data.Age.describe()
plt.title('Age Distribution Onboard')
sns.distplot(train_data.Age[train_data.Age.notnull()])
train_data.Age.isnull().sum()
train_data[train_data.Age.isnull()]
#Extract Title to separate field
def ExtractTitle(name):
    name = name.split(',')
    last_name = name[1].split('.')
    title = last_name[0]
    return title
#Clean titles into 4 categories
def CleanTitle(title):
    
    title=title.strip(' ')
    
    title_equivalents_dict = {'Miss':["Miss","Ms","Mlle"],'Mr':['Mr','Rev','Dr','Col','Major','Jonkheer','Capt','Don','Sir'],
                             'Mrs':['Mrs','Mme','the Countess','Lady','Dona'],'Master':['Master']}
    
    main_titles_lst = ['Miss','Mr','Mrs','Master']
    
    title_to_return = title
    
    for main_title in main_titles_lst:
        if title in title_equivalents_dict[main_title]:
            title_to_return = main_title
            
    return title_to_return
train_data['Title'] = train_data.Name.apply(ExtractTitle)
train_data['Title']=train_data.Title.apply(CleanTitle)
train_data.Title.value_counts()
#Create function to return dict of average ages of each age category
def CalculateAverageAge():
    avg_ages_by_title = {'Mr':0,'Mrs':0,'Miss':0,'Master':0}
    title_lst = ['Mr','Mrs','Miss','Master']
    for title in title_lst:
        title_data = train_data[train_data.Title==title]
        avg_age = title_data.Age.mean()
        avg_ages_by_title[title] = avg_age
    return avg_ages_by_title
avg_ages_by_title = CalculateAverageAge()
print(avg_ages_by_title)
def ImputeAverageAge(title,avg_ages_by_title):
    return avg_ages_by_title[title]
#Create column with average age for each person
avg_ages_by_title = CalculateAverageAge()
train_data['Avg_Ages'] = train_data.Title.apply(lambda title: ImputeAverageAge(title,avg_ages_by_title))
train_data.Avg_Ages.describe()
#Fill blank values in Age with corresponding Avg_Ages value
train_data['Age']=train_data.Age.fillna(train_data['Avg_Ages'])
train_data.drop(['Avg_Ages'],axis=1,inplace= True)
train_data.Age.describe()
plt.title('Age Distribution Onboard After Imputation')
sns.distplot(train_data.Age[train_data.Age.notnull()])
plt.title('Survival Rate and Age')
sns.swarmplot(y='Age',x='Survived',data=train_data)
def CheckIfMinor(age):
    if age < 18:
        return 1
    return 0
train_data['Is_Minor'] = train_data.Age.apply(CheckIfMinor)
plt.title('Surival Rate by Age Status')
sns.countplot(x='Is_Minor',hue='Survived',data=train_data)
minor_survival_corr = train_data[['Survived','Is_Minor']].corr()
plt.figure(figsize=(10,10))
plt.title("Correlation between being a minor and surviving")
sns.heatmap(minor_survival_corr,annot=True,square=True)
train_data.head()
#Create field Num_Family by adding Parch and SibSp fields.
train_data['Num_Family']  = train_data['SibSp'] + train_data['Parch']
#Rework Has_Family to simply check that Num_Family is non-zero
def CheckForFamily(num_family):
    if num_family > 0:
        return 1
    return 0
train_data['Has_Family'] = train_data.Num_Family.apply(CheckForFamily)
plt.title('Survival If Family is onboard')
sns.countplot(x='Has_Family',hue='Survived',data=train_data)
family_survival_corr = train_data[['Survived','Has_Family']].corr()
plt.figure(figsize=(10,10))
plt.title("Correlation between having family aboard and surviving")
sns.heatmap(family_survival_corr,annot=True,square=True)
train_data.head()
train_data.Fare.describe()
plt.title('Fare Distribution')
sns.distplot(train_data.Fare)
plt.figure(figsize=(10,10))
plt.title('Fare Variation by ticket class')
sns.boxplot(x='Pclass',y='Fare',data=train_data)
plt.figure(figsize=(10,10))
plt.title('Average Fare by Survival')
sns.boxplot(x='Survived',y='Fare',data=train_data)
fare_survival_corr = train_data[['Fare','Pclass','Survived']].corr()
plt.figure(figsize=(10,10))
plt.title('Correlation between Fare, Ticket Class, and Survival')
sns.heatmap(fare_survival_corr,annot=True,square=True)
train_data.head()
train_data.Cabin.value_counts()
train_data.Cabin.fillna('None',inplace=True)
#See if the passenger had a cabin
def CheckForCabin(cabin):
    if cabin == 'None':
        return 0
    return 1
train_data['HasCabin'] = train_data.Cabin.apply(CheckForCabin)
cabin_pclass_corr = train_data[['Pclass','HasCabin']].corr()
plt.figure(figsize=(10,10))
plt.title('Correlation between having a cabin and class')
sns.heatmap(cabin_pclass_corr,annot=True,square=True)
plt.figure(figsize=(10,10))
plt.title('Ticket Fare by HasCabin')
sns.boxplot(x='HasCabin',y='Fare',data=train_data)
plt.title('Survival by HasCabin')
sns.countplot(x='HasCabin',hue='Survived',data=train_data)
cabin_survival_corr = train_data[['HasCabin','Pclass','Survived']].corr()
plt.figure(figsize=(10,10))
plt.title('Correlation between Survival and Cabin')
sns.heatmap(cabin_survival_corr,annot=True,square=True)
train_data.head()
plt.title('Point of Embarkation vs Survival')
sns.countplot(x='Embarked',hue='Survived',data=train_data)
plt.title('Point of Embarkation vs Pclass')
sns.countplot(x='Embarked',hue='Pclass',data=train_data)
train_data.head()
plt.title('Titles vs Survival')
sns.countplot(x='Title',hue='Survived',data=train_data)
train_data.head()
test_data.head()
#Generating Title field:
test_data['Title'] = test_data.Name.apply(ExtractTitle)
test_data['Title'] = test_data.Title.apply(CleanTitle)
test_data.head()
#Imputing Age Values by Title
avg_ages_by_title = {'Mr':32.41638225255973,'Mrs':36.46969696969697,'Miss':22.65681818181818,'Master':3.5596428571428573}
test_data['Avg_Ages'] = test_data.Title.apply(lambda title: ImputeAverageAge(title,avg_ages_by_title))
test_data['Age']=test_data.Age.fillna(test_data['Avg_Ages'])
#Generating the Is_Minor and Has_Family features
test_data['Is_Minor'] = test_data.Age.apply(CheckIfMinor)

test_data['Num_Family'] = test_data['SibSp'] + test_data['Parch']
test_data['Has_Family'] = test_data.Num_Family.apply(CheckForFamily)
#Generate the Last_Name feature
test_data['Last_Name'] = test_data.Name.apply(SplitName)
test_data.head()
#Choosing initial features
features_being_used = ['Pclass','Sex','Is_Minor','Has_Family','Embarked','Title']
#Getting the testing and training data ready
target_variable = 'Survived'

y_train = train_data[target_variable]
X_train = train_data[features_being_used]

y_test = test_data[target_variable]
X_test = test_data[features_being_used]
#Dummy the Categorical Variables in X_train and X_test
for var in X_train.columns:
    if X_train[var].dtypes == object:
        X_train[var] = pd.Categorical(X_train[var])
X_train = pd.get_dummies(X_train)

for var in X_test.columns:
    if X_test[var].dtypes == object:
        X_test[var] = pd.Categorical(X_test[var])
X_test = pd.get_dummies(X_test)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
from sklearn.tree import export_graphviz
dt = DecisionTreeClassifier(random_state=2)
model = dt.fit(X_train,y_train)
pred = model.predict(X_test)
score_base = accuracy_score(pred,y_test)
print('Accuracy is: '+str(score_base))
print(classification_report(pred,y_test))
rf = RandomForestClassifier(random_state=2)
model = rf.fit(X_train,y_train)
pred = model.predict(X_test)
score_base = accuracy_score(pred,y_test)
print('Accuracy is: '+str(score_base))
print(classification_report(pred,y_test))
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
test_data.head()
#Extracting the Title from the Age variable:
class Title(TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self,X,y=None):
        X['Title'] = X.Name.apply(ExtractTitle)
        X['Title'] = X.Title.apply(CleanTitle)
        return X
    
    def fit(self,X,y):
        return self
#Imputing Age Values
class Impute_Age(TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self,X,y=None):
        avg_ages_by_title = {'Mr':32.41638225255973,'Mrs':36.46969696969697,'Miss':22.65681818181818,'Master':3.5596428571428573}
        X['Avg_Ages'] = X.Title.apply(lambda title: ImputeAverageAge(title,avg_ages_by_title))
        X['Age']=X.Age.fillna(X['Avg_Ages'])
        return X
    
    def fit(self,X,y):
        return self
#Creating the feature Is_Minor from Age:
class Is_Minor(TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self,X,y=None):
        X['Is_Minor'] = X.Age.apply(CheckIfMinor)
        return X
    
    def fit(self,X,y):
        return self
#Creating the feature Has_Family from SibSp and ParCh:
class Has_Family(TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self,X,y=None):
        X['Num_Family'] = X['SibSp'] + X['Parch']
        X['Has_Family'] = X.Num_Family.apply(CheckForFamily)
        return X
    
    def fit(self,X,y):
        return self
#Keeping only the useful features:
class Prune_Features(TransformerMixin):
    def __init__(self):
        self.features = ['Pclass','Sex','Is_Minor','Has_Family','Embarked','Title']
        pass
    
    def transform(self,X,y=None):
        X = X[self.features]
        return X
    
    def fit(self,X,y):
        return self
#Dummy the Categorical features
class Dummy(TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self,X,y=None):
        '''
        for var in X.columns:
            if X[var].dtypes == object:
                X[var] = pd.Categorical(X[var])
        '''
        X = pd.get_dummies(X)
        return X
    
    def fit(self,X,y):
        return self
#Creating the Pipeline
processor = Pipeline([('title',Title()),
                ('impute_age',Impute_Age()),
                ('is_minor',Is_Minor()),
                ('has_family',Has_Family()),
                ('prune_features',Prune_Features()),
                ('dummy',Dummy()),
                #Decision Tree
                #('dt',DecisionTreeClassifier(random_state=2))
                #Random Forest
                ('rf',RandomForestClassifier(random_state=2))
                ])
#Training data:
y_train = train_data['Survived']
X_train = train_data.drop(['Survived'],axis=1)
#Testing data:
y_test = test_data['Survived']
X_test = test_data.drop(['Survived'],axis=1)
#Prediction
pipe_model = processor.fit(X_train,y_train)
pred = pipe_model.predict(X_test)
score_base = accuracy_score(pred,y_test)
print('Accuracy is: '+str(score_base))
print(classification_report(pred,y_test))
final = pd.read_csv('../input/test.csv')
final.head()
pred_final = pipe_model.predict(final)
pred_final
final.head()
submission = final.loc[0::,['PassengerId']]
submission.head()
submission['Survived'] = pred_final
submission.head()
submission.to_csv('Submission.csv')
