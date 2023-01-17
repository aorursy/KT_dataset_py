import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
# Print current directory files
print("Files in current directory : {}\n".format(os.listdir("../input")))

# loading train data and veiwing shape 
train = pd.read_csv('../input/train.csv')
print("Train shape : {}\n".format(train.shape))

# loading test data and veiwing shape 
test = pd.read_csv('../input/test.csv')
print("Test shape : {}\n".format(test.shape))

# print all columns
print("Columns : {} ".format(train.columns))

# View of train data 
train.head(5)
# Datatypes of columns of titanic data 
print(train.dtypes)

# Description of train Data
train.describe()
# Visualizing Total count of survivors based on their sex group ( male / female)
## Female survivors are more than male survivors 
sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()
print(train['Sex'].value_counts())
## Visualizing Total count of survivors based on their pclass ( 1st , 2nd , 3rd)
pclass_pivot = train.pivot_table(index="Pclass",values="Survived")
pclass_pivot.plot.bar()
plt.show()
print(train['Pclass'].value_counts())
# 1st class are have more survival rate than 2nd and 2nd pclass has more survival rate then 3rd
# Visulazing
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass');
# Check the correlation for the current numeric feature set.
print(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())

sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f")
sns.pairplot(train,hue="Survived",markers="+",diag_kind="kde");
## Missing values  ( Age and Cabin has high missing values )

miss_train = pd.DataFrame(train.isnull().sum(), columns=['Miss_train_data'])
miss_test = pd.DataFrame(test.isnull().sum(), columns=['Miss_test_data'])
print(pd.concat([miss_train,miss_test], join = 'inner', axis = 1))
# Filling missing value in the Fare column which we'll fill with the mean.
sns.catplot(x='Pclass', y='Fare', data=test, kind='point')
# Since Fare and Pclass are related
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())
# Replace two missing values with most ouccuring "S"
train["Embarked"]=train['Embarked'].fillna("S")
test["Embarked"]=test["Embarked"].fillna("S")
# fuction to binning of continous age into binning
def binning_age(df):
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teen","Young-Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

train = binning_age(train)
test = binning_age(test)

# Plotting Age_categories
print("Missing values in Age column: {}".format(train['Age'].isnull().sum()))
sns.catplot(x='Age_categories', y='Survived', data=train, kind='bar')
plt.show()
## Binning of Fare Cateogry in 4 bins 
def binning_fare(df):
    cut_points = [0,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

# Apply Fare binning to both train and test 
train = binning_fare(train)
test = binning_fare(test)

sns.catplot(x='Fare_categories', y='Survived', data=train, kind='bar')
train['Fare_categories'].value_counts()
# Get the titles
for data in [train, test]:
    # Use split to get only the titles from the name
    data['Title'] = data['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(data['Title'].value_counts())
## Create bins using Name 
titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}

extracted_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
train["Title"] = extracted_titles.map(titles)

extracted_titles = test["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
test["Title"] = extracted_titles.map(titles)
## Min maxscaling of Sibsp Parch and Fare
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
#columns = ["SibSp","Parch","Fare"]
columns = ['Fare']
train[columns] = min_max_scaler.fit_transform(train[columns])
test[columns] = min_max_scaler.transform(test[columns])
train.describe()
# Function to create one-hot key coding  using get_dummies for Categorical variable
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix = column_name)
    df = pd.concat([df,dummies],axis = 1)
    return df
# Columns in list to create dummies variable 
column = ['Age_categories','Pclass', 'Sex','Embarked','Fare_categories','Title' ]
for i in column:
    train = create_dummies(train,i)
    test = create_dummies(test,i)
    
print(train.columns)
print("Train shape : {}".format(train.shape))
# Creating a variable with Cabin or not 
for df in [train,test]:
    df['Has_cabin'] = df['Cabin'].notna().astype(int)  

# Creating new variable using Sibsp and Parch
for df in [train,test]:
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Single'] = (df['family_size'] >  1).astype(int)

print("New train shape : {}\n".format(train.shape))
print(train.isna().sum())
print("\n")
print(test.isna().sum())
#[df.drop(columns=['PassengerId','Pclass','Cabin','SibSp','Parch','Age'], inplace = True) for df in [train, test]]
print("Train shape : {}".format(train.shape))
train.corr()
# Use only feature with high coefficnets
columns = ['Fare','Age_categories_Missing','Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teen','Age_categories_Adult',
           'Age_categories_Senior','Pclass_1','Pclass_3','Sex_female','Embarked_C',
           'Embarked_S','Fare_categories_0-12','Fare_categories_50-100', 
           'Fare_categories_100+', 'Title_Master','Title_Miss', 'Title_Mr',
           'Title_Mrs', 'Title_Officer','Has_cabin','Single']
holdout = test 

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2,random_state=0)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
print_score(lr, X_train, y_train, X_test, y_test, train=True)
print_score(lr, X_train, y_train, X_test, y_test, train=False)
from sklearn.svm import LinearSVC

svm_clf = LinearSVC(dual=False)
svm_clf.fit(X_train, y_train)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train.ravel())
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)
from sklearn.ensemble import RandomForestClassifier

ada_rf_clf = AdaBoostClassifier(RandomForestClassifier())
ada_rf_clf.fit(X_train, y_train)

print_score(ada_rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)

print_score(gbc, X_train, y_train, X_test, y_test, train=True)
print_score(gbc, X_train, y_train, X_test, y_test, train=False)

import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(all_X[columns], all_y)
holdout_predictions = xgb_clf.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("xg_boost.csv",index=False)

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(all_X[columns], all_y)
holdout_predictions = gbc.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("gradient_boost.csv",index=False)
