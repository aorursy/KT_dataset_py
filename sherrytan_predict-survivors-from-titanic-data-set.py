import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")
train.info()
train.describe()
for column in train.select_dtypes(exclude=[np.number]).columns.tolist():
    print("{}: {} unique values".format(column, train[column].nunique()))  
train.head()
for column in ['Survived', 'Pclass']:
    train[column] = train[column].astype('str')
#visualize how categorical input variables affect survival
cols = ['Pclass', 'Sex', 'SibSp','Parch', 'Embarked']
for col in cols:
    plt.figure()
    sns.countplot(x=col, data = train, hue="Survived")
    plt.title(col)
# visualize how numerical input variables affect survival
for status in train["Survived"].unique():
    plt.hist(x='Age',data=train[(train['Survived']==status) & (~train['Age'].isna())], alpha=0.5, label=status, bins=30)
    plt.title("Age distribution")
    plt.legend(title="Survived")
def transform_family_info(df):
    df["HasSibSp"] = 1
    df["HasParch"] = 1
    df["HasFamily"] = 1
    df.loc[df["SibSp"]==0,"HasSibSp"]=0
    df.loc[df["Parch"]==0,"HasParch"]=0
    df.loc[(df["HasSibSp"]==0)&(df["HasParch"]==0), "HasFamily"]=0
    
    return df


train = transform_family_info(train)

for col in ["HasSibSp","HasParch","HasFamily"]:
    plt.figure()
    sns.countplot(x=col, data = train, hue="Survived")
    plt.title(col)
train['title'] = train['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip()) #extract titles from names
sns.boxplot(x="Age", y="title", data=train) #visualize age distributions by title
age_medians  = train.pivot_table(columns="title", aggfunc='median', values="Age") #get median ages
age_medians['overall'] = train['Age'].median() #get overall median age for the entire data set. this acts as a default value in case new titles are present in test data

def fill_age(df): #fill NA values for 'Age' column based on title 
    title = df['title'].unique().item()
    try:
        df['Age'].fillna(age_medians.loc['Age',title], inplace=True)
    except:
        df['Age'].fillna(age_medians.loc['Age','overall'], inplace=True)
    return df

train = train.groupby('title').apply(lambda x: fill_age(x))
def get_age_group(df):
    
    df['Age5_orLess']=0
    df.loc[df['Age']<=5,'Age5_orLess']=1
    
    return df

train = get_age_group(train)
def get_cabin_info(df):
    
    #extract Cabin Deck and Number
    df['Cabin_Deck']= np.NaN
    df['Cabin_Number']= np.NaN
    df['Cabin_Number_Loc']= np.NaN
    df.loc[~df['Cabin'].isna(),'Cabin_Deck']  = df.loc[~df['Cabin'].isna(),'Cabin'].apply(lambda x:x[0]) #extract alphabet
    df.loc[~df['Cabin'].isna(), 'Cabin_Number']  = df.loc[~df['Cabin'].isna(), 'Cabin'].apply(lambda x:x[1:].split(' ')[0]) #retain only first booth number if entries have multiple booths
    df['Cabin_Number'] =  pd.to_numeric(df['Cabin_Number'], errors='coerce')
    df.loc[df['Cabin_Number']%2==0, 'Cabin_Number_Loc']="even"
    df.loc[df['Cabin_Number']%2==1, 'Cabin_Number_Loc']="odd"
    
    return df

train = get_cabin_info(train)
#visualize number of survivors by cabin deck
sns.countplot(x='Cabin_Deck',data=train[~train['Cabin_Deck'].isna()],hue="Survived")
# There is no Deck T on the Titanic, so that observation will be set to 'NA'
train['Cabin_Deck'] = train['Cabin_Deck'].replace("T",np.NaN) 
#visualize number of survivors by cabin number
for status in train["Survived"].unique():
    plt.hist(x='Cabin_Number',data=train[(train['Survived']==status) & (~train['Cabin_Number'].isna())], alpha=0.5, label=status, bins=30)
    plt.title("Cabin Number distribution")
    plt.legend(title="Survived")
#visualize number of survivors by Cabin_Number_Loc
sns.countplot(x='Cabin_Number_Loc',data=train[~train['Cabin_Number_Loc'].isna()],hue="Survived")
# number of missing Cabin values
train[['Cabin','Cabin_Deck','Cabin_Number','Cabin_Number_Loc']].isna().sum()
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode())
train.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#include only variables of interest
variables = ["Pclass","Sex","Age","SibSp","Parch", "Embarked", "HasSibSp", "HasParch", "Age5_orLess"]
cat_var = ["Sex","Embarked","Pclass"]

X = train[variables]

X = pd.get_dummies(X, columns=cat_var, drop_first=True)
y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=43)

def randomforestclassifier(X,y):
    rf = RandomForestClassifier(random_state=77)

    params = {'n_estimators': np.arange(100,500,100),'min_samples_split':np.arange(2,30,4),
              'criterion':["gini","entropy"]}

    rf_model_cv = GridSearchCV(rf, params, cv=StratifiedKFold(n_splits = 5, random_state=77), scoring='accuracy')
    rf_model_cv.fit(X,y)
    
    print("Cross validation score:{:.3f}".format(rf_model_cv.best_score_))
    print("Best params:{}".format(rf_model_cv.best_params_))

    return rf_model_cv

rfc = randomforestclassifier(X_train,y_train)
def logisticregression(X,y):
    log = LogisticRegression(max_iter=500)

    params = {'C':np.linspace(0.1,1,10), 'solver':['liblinear','lbfgs', 'newton-cg']}

    log_model_cv = GridSearchCV(log, params, cv=StratifiedKFold(n_splits=5,random_state=77), scoring = "accuracy")
    log_model_cv.fit(X,y)

    print("Cross validation score:{:.3f}".format(log_model_cv.best_score_))
    print("Best params:{}".format(log_model_cv.best_params_))
    
    return log_model_cv
    
log = logisticregression(X_train,y_train)
#Visualize coefs of log model
weights_log = pd.Series(log.best_estimator_.coef_.transpose()[:,0], index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8,6))
ypos = np.arange(0,len(weights_log))[::-1]

positive_weights = weights_log[weights_log>=0]
ypos_positive = ypos[weights_log>=0]
negative_weights = weights_log[weights_log<0]
ypos_negative = ypos[weights_log<=0]
ax.barh(ypos_positive,positive_weights, color='#77B7D8')
ax.barh(ypos_negative,negative_weights, color='#C16C82')

ax.set_yticks(ypos)
ax.set_yticklabels(weights_log.index)

for i,value in zip(ypos, weights_log.values):
    ax.annotate("{:.2f}".format(value), xy=(value, i))
    
plt.title("Coefficients for Log Regression Model")
## evaluate prediction accuracy on held out labelled data set using log model
y_pred = log.predict(X_test)

print("MODEL PERFORMANCE")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test,y_pred)*100))
print("Confusion matrix:\n{}".format(confusion_matrix(y_test,y_pred)))
print("Recall:\n{:.2f}%".format(100*recall_score(y_test.astype('int'), y_pred.astype('int'))))
print("Precision:\n{:.2f}%".format(100*precision_score(y_test.astype('int'), y_pred.astype('int'))))
#evaluate prediction accuracy using a baseline guess of y_train.mode() for all predictions
y_pred2 = np.repeat(y_train.mode(), len(y_test))

print("BASELINE PERFORMANCE (all outcomes assumed to take the mode of y_train)")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test,y_pred2)*100))
print("Confusion matrix:\n{}".format(confusion_matrix(y_test,y_pred2)))
test = pd.read_csv("../input/test.csv")
test.isnull().sum()/len(test)
# define a function that consolidates the data cleaning and preprocessing steps to facilitate treatment of unlabelled test set 
def preproc(df):
    
    #fill missing age values
    df['title'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip()) #extract titles from names
    df = df.groupby('title').apply(lambda x: fill_age(x))
    
    #change Pclass to categorical var
    df['Pclass']=df['Pclass'].astype('str')
    
    #fill missing Embarked data
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
    
    #create categorical SibSp/Parch/Family/Age columns
    df = transform_family_info(df)
    df = get_age_group(df)
    
    return df


test_cleaned = preproc(test) #preprocess test data in the same way as train
X_test_unlabelled = test_cleaned[variables]
X_test_unlabelled = pd.get_dummies(X_test_unlabelled , columns=cat_var)


def add_missing_dummy_columns(train, test): #ensure test set is not missing columns needed for prediction
    missing_cols = set(train.columns) - set(test.columns) 
    for c in missing_cols:
        test[c] = 0
    test = test[train.columns]
    return test

X_test_unlabelled = add_missing_dummy_columns(X,X_test_unlabelled)
filename = "submission.csv"

predictions = log.predict(X_test_unlabelled) 
predictions_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
predictions_df.to_csv(filename, index=False)
