# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
print('pandas version: ',pd.__version__)
print('numpy version: ',np.__version__)
print('matplotlib version: ',np.__version__)
print('seaborn version: ',sns.__version__)
#set style of sns and display in workbook.
sns.set_style(style='whitegrid')
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#combine used to apply updates to both training and test dataset simultaneously
combine = [train, test]
#define function to determine the percentage of survivors by certain feature
def per_survived(DataFrame,feature,feature_value):
    """
    Take in the feature to analyse and the value of that feature and returns the percentage 
    of no-shows. Intended to use as part of lambda function to update DataFrame.
    
    Parameters
    ----------
    df : the DataFrame to analyse
    feature : str
        feature of the DataFrame to analyse
    feature_value : scalar
        the feature_value to analyse
        
    """
    measure = DataFrame[(DataFrame['Survived']==1) & (DataFrame[feature]==feature_value)]['Survived'].count()
    total = DataFrame[DataFrame[feature]==feature_value][feature].count()
    return round((measure / total) * 100,2)
def set_age_group(x):
    """Set an age range
    """
    if x <=3:
        return 'infant',0
    elif x <=10:
        return 'child',1
    elif x <=18:
        return 'adolescent',2
    elif x <= 25:
        return 'Young Adult',3
    elif x <= 45:
        return 'Adult',4
    elif x <= 75:
        return 'Middle Aged',5
    else:
        return 'Elderly',6
def set_fare_bands(x,df,column_in,column_return):
    """Takes in the fare,the DataFrame of FareBands,the column name of the FareBands as an interval
    and returns the FareBand as a scalar value.
    """ 
    for i in range (len(df)):
        if x in df[column_in].loc[i]:
            return df[column_return].loc[i]
train.head()
train.info()
# Grab info about DataFrame
train.describe()
# Check for null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print(train.isnull().sum())
# Check for null values
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print(test.isnull().sum())
sns.countplot(x='Survived',data=train)
print('Percentage survived = {}%'.format(round((train[train['Survived']==1]['Survived'].sum()/len(train))*100,1)))
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='viridis')
combine = [train,test]
for dataset in combine:
    dataset['Age_Group'] = dataset['Age'].apply(lambda x: set_age_group(x))
# Create a DataFrame with value counts of patients ages
train_ages = pd.DataFrame(train['Age_Group'].value_counts())
# Clean up DataFrame
train_ages.reset_index(inplace=True)
train_ages['Description'],train_ages['Rank'] = zip(*train_ages['index'])
train_ages.set_index('Rank',inplace=True)
train_ages.sort_index(inplace=True)
train_ages.rename(columns={'Age_Range':'Count'},inplace=True)
# Add a column of the percentage survived
train_ages['Survival Percent'] = train_ages['index'].apply(lambda x: per_survived(train,'Age_Group',x))
#Drop extra information from train and test dataframe to leave just age group
combine = [train,test]
for dataset in combine:
    dataset['Age_Group'] = dataset['Age_Group'].apply(lambda x : x[1])
train_ages
plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
sns.barplot(x='Description',y='Survival Percent',data=train_ages,color='red')
sns.violinplot(x='Sex',y='Age',data=train,palette='RdBu_r')
sns.countplot(x='Survived',hue='Sex',data=train,palette='viridis')
print('female survived: {}%' .format(per_survived(train,'Sex','female')))
print('male survived: {}%' .format(per_survived(train,'Sex','male')))
male_survived = train[(train['Survived']==1) & (train['Sex']=='male')]['Survived'].sum()
male_perished = len(train) - male_survived
female_survived = train[(train['Survived']==1) & (train['Sex']=='female')]['Survived'].sum()
female_perished = len(train) - female_survived
obs = np.array([[male_survived,male_perished], [female_survived,female_perished]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print ('p value: ',p)
sns.violinplot(x='Survived',y='Age',hue='Sex',split=True,data=train,palette='RdBu_r')
combine = [train, test]
for dataset in combine:
    dataset['Alone']= dataset.apply(lambda x: 1 if x['SibSp']==0 & x['Parch']==0 else 0,axis=1)
combine = [train, test]
for dataset in combine:
    dataset['Relatives'] = dataset.apply(lambda x: x['SibSp']+x['Parch'],axis=1)
train.head(20)
sns.countplot(x='Alone',hue='Survived',data=train)
sns.countplot(x='Relatives',hue='Survived',data=train)
sns.violinplot(x='Alone',y='Age',data=train)
sns.violinplot(x='Alone',y='Age',hue='Survived',split=True,data=train)
train['Embarked'].value_counts()
sns.countplot(x='Embarked',hue='Survived',data=train)
print('Southampton survived: {}%' .format(per_survived(train,'Embarked','S')))
print('Cherbourg survived: {}%' .format(per_survived(train,'Embarked','C')))
print('Queenstown survived: {}%' .format(per_survived(train,'Embarked','Q')))
print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)

train = train.drop(['PassengerId','Ticket', 'Cabin'], axis=1)
#keep PassengerId for submission
test = test.drop(['PassengerId','Ticket', 'Cabin'], axis=1)
combine = [train, test]

"After", train.shape, test.shape, combine[0].shape, combine[1].shape
train.head()
test.head()
plt.figure(figsize=(8,6))
sns.distplot(train['Fare'],bins=50,kde=False)
imput = test.sort_values('Fare')['Fare'].median()

test['Fare'].fillna(value=imput,inplace=True)
test['Fare'].isnull().sum()
combine = [train,test]
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].astype(int)
train['FareBand'] = pd.qcut(train['Fare'], 6)
df_Fare_bands = train[['FareBand', 'Survived']].groupby(['FareBand'], 
                as_index=False).mean().sort_values(by='FareBand', ascending=True)
df_Fare_bands.reset_index(inplace=True)
df_Fare_bands.rename(columns={'index':'Band','FareBand':'Interval'},inplace=True)
df_Fare_bands
combine = [train,test]
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].apply(lambda x: set_fare_bands(x,df_Fare_bands,'Interval','Band'))
train.drop('FareBand',axis=1,inplace=True)
train.head()
combine = [train, test]
for dataset in combine:
    dataset['Sex'].replace(['female','male'],[0,1],inplace=True)
train.head()
#impute missing values with most common.
combine = [train, test]
for dataset in combine:
    dataset['Embarked'].fillna(value='S',inplace=True)
train['Embarked'].isnull().sum()
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test,columns=['Embarked'])
train.head()
import re

combine = [train,test]

for dataset in combine:
    dataset['Title'] = dataset['Name'].apply(lambda x:list(filter(None,re.split('[,. ]',x)))[1])


print(train['Title'].value_counts().head(10))
print('number of unique titles: {}'.format(len(train['Title'].value_counts())))
list_of_Names = train['Title'].value_counts().head(4).index.tolist()
titles = {list_of_Names[i]:i for i in range(len(list_of_Names))}
titles
combine = [train,test]
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(titles)
train['Title'].isnull().sum()
combine = [train,test]
for dataset in combine:
    dataset['Title'].fillna(value=4,inplace=True)
    dataset['Title'] = dataset['Title'].astype(int)
train.head()
train['Title'].isnull().sum()
ave_age = {i:np.ceil(train[train['Title']==i]['Age'].mean()) for i in range(len(train['Title'].unique()))}
ave_age
combine = [train,test]
for dataset in combine:
    dataset['Age'] = dataset.apply(lambda x: ave_age[x['Title']] if pd.isnull(x['Age']) else x['Age'],axis=1)
train.head(10)
#final tidy
combine = [train,test]
for dataset in combine:
    dataset.drop(['Name','Age'],axis=1,inplace=True)
train.head()
plt.figure(figsize=(12,6))
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test  = test
train.shape, test.shape, X_train.shape,y_train.shape, X_test.shape
def train_model(model,name,X,y):
    """
    Takes a machine learning model from scikit-learn alongside two Data Frames containing features and labels,
    fits this training data and returns the mean accuracy of the trained model
    
    model: scikit learn model
    X_train: DataFrame of features
    y_train: DataFrame of labels
    """
    model.fit(X_train,y_train)    
    acc_model = model.score(X_train, y_train)
    print('Accuracy of {name}: {acc_model:.{digits}f}%'.format(name=name,acc_model=acc_model*100,digits=2))
def validation_scores(model,X,y,scoring_metrics):
    """
    Takes a machine learning model from scikit-learn alongside two DataFrames containing features and labels,
    performs cross validation and returns the mean of the scoring metric and the std for each metric within
    the scoring_metrics list.
    
    model: scikit learn model
    X_train: DataFrame of features
    y_train: DataFrame of labels
    scoring_metrics: list of scoring metrics
    """
    
    for i in range(len(scoring_metrics)):
        metric = scoring_metrics[i]
                
        score = cross_val_score(model,X=X_train,y=y_train,cv=10,scoring=metric)
        
        metric_cap = ' '.join(word[0].upper() + word[1:] for word in metric.split())
        
        print('{metric}: {mean_score:.{digits}f}% (+/- {std_score:.{digits}f}'.format(metric=metric_cap,digits=2,
                                                    mean_score= score.mean()*100,std_score = score.std()*100))
     
names = ['Logistic Regression','SVM','KNN','Random Forest']
classifiers = [LogisticRegression(),SVC(),KNeighborsClassifier(),RandomForestClassifier()]
df_classifiers = pd.DataFrame
for name, clf in zip(names,classifiers):
    train_model(clf,name,X_train,y_train)
validation_scores(SVC(),X_train,y_train,['accuracy','precision','recall','f1'])
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.001], 'kernel': ['linear','rbf'],'class_weight':[None,'balanced']}
model = SVC()
grid = GridSearchCV(model,param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
optimised_model = grid.best_estimator_
train_model(optimised_model,'optimised SVM',X_train,y_train)
validation_scores(optimised_model,X_train,y_train,['accuracy','precision','recall','f1'])
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
              "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
grid = GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
optimised_model  = grid.best_estimator_
train_model(optimised_model,'Random Forest',X_train,y_train)
validation_scores(RandomForestClassifier(),X_train,y_train,['accuracy','precision','recall','f1'])
validation_scores(optimised_model,X_train,y_train,['accuracy','precision','recall','f1'])
y_pred = optimised_model.predict(X_test)