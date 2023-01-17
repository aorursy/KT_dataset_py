#loading packages for analyze
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
#since the original code is in colab notebook, this is used for reading local csv file
#from google.colab import files
#uploaded = files.upload()
#read data into dataframe
import pandas as pd

df = pd.read_csv("../input/train.csv",na_values='NULL')
df_test = pd.read_csv("../input/test.csv",na_values='NULL')
df.head()
df.info()
#Look at the summary of data set
df.describe()
#See relationship by change hue in the function
sns.pairplot(df, hue = 'Sex')
#look into each variable distribution
sns.barplot(data = df, x = 'Sex',y='Survived')
sns.lmplot(data = df, x = 'Age',y='Survived')
sns.barplot(data = df, x = 'Pclass',y='Survived')
sns.lmplot(data = df, x = 'Fare',y='Survived')
sns.barplot(data = df, x = 'Embarked',y='Survived')
#deduplication
df.drop_duplicates(inplace = True)
df_test.drop_duplicates(inplace = True)
#Drop unnecessary columns(too many missing value and massive format)
df = df.drop(columns=['Cabin','Ticket','Name'])
df_test = df_test.drop(columns=['Cabin','Ticket','Name'])
#use median to replace missing value in Age and Fare columns
median_age = df['Age'].median()
median_age_t = df_test['Age'].median()
median_fare_t = df_test['Fare'].median()
df['Age'].fillna(median_age, inplace = True)
df_test['Age'].fillna(median_age_t, inplace = True)
df_test['Fare'].fillna(median_fare_t, inplace = True)
#get dummy variable for Embarked column
df["Embarked"] = df["Embarked"].fillna("S")
df_test["Embarked"] = df_test["Embarked"].fillna("S")

df_Emb = pd.get_dummies(df['Embarked'])
df_Emb_t = pd.get_dummies(df_test['Embarked'])

df = df.join(df_Emb)
df_test = df_test.join(df_Emb_t)
#get dummy variable for gender column
df_sex = pd.get_dummies(df['Sex'])
df = df.join(df_sex)
df_sex_t = pd.get_dummies(df_test['Sex'])
df_test = df_test.join(df_sex_t)
#drop original columns
df = df.drop(columns=['Sex','Embarked'])
df_test = df_test.drop(columns=['Sex','Embarked'])
#identify the expected output column and unique column for identify each passenger
target='Survived' #EXPECTED OUTPUT
IDcol= 'PassengerId'
x_columns = [x for x in df.columns if x not in [target,IDcol]]

X = df[x_columns]
y = df['Survived']

#initial model
rf0 = RandomForestClassifier(n_estimators= 100, oob_score=True, random_state=10)
model0 = rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
#using grid sesarch to find out best parameter in the model
#1.n_estimators  (118)
param_test1 = {'n_estimators':list(range(100,130,2))}  
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,oob_score = True ,n_jobs = -1,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
#2.max_depth & min_samples_split      (7,2)     
param_test2 = {'max_depth':list(range(2,10,1)), 'min_samples_split':list(range(2,10,2))}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 118, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, n_jobs = -1, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_

#fit the best conbination from above into new model rf1
rf1 = RandomForestClassifier(n_estimators= 118, max_depth=7, min_samples_split=2,
                                  oob_score=True, random_state=10)
model1 = rf1.fit(X,y)
print(rf1.oob_score_)
y_predprob = rf1.predict_proba(X)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

#get the importance of each variable
list(zip(df[x_columns],rf1.feature_importances_))
#apply model to test data
PRED = rf0.predict(df_test[x_columns])

#get the final output
df_test['Survived'] = PRED
output = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':df_test['Survived']})
output.to_csv('Titanic.csv', index=False, sep=',')

#files.download('Titanic.csv')
