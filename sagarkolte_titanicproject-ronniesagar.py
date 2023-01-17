def c_log(x):
    if x==0:
        return -200
    else:
        return np.log(x)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy import stats
import seaborn as sns
import math
import re
regex = '[A-Z]'
scaler = MinMaxScaler()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
####################################################################################################################
####################################################################################################################
####Preparing the Data#####
data = pd.read_csv('/kaggle/input/titanic/train.csv')#----->Loading Data
data['Cabin'] = data.apply(lambda row: 0 if pd.isnull(row['Cabin']) else row['Cabin'],axis=1)
data['Cabin'] = data.apply(lambda row: re.findall(regex,row['Cabin'])[0] if row['Cabin']!=0 else 0, axis=1)
data['Cabin'] = data.apply(lambda row: np.mean(data[data['Cabin']==row['Cabin']]['Age']),axis=1)
#data = data.drop(['Cabin'],axis=1)#------------------------>Too many missing values
data = data.dropna(subset =['Embarked'], how = 'any')#----->Drop 2 rows
#data = data[data['Fare']<500]
####Imputing the Age Column######
list_age = list(data['Age'])
lower = 0
upper = 100
mu = data['Age'].mean()
sigma = data['Age'].std()
for i in range(len(data)):
    if math.isnan(list_age[i]):
        list_age[i]=stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)        
data['Age'] = list_age#------------------------------------>Impute Data for Age
#data['Child'] = data.apply(lambda row:1 if row['Age']<=12 else 0, axis=1)
#data['Litter'] = data.apply(lambda row: 1 if row['SibSp']>=2 else 0, axis=1)
#data['LC'] = data.apply(lambda row:1 if (row['Child']==1)&(row['Litter']==0) else 0, axis=1)
#data = data.drop(['Child','Litter'],axis=1)

#####Drop columns, and make one hot encoding for categorical variables#######
dummy1 = pd.get_dummies(data['Sex'])#---------------------->Create one hot encoding
data = pd.concat([data,dummy1],axis=1)
data = data.drop(['Sex','male'],axis=1)
data['Litter'] = data.apply(lambda row: 1 if (row['SibSp']>=2)&(row['female']==0) else 0, axis=1)
#data['Cheap'] = data.apply(lambda row:1 if row['Fare']<40 else 0,axis=1)
#data['unfort_fem'] = data.apply(lambda row:1 if (row['Cheap']==1 and row['female']==1) else 0,axis=1)
data['log(fare/age)'] = data.apply(lambda row:row['Fare']/row['Age'],axis=1)
data['log(fare/age)'] = data.apply(lambda row:1 if row['log(fare/age)']>20 else 0, axis=1)
data['company'] = data.apply(lambda row:row['SibSp']+row['Parch'],axis=1)
#data = data.drop(['SibSp','Parch'],axis=1)
#dummy2 = pd.get_dummies(data['Pclass'])
#data = pd.concat([data,dummy2],axis=1)
#data = data.drop(['Pclass',1,2],axis=1)
data['Pclass'] = data.apply(lambda row:np.mean(np.array(data[(data['Pclass']==row['Pclass'])&(data['Embarked']==row['Embarked'])]['Fare'])),axis=1)
data = data.drop(['PassengerId','Name','Ticket','Embarked'],axis=1)
labels = data['Survived']
train = data.drop('Survived',axis=1)#---------------------->Create train data and labels
#train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
survived_males = data[(data['female']==0)&(data['Survived']==1)]
males = data[(data['female']==0)&(data['Litter']==0)]
males.sort_values(['Survived','Pclass','Fare'],ascending=False).head(50)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size = 0.2, random_state = 5)
# Create the model with 100 trees
model_RF = RandomForestClassifier(n_estimators=3000, 
                               bootstrap = True,
                               max_features = 'sqrt',max_depth=10)

from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(solver='newton-cg')

from xgboost import XGBClassifier
model_xg = XGBClassifier(n_estimators=3000)

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=10)

model_RF.fit(X_train, y_train)
model_log.fit(X_train,y_train)
model_xg.fit(X_train,y_train)
model_knn.fit(X_train,y_train)
def generate_cr(model,X_test,y_test):
    y_true = y_test
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report
    #print(classification_report(y_true,y_pred))
    cr = classification_report(y_true,y_pred)
    model_pred = pd.DataFrame({'model_pred':y_pred})
    df = pd.concat([X_test,model_pred],axis=1)
    return cr,df

def generate_cm(model,X_test,y_test):
    y_true = y_test
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix 
    #print(classification_report(y_true,y_pred))
    #cr = classification_report(y_true,y_pred)
    cm = confusion_matrix(y_true, y_pred) 
    model_pred = pd.DataFrame({'model_pred':y_pred})
    df = pd.concat([X_test,model_pred],axis=1)
    return cm,df
model_list = [model_RF,model_log,model_xg,model_knn]
g = lambda model:generate_cr(model,X_test,y_test)
cr_list = list(map(g,model_list)) 
print(cr_list[0][0])
import warnings
warnings.filterwarnings("ignore")
#test_df = pd.concat([X_test,y_test],axis=1)
y_pred = pd.DataFrame(model_RF.predict(X_test))
X_test['pred'] = X_test.apply(lambda row:model_RF.predict(np.array(row).reshape(1, -1))[0],axis=1)
test_df = pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True)],axis=1)
'''
test_df['MCtest'] = test_df.apply(lambda row: 1 if (row['female']==0)&(row['Cabin']==1)&(row[3]==0)&('SibSp'!=0) else 0,axis=1)
test_df['PCtest'] = test_df.apply(lambda row: 1 if (row['MCtest']==1)&(row['SibSp']==0)&(row['Parch']!=0) else 0, axis=1)
test_df['npred'] = test_df.apply(lambda row:1-row['pred'] if row['MCtest']==1 else row['pred'],axis=1)
test_df['npred'] = test_df.apply(lambda row:1-row['npred'] if row['PCtest']==1 else row['npred'],axis=1)
#test_df['FNCtest'] = test_df.apply(lambda row: 1 if (row['female']==1)&(row['Cabin']==0)&(row[3]==1) else 0,axis=1) 
#test_df['npred'] = test_df.apply(lambda row:1-row['pred'] if row['FNCtest']==1 else row['pred'],axis=1)


from sklearn.metrics import classification_report
cr = classification_report(test_df['Survived'],test_df['npred'])
print(cr)
'''
import warnings
warnings.filterwarnings("ignore")


def spl_predict(model,df):
    df['pred'] = model.predict(df)
    df['MCtest'] = df.apply(lambda row: 1 if (row['female']==0)&(row['Cabin']==1)&(row[3]==0)&(row['SbSp']!=0) else 0,axis=1) 
    df['PCtest'] = df.apply(lambda row: 1 if (row['MCtest']==1)&(row['SibSp']==0)&(row['Parch']!=0) else 0, axis=1)
    df['npred'] = df.apply(lambda row:int(1-row['pred']) if row['MCtest']==1 else int(row['pred']),axis=1)
    df['npred'] = df.apply(lambda row:1-row['npred'] if row['PCtest']==1 else row['npred'],axis=1)
    return list(df['npred'])

wrong_df = test_df[test_df['pred']!=test_df['Survived']]
wrong_df['Wrong'] = wrong_df.apply(lambda row:1,axis=1)
right_df = test_df[test_df['pred']==test_df['Survived']].reset_index(drop=True)
right_df['Wrong'] = right_df.apply(lambda row:0,axis=1)
right_wrong = pd.concat([pd.DataFrame(wrong_df),pd.DataFrame(right_df)],ignore_index=True)
#right_wrong['MCtest'] = right_wrong.apply(lambda row: 1 if (row['female']==0)&(row['Cabin']==1)&(row[3]==0) else 0,axis=1) 
right_wrong['age/fare'] = right_wrong.apply(lambda row:(row['Age']+0.01)/(row['Fare']+0.01),axis=1)
#right_wrong = right_wrong[right_wrong['MCtest']==1]
wrong = right_wrong['Wrong']
#right_wrong = right_wrong.drop(['pred','Wrong','age/fare','SibSp','Parch','company','log(fare/age)'],axis=1)
lut = dict(zip(wrong.unique(), "rb"))
row_colors = wrong.map(lut)
g = sns.clustermap(right_wrong, row_colors=row_colors,standard_scale=1)
wrong_df[wrong_df['female']==0].sort_values('Fare').reset_index(drop=True)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
passid = test['PassengerId']
print(list(test))
####Imputing the Age Column######
list_age = list(test['Age'])
lower = 0
upper = 100
mu = data['Age'].mean()
sigma = data['Age'].std()
for i in range(len(test)):
    if math.isnan(list_age[i]):
        list_age[i]=stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)        
test['Age'] = list_age#------------------------------------>Impute Data for Age

test['Fare'] = test.apply(lambda row:get_fare_pclass(test,row['Pclass']) if pd.isnull(row['Fare']) else row['Fare'],axis=1)
#test['Cabin'] = test.apply(lambda row: 0 if pd.isnull(row['Cabin']) else 1,axis=1)
test['Cabin'] = test.apply(lambda row: 0 if pd.isnull(row['Cabin']) else row['Cabin'],axis=1)
test['Cabin'] = test.apply(lambda row: re.findall(regex,row['Cabin'])[0] if row['Cabin']!=0 else 0, axis=1)
test['Cabin'] = test.apply(lambda row: np.mean(data[data['Cabin']==row['Cabin']]['Age']),axis=1)



#test = test.drop('Cabin',axis=1)

#####Drop columns, and make one hot encoding for categorical variables#######
test = test.drop(['PassengerId','Name','Ticket','Embarked'],axis=1)
dummy1 = pd.get_dummies(test['Sex'])#---------------------->Create one hot encoding
test = pd.concat([test,dummy1],axis=1)
test = test.drop(['Sex','male'],axis=1)
#dummy2 = pd.get_dummies(test['Embarked'])
#test = pd.concat([test,dummy2],axis=1)
#test = test.drop(['Embarked','S'],axis=1)
#test['Cheap'] = test.apply(lambda row:1 if row['Fare']<40 else 0,axis=1)
#test['unfort_fem'] = test.apply(lambda row:1 if (row['Cheap']==1 and row['female']==1) else 0,axis=1)
test['age>fare'] = test.apply(lambda row:c_log(row['Fare']/row['Age']),axis=1)
dummy2 = pd.get_dummies(test['Pclass'])
test = pd.concat([test,dummy2],axis=1)
test = test.drop(['Pclass',1,2],axis=1)
print(list(test))
#labels = test['Survived']
#train = test.drop('Survived',axis=1)#---------------------->Create train data and labels
data = pd.read_csv('/kaggle/input/titanic/train.csv')#----->Loading Data
data['Cabin'] = data.apply(lambda row: 0 if pd.isnull(row['Cabin']) else row['Cabin'],axis=1)
data['Cabin'] = data.apply(lambda row: re.findall(regex,row['Cabin'])[0] if row['Cabin']!=0 else 0, axis=1)
data['Cabin'] = data.apply(lambda row: np.mean(data[data['Cabin']==row['Cabin']]['Age']),axis=1)
#data = data.drop(['Cabin'],axis=1)#------------------------>Too many missing values
data = data.dropna(subset =['Embarked'], how = 'any')#----->Drop 2 rows
#data = data[data['Fare']<500]
####Imputing the Age Column######
list_age = list(data['Age'])
lower = 0
upper = 100
mu = data['Age'].mean()
sigma = data['Age'].std()
for i in range(len(data)):
    if math.isnan(list_age[i]):
        list_age[i]=stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)        
data['Age'] = list_age#------------------------------------>Impute Data for Age
#data['Child'] = data.apply(lambda row:1 if row['Age']<=12 else 0, axis=1)
#data['Litter'] = data.apply(lambda row: 1 if row['SibSp']>=2 else 0, axis=1)
#data['LC'] = data.apply(lambda row:1 if (row['Child']==1)&(row['Litter']==0) else 0, axis=1)
#data = data.drop(['Child','Litter'],axis=1)

#####Drop columns, and make one hot encoding for categorical variables#######
dummy1 = pd.get_dummies(data['Sex'])#---------------------->Create one hot encoding
data = pd.concat([data,dummy1],axis=1)
data = data.drop(['Sex','male'],axis=1)
data['Litter'] = data.apply(lambda row: 1 if (row['SibSp']>=2)&(row['female']==0) else 0, axis=1)
#data['Cheap'] = data.apply(lambda row:1 if row['Fare']<40 else 0,axis=1)
#data['unfort_fem'] = data.apply(lambda row:1 if (row['Cheap']==1 and row['female']==1) else 0,axis=1)
data['log(fare/age)'] = data.apply(lambda row:row['Fare']/row['Age'],axis=1)
data['log(fare/age)'] = data.apply(lambda row:1 if row['log(fare/age)']>20 else 0, axis=1)
data['company'] = data.apply(lambda row:row['SibSp']+row['Parch'],axis=1)
#data = data.drop(['SibSp','Parch'],axis=1)
#dummy2 = pd.get_dummies(data['Pclass'])
#data = pd.concat([data,dummy2],axis=1)
#data = data.drop(['Pclass',1,2],axis=1)
data['Pclass'] = data.apply(lambda row:np.mean(np.array(data[(data['Pclass']==row['Pclass'])&(data['Embarked']==row['Embarked'])]['Fare'])),axis=1)
data = data.drop(['PassengerId','Name','Ticket','Embarked'],axis=1)


submission1 = pd.DataFrame({'PassengerId':list(passid),'Survived':spl_predict(model_RF,test)})
submission.to_csv('submission20200611-1.csv',index=False)
na_dict = {col:test[col].isna().sum() for col in test.columns}
na_dict
submission2 = pd.DataFrame({'PassengerId':list(passid),'Survived':model_RF.predict(test)})
print(submission1)
def get_prob_fare(fare_invest,fare,tol):
    relevant_rows = fare_invest[(fare_invest['Age']>fare-tol) & (fare_invest['Age']<fare+tol)]
    num = len(relevant_rows[relevant_rows['Survived']==1])
    den = len(relevant_rows)
    try:
        prob = num/den
    except:
        prob = 0
    return prob

def get_fare_pclass(df,pclass):
    relevant_df = df[df['Pclass']==pclass]
    m = np.mean(relevant_df['Fare'])
    return m
get_fare_pclass(train,3)
get_prob_fare(data,36,5)
age_sur = [get_prob_fare(data[data['female']==0],i,4) for i in range(70)]
sns.lineplot(list(range(70)),age_sur)

