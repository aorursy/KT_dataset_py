# 1) Using RandomForest model as introduced by the Tute
# 2) Estimated the missing Age and hence implemented them for tranning
# 3) Some interesting inference:
#    Sex=female->survived-Up/Higher Pclass ->survived-Up
#    Rev are all male, with 0 survival rate (they serve the Lord), although only 7 Rev on the ship.
#    Male Dr. can have higher survival rate (they are welcomed by lifeboat/they know to to survice) 
#    Master are all male and they have higher survival rate than usual male
#    Married woman (Mrs) has survival rate 10% higher than sigle lady (Miss)
#P1 Database liabaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
#P2 Machine learning liabaries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#P3 Load the Data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# 2.1 Select the label and features in train_data for further analysis
col_take=['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked']
dfC=train_data[col_take].copy() # check point ->dfC to be cleansed
# 2.2 Turn the Sex into binary value
dfC.Sex=train_data['Sex'].map({'male':0,'female':1})
# 2.3 Calculate the average age in different sub-groups (Sex and Pclass)
age_median=train_data.groupby(['Sex','Pclass']).Age.median()
display(age_median)
#2.4 A func to estimate the missing value by averaging the not null value in sub_group
def group_fill(rec,median):
    sex=rec['Sex']
    Pclass=rec['Pclass']
    ind=tuple((sex,Pclass))
    est=median[ind]
    return est
#2.5 Estimate the missing age and fill the Null age
ageEst=train_data[train_data['Age'].isnull()].apply(lambda x: group_fill(x,age_median),axis=1)
dfC.loc[:,'Age'].fillna(value=ageEst,inplace=True)
#3.5 Extract title of passenger name
title=train_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0]).copy()
title=title.str.strip()
dfC['Title']=title
#2.6 Repeat the same process to the test data
#1 select the features
col_take_T=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked'] 
dfC_T=test_data[col_take_T]
#2 turn Sex into binary
dfC_T.Sex=test_data['Sex'].map({'male':0,'female':1})
#3 Averaging age in each subgroup
age_medianT=test_data.groupby(['Sex','Pclass']).Age.median()
#4 Apply the 'short func' to fill the missing age
ageEst_test=test_data[test_data['Age'].isnull()].apply(lambda x: group_fill(x,age_medianT),axis=1)
dfC_T.loc[:,'Age'].fillna(value=ageEst_test,inplace=True)
#5 Extract the title from passenger name
title_T=test_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0]).copy()
title_T=title_T.str.strip()
dfC_T['Title']=title_T
dfC_T.Title[0]
#2.7 Special case found-> one of the fare is missing
# dfC_T.isna().any().sum()
# Estimate by median fare in each sub-group( Male's is cheaper, lower class is cheaper)
fare_medianT=test_data.groupby(['Sex','Pclass']).Fare.median()
est_Fare=test_data[test_data['Fare'].isnull()].apply(lambda x: group_fill(x,fare_medianT),axis=1)
dfC_T.loc[:,'Fare'].fillna(value=est_Fare,inplace=True)
display(dfC_T)
#3.1 Check the correlation
# 1. The age is not correlated to survival rate
# 2. The Pclass and Sex are inportant factor, so is the fare (but the fare is also correlated to Class)
# 3. SibSp and Parch not very correlated to survival (may need some feature engineering?)
# 4. Par and SibSp are highly correlated to each otder
data_exp=dfC.corr()
f=plt.figure(figsize=(10,5))
ax=f.add_subplot(111)
sns.heatmap(data_exp,ax=ax,vmin=-1,vmax=1,annot=True,annot_kws={"size": 20},cmap='RdBu')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 16)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 16)
## 3.2 Use 4 type of people instead of Sex, normal Mr, Master, Mrs, Miss, 
# display(dfC.Title.unique())
test=dfC.groupby(['Sex','Title']).count()
l1=[]
l2=[]
for i in test.index:
    if i[0] == 0:
        l1.append(i[1])
    else:
        l2.append(i[1])
l1.remove('Master')
l2.remove('Mrs')
title_dict={'Mr':l1,'Master':'Master','Miss':l2,'Mrs':'Mrs'}
title_dict
dfC.Title.map(title_dict)
dfC.groupby(['Sex','Title'])[['Survived']].mean()
title_dict
# dfC.groupby(['Title'])[['Survived']].mean() 
# title_surv=dfC[dfC['Survived']==1].groupby(['Title'])['Survived'].count()
# title_pass=dfC[dfC['Survived']==0].groupby(['Title'])['Survived'].count()
# f2=plt.figure(figsize=(10,5))
# ax2=f2.add_subplot(111)
# ax2.bar(title_surv.index,title_surv)
# ax2.bar(title_pass.index,title_pass)
#3.1 Call the solvers
std=StandardScaler()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
#3.2 Standarise the data
col_for_std=['Pclass','Age','SibSp','Parch','Fare']
std.fit(dfC[col_for_std])
dfC_std=dfC.copy()
dfC_std[col_for_std]=std.transform(dfC[col_for_std])
Y=dfC_std['Survived']
# Feature selection
X=dfC_std.iloc[:,1:] #-> can skip the Fare
fs=['Pclass','Sex','Fare','Age']
X=dfC_std.loc[:,fs]
X
#3.3 Sovle
model.fit(X,Y)
model.score(X,Y)
#3.4 Validate the test_data
dfC_T_std=dfC_T.copy()
# Transform the validating data to the same scale as that in train_data
dfC_T_std[col_for_std]=std.transform(dfC_T_std[col_for_std])
# Feature selection
# X_T=dfC_T_std.iloc[:,0:]#-> can skip the Fare
X_T=dfC_T_std.loc[:,fs]
X_T
# 3.5 Predict the labels of test_data
Y_pred=model.predict(X_T)
# y_out=pd.DataFrame(index=test_data.PassengerId,columns=['Survived'],data=Y_pred)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})
output
#3.5  Output the data
output.to_csv('my_submission.csv', index=False)