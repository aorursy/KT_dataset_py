import numpy as np

import pandas as pd

import matplotlib.pyplot as plt       #Data Visualization

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import train_test_split

from sklearn import metrics



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



from sklearn.ensemble import VotingClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

import statsmodels.api as sm
df1=pd.read_csv('../input/train.csv')

df1.head()
df2=pd.read_csv('../input/test.csv')

df2.head()
df1.shape,df2.shape
print ("Rows     : " ,df1.shape[0])

print ("Columns  : " ,df1.shape[1])

print ("\nFeatures : \n" ,df1.columns.tolist())

print ("\nMissing values :  ", df1.isnull().sum())

print ("\nUnique values :  \n",df1.nunique())

print ("\nData types : \n",df1.dtypes)
df1['Survived'].value_counts()
print(df1['Survived'].value_counts(normalize=True))

df1['Survived'].value_counts().plot.bar()
print("Pclass \n",df1['Pclass'].value_counts(normalize=1),"\n")

# there are different proportion of people belonging to each class

print("Sex \n",df1['Sex'].value_counts(normalize=1),"\n")

# females are present in higher proportion than males

print("SibSp \n",df1['SibSp'].value_counts(normalize=1),"\n")

# very less people with either their sibling or spouse are travelling

print("Parch \n",df1['Parch'].value_counts(normalize=1),"\n")

# people travelling with their parent or child are less as compared to those without them

print("Embarked \n",df1['Embarked'].value_counts(normalize=1),"\n")

#Most people embarked Port of Embarkation Southampton
plt.figure(figsize=(12,8))



plt.subplot(221) 

df1['SibSp'].value_counts().plot.bar(title='SibSp')



plt.subplot(222) 

df1['Parch'].value_counts().plot.bar(title='Parch')



plt.show()
plt.figure(figsize=(12,8))



plt.subplot(221) 

df1['Pclass'].value_counts().plot.bar(title='Pclass')



plt.subplot(222) 

df1['Sex'].value_counts().plot.bar(title='Sex')



plt.subplot(223) 

df1['Embarked'].value_counts().plot.bar(title='Embarked')



plt.show()

df1[['Age','Fare']].describe().T
sns.heatmap(df1[['Age','Fare']].corr())
plt.figure(figsize=(15,10))



plt.subplot(221)

sns.distplot(df1['Age'].dropna())



plt.subplot(222)

sns.boxplot(df1['Age'])



plt.subplot(223)

sns.distplot(df1['Fare'])



plt.subplot(224)

sns.boxplot(df1['Fare'])



plt.show()
pd.crosstab(df1['Survived'],df1['Pclass'], normalize=1).plot.bar()
pd.crosstab(df1['Survived'],df1['Sex'], normalize=1).plot.bar()
pd.crosstab(df1['Survived'],df1['SibSp'], normalize=1).plot.bar()



#Those having higher number of siblings/spouse didnt survive
pd.crosstab(df1['Survived'],df1['Parch'], normalize=1).plot.bar()



#People with higher number of parent/child didnt survive
pd.crosstab(df1['Survived'],df1['Embarked'], normalize=1).plot.bar()



#Proportion of poeople survived is higher for embarked=C
sns.boxplot(x='Survived', y='Age', data=df1)



#There isnt much difference between median age of people who survived and not survived, but overall, more younger people survived
sns.boxplot(x='Survived', y='Fare', data=df1)



#People who survived paid higher fare
df2['Survived']='to_check'

data = pd.concat([df1, df2], axis=0)

data.head()
import copy

data1=copy.deepcopy(data)

print ("Missing values :  \n", data1.isnull().sum())
data1.shape
data1=data1.drop('Cabin', axis=1)

print('Mode of Embarked : ',(data1['Embarked']).mode()[0])
data1["Embarked"].fillna(value="S", inplace=True)

data1["Fare"].fillna(value=np.median(data1["Fare"].dropna()), inplace=True)

data1['Fare'].isnull().sum()
req=data1["Name"].str.split(", ",expand=True)

req=req[1].str.split(". ",expand=True)

req.head()
data1["Title"]=req[0]

data1["Title"].value_counts()
pd.crosstab(data1["Age"].isnull(), data1["Title"])
data1_title=data1.groupby("Title")

data1.groupby("Title")["Age"].median()
data1['Age'].fillna(data1.groupby(['Title'])['Age'].transform(np.median),inplace=True)

pd.crosstab(data1["Age"].isnull(), data1["Title"])
data1.isnull().sum()
data1['Age'] = (data1['Age']-np.min(data1['Age']))/(np.max(data1['Age'])-np.min(data1['Age']))

data1['Fare'] = (data1['Fare']-np.min(data1['Fare']))/(np.max(data1['Fare'])-np.min(data1['Fare']))

data1.head()
data2 = data1.drop(['Ticket','PassengerId','Name'], axis=1)

data2.head()
data2["Family"]=data2["SibSp"]+data2["Parch"]

data3=data2.drop(['SibSp','Parch'], axis=1)
col=["Sex","Embarked","Title","Pclass"]

l=pd.get_dummies(data=data3, columns=col, drop_first=True)

l.head()
l.columns
train=l[l['Survived']!='to_check']

test=l[l['Survived']=='to_check'].drop('Survived', axis=1)
# UDF for calculating vif value

def vif_cal(input_data, dependent_col):

    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared  

        vif=round(1/(1-rsq),2)

        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)

vif_cal(train, 'Survived')
vif_cal(train.drop(['Title_Dona'],axis=1), 'Survived')
vif_cal(train.drop(['Title_Dona','Sex_male'],axis=1), 'Survived')
vif_cal(train.drop(['Title_Dona','Sex_male','Age',],axis=1), 'Survived')
vif_cal(train.drop(['Title_Dona','Sex_male','Age','Title_Miss'],axis=1), 'Survived')
x = train.drop(['Title_Dona','Sex_male','Age','Title_Miss','Survived'], axis=1)

y = train['Survived'].astype('int')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

x_train.shape, y_train.shape
x_final=test.drop(['Title_Dona','Sex_male','Age','Title_Miss'], axis=1)
test1=copy.deepcopy(x_test)

test1['Survived']=train['Survived'].mode()[0]

metrics.accuracy_score(test1['Survived'],y_test)
lr = LogisticRegression()

model_lr = lr.fit(x_train,y_train)

pred_lr = model_lr.predict(x_test)

metrics.accuracy_score(y_test,pred_lr)
dtree = DecisionTreeClassifier()

model_dtree = dtree.fit(x_train,y_train)

pred_dtree = model_dtree.predict(x_test)

metrics.accuracy_score(y_test, pred_dtree)
knn = KNeighborsClassifier()

model_knn = knn.fit(x_train,y_train)

pred_knn = model_knn.predict(x_test)

metrics.accuracy_score(y_test, pred_knn)
rf = RandomForestClassifier()

model_rf = rf.fit(x_train,y_train)

pred_rf = model_rf.predict(x_test)

metrics.accuracy_score(y_test, pred_rf)
gb=GaussianNB()

model_gb = gb.fit(x_train,y_train)

pred_gb = model_gb.predict(x_test)

metrics.accuracy_score(y_test, pred_gb)
svm = SVC(kernel='linear') 

svm.fit(x_train,y_train)

pred_svm = svm.predict(x_test)

metrics.accuracy_score(y_test, pred_svm)
voting_model = VotingClassifier([('lr', lr),('dtree', dtree),('knn',knn),('rf', rf),('gb',gb),('svm',svm)], voting='hard')

voting_model.fit(x_train, y_train)

voting_predict = voting_model.predict(x_test)

metrics.accuracy_score(y_test, voting_predict)
model_db=pd.DataFrame()

score=[]

model_name=[]

for i in [lr,dtree,knn,rf,gb,svm]:

    Scores = cross_val_score(i, x, y, scoring='accuracy', cv=20)

    score.append(Scores.mean())

    model_name.append(i)

    

model_db["Model"]=model_name

model_db["Scores"]=score

model_db
voting_model = VotingClassifier([('lr', lr),('dtree', dtree),('knn',knn),('rf', rf)], voting='hard')

voting_model.fit(x_train, y_train)

voting_predict = voting_model.predict(x_test)

metrics.accuracy_score(y_test, voting_predict)
df=pd.read_csv('../input/gender_submission.csv')

df.head()
df['Survived']=voting_model.predict(x_final)

df.head()
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(df)
dtree_param = {

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':[2,3,4,5,6,7,8],

    'min_samples_split':[10,20,30,40,50],

    'max_features':['auto','sqrt','log2'],

    'random_state':[123]

}



grid_search = GridSearchCV(estimator=dtree, param_grid=dtree_param, cv=5)

cv_grid = grid_search.fit(x_train,y_train)

cv_grid.best_params_
dtree_tuned = DecisionTreeClassifier(criterion='gini',max_depth=6,max_features='auto',

                                     min_samples_split=40, random_state=123,splitter='random')

model_dtree_tuned = dtree_tuned.fit(x_train,y_train)

pred_dtree_tuned = model_dtree_tuned.predict(x_test)

metrics.accuracy_score(y_test, pred_dtree_tuned)
confusion = metrics.confusion_matrix(model_lr.predict(x_test), y_test)

print(confusion)

print(metrics.accuracy_score(model_lr.predict(x_test), y_test))
confusion = metrics.confusion_matrix(svm.predict(x_test), y_test)

print(confusion)

print(metrics.accuracy_score(svm.predict(x_test), y_test))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 6))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds

draw_roc(model_lr.predict(x_test), y_test)
#draw_roc(y_pred_final.Churn, y_pred_final.predicted)

"{:2.2f}".format(metrics.roc_auc_score(model_lr.predict(x_test), y_test))

'0.91'

draw_roc(svm.predict(x_test), y_test)
#draw_roc(y_pred_final.Churn, y_pred_final.predicted)

"{:2.2f}".format(metrics.roc_auc_score(svm.predict(x_test), y_test))

print(metrics.classification_report(y_test,svm.predict(x_test) ))

prediction=model_lr.predict(x_final)

prediction
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,VotingClassifier

from xgboost import XGBClassifier
bagg = BaggingClassifier()

bagg_param = {

    'n_estimators':[5,10,15,20,25],

    'oob_score':[True,False],

    'random_state':[123]

}



grid_search = GridSearchCV(estimator=bagg, param_grid=bagg_param, cv=5)

cv_grid = grid_search.fit(x_train,y_train)

cv_grid.best_params_
bagg_lr = BaggingClassifier(base_estimator=lr, n_estimators=25, bootstrap=True, oob_score=True, random_state=123)

model_bagg_lr = bagg_lr.fit(x_train,y_train)

pred_bagg_lr = model_bagg_lr.predict(x_test)

x_re=metrics.accuracy_score(y_test, pred_bagg_lr)

x_re
bagg_sv = BaggingClassifier(base_estimator=svm, n_estimators=25, bootstrap=True, oob_score=True, random_state=123)

model_bagg_sv = bagg_sv.fit(x_train,y_train)

pred_bagg_sv = model_bagg_sv.predict(x_test)

x_re=metrics.accuracy_score(y_test, pred_bagg_sv)

x_re
adb_lr = AdaBoostClassifier(lr, n_estimators=10, learning_rate=1)

adb_lr.fit(x_train,y_train)

pred_adb_lr = adb_lr.predict(x_test)

x_re=metrics.accuracy_score(y_test,pred_adb_lr)

x_re
xgb=XGBClassifier()

model_xgb = xgb.fit(x_train,y_train)

pred_xgb = model_xgb.predict(x_test)

x_re=metrics.accuracy_score(y_test, pred_xgb)

x_re
voting_model = VotingClassifier([('lr', lr),('dtree', dtree),('knn',knn),('rf', rf),

                                 ('model_bagg_lr', model_bagg_lr),

                                ('model_bagg_sv',model_bagg_sv),('adb_lr',adb_lr),

                                 ('model_xgb',model_xgb)], voting='hard')

voting_model.fit(x_train, y_train)

voting_predict = voting_model.predict(x_test)

metrics.accuracy_score(y_test, voting_predict)
df['Survived']=voting_model.predict(x_final)

df.head()
create_download_link(df)
df['Survived']=xgb.predict(x_final)

df.head()
create_download_link(df)