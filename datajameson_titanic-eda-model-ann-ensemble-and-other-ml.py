import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='notebook', palette='deep')
import warnings
warnings.filterwarnings('ignore')
import operator
sns.set_context("talk", font_scale = 1, rc={"grid.linewidth": 3})
pd.set_option('display.max_rows', 100, 'display.max_columns', 100)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import tensorflow
train= pd.read_csv('../input/titanic/train.csv') #Trainig data set 
test= pd.read_csv('../input/titanic/test.csv') #Testing data set
#gender = pd.read_csv('dataset/titanic/gender_submission.csv')
train.head()
test.head()
print(len(train))
print(len(test))
print('Train Data Info')
print(train.info())
print('\n')
print('Test Data Info')
print(test.info())
train.describe()
test.describe()
pd.DataFrame([train.isnull().sum(),train.isnull().sum()/len(train)*100]).T.\
rename(columns={0:'Total',1:'Missing Perc'})
pd.DataFrame([test.isnull().sum(),test.isnull().sum()/len(test)*100]).T.\
rename(columns={0:'Total',1:'Missing Perc'})
train[train.Embarked.isnull()]
df = train[train['Fare']<train['Fare'].std()*3]
plt.figure(figsize=(10,10))
sns.boxplot(x=df['Embarked'], y=df['Fare'], data=df,hue=df['Pclass'])
#plt.yticks(range(0,550,50))
plt.show()
train['Embarked'].fillna('C',inplace=True)
survivers = train.Survived
train.drop(["Survived"],axis=1, inplace=True)
master=pd.concat([train,test])
master.Cabin.fillna("N", inplace=True)
master['Cabin'] = master['Cabin'].apply(lambda x:list(str(x))[0].upper())
master.head()
master.groupby('Cabin')['Fare'].describe()
def repl_N(val):
    n = 0
    if val==35:
        n = 'T'
    elif val<=14:
        n = 'G'
    elif 14<val<=26:
        n='F'
    elif 26<val<=39:
        n='A'
    elif 39<val<=53:
        n='E'
    elif 53<val<=80:
        n='D'
    elif 80<val<=115:
        n='C'
    else:
        n='B'
    return n


master_N = master[master['Cabin']=='N']
master_notN = master[~(master['Cabin']=='N')]
master_N['Cabin']=master_N['Fare'].apply(lambda x:repl_N(x))
master = pd.concat([master_N,master_notN])
fare_mean= master[(master['Pclass']==3) & (master['Embarked']=='S')]['Fare'].mean()
master['Fare'].fillna(fare_mean,inplace=True)
missing_value = test[(test.Pclass == 3) & 
                     (test.Embarked == "S") & 
                     (test.Sex == "male")].Fare.mean()
## replace the test.fare null values with test.fare mean
test.Fare.fillna(missing_value, inplace=True)
train = master.sort_values('PassengerId')[:891]
test= master.sort_values('PassengerId')[891:]
train['Survived'] = survivers
pd.DataFrame([master.isnull().sum(),master.isnull().sum()/len(master)*100]).T.\
rename(columns={0:'Total',1:'Missing Perc'})
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1
passenger_test= test['PassengerId']
train.drop('PassengerId',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
def showvalues(ax,m=None):
    for p in ax.patches:
        ax.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),\
                    ha='center', va='center', fontsize=14, color='k', rotation=0, xytext=(0, 7),\
                    textcoords='offset points',fontweight='light',alpha=0.9) 
plot_df= train.groupby('Sex')['Survived'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
plot_df
plt.figure(figsize=(10,8))
col = {1:'#99ff99', 0:'#ff9999'}
ax= sns.barplot(x='Sex',y='percent',data=plot_df,hue='Survived',palette=col)
showvalues(ax)
plt.title('Percentage of Passenger Survived Sex wise', pad=30)
plt.xlabel('Sex')
plt.ylabel('Percentage of Passenger Survived')
leg = ax.get_legend().texts
leg[0].set_text("No")
leg[1].set_text("Yes")
plt.show()
plot_df= train.groupby('Pclass')['Survived'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
plot_df
plt.figure(figsize=(10,8))
col = {1:'#99ff99', 0:'#ff9999'}
ax= sns.barplot(x='Pclass',y='percent',data=plot_df,hue='Survived',palette=col)
showvalues(ax)
plt.title("Percentage of Passenger Survived vs PClass", pad=30)
plt.xlabel("Passenger Class");
plt.ylabel("Percentage of Passenger Survived")
leg = ax.get_legend().texts
leg[0].set_text("No")
leg[1].set_text("Yes")
plt.show()
col = {0:'#99ff99', 1:'#ff9999'}
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='Sex',data=train,y='Age',hue='Survived',palette=col)
leg = ax.get_legend().texts
leg[0].set_text("No")
leg[1].set_text("Yes")
plt.show()
plt.figure(figsize=(20,12))
plt.subplot(1,2,1)
ax=sns.distplot(train['Fare'])
plt.subplot(1,2,2)
ax=sns.boxplot(x='Pclass',data=train,y='Fare',hue='Sex',palette='cool')
ax.set_yscale('log')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='r',shade=True,label='Deceased')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='Survived')
plt.xlabel('Fare')
plt.ylabel('Frequency of Passenger Survived')
plt.subplot(2,1,2)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='r',shade=True,label='Deceased')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='Survived')
plt.xlabel('Age')
plt.ylabel('Frequency of Passenger Survived')
plt.show()
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 
               color='red',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived', 
              )
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25, pad = 40)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)
plt.xlabel("Passenger Class", fontsize = 15,labelpad =20)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);

ax = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",palette = col)
ax = ax.map(plt.hist, "Age", edgecolor = 'white').add_legend()
ax.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)



sns.factorplot(x='Parch',y='Survived',data=train,col='Sex',color='g',ci=95.0)

sns.catplot(x='Parch',y='Survived',data=train,col='Sex',color='g')

sns.factorplot(x='SibSp',y='Survived',data=train,col='Sex',ci=95.0,color='g')

sns.factorplot(x='family_size',y='Survived',data=train,col='Sex',ci=95.0,color='g')
plot_df= train.groupby('Cabin')['Survived'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
plt.figure(figsize=(16,8))
col = {1:'#99ff99', 0:'#ff9999'}
ax= sns.barplot(x='Cabin',y='percent',data=plot_df,hue='Survived',palette=col)
showvalues(ax)
plt.title("Percentage of Passenger Survived vs Cabin", pad=30)
plt.xlabel("Cabin");
plt.ylabel("Percentage of Passenger Survived")
leg = ax.get_legend().texts
leg[0].set_text("No")
leg[1].set_text("Yes")
plt.show()

sns.factorplot(x='Embarked',y='Survived',data=train,col='Sex',ci=95.0,color='g')

sns.factorplot(x='Cabin',y='Survived',data=train,col='Embarked',hue='Sex',ci=95.0,size=6)
plt.figure(figsize=(20,10))
ax=sns.boxplot(x='Cabin',data=train,y='Fare',hue='Sex',palette='cool')
ax.set_yscale('log')
plt.show()

plot_df= train[train['Age']<10]['Survived'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
plt.figure(figsize=(10,8))
ax=sns.barplot(x=plot_df['index'],y=plot_df['percent'],palette=col)
locs, labels = plt.xticks()
plt.xticks(ticks=locs,labels=['Not Survived','Survived'])
plt.ylabel('Percentage of Children Survived in Age < 10')
plt.xlabel('Survived or Not Survived')
showvalues(ax)
train['Survived'].value_counts(normalize=True)*100
train[(train['Embarked']=='Q') & (train['Sex']=='male')]['Survived'].value_counts()
train[(train['Embarked']=='Q') & (train['Sex']=='male')]['Survived'].value_counts(normalize=True)*100
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)
plot_df
plot_df=train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',
       'family_size']]
plt.figure(figsize=(15,10))
mask = np.zeros_like(plot_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(plot_df.corr(),annot=True,cmap='cividis',mask=mask)
np.info([len(i) for i in train.Name])
train['name_len'] = [len(i) for i in train.Name]
test['name_len'] = [len(i) for i in test.Name]
def name_length(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=50):
        a = 'long'
    else:
        a = 'very long'
    return a
train['name_len_rnge'] = train['name_len'].map(name_length)
test['name_len_rnge'] = test['name_len'].map(name_length)
plt.figure(figsize=(10,8))
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'name_len'] , color='r',shade=True,label='Deceased')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'name_len'] , color='g',shade=True, label='Survived')
plt.xlabel('Name Length')
plt.ylabel('Frequency of Passenger Survived')
plt.show()
sns.distplot([len(i) for i in train.Name])
train['title']=train['Name'].apply(lambda x:x.split('.')[0].split(',')[1].strip())
test['title']=test['Name'].apply(lambda x:x.split('.')[0].split(',')[1].strip())
## we are writing a function that can help us modify title column
def replace_title(df):
    
    result=[]
    for val in df:
        if val in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col','Dona']:
            val = 'rare'
            result.append(val)
        elif val in ['Ms', 'Mlle']:
            val = 'Miss'
            result.append(val)
        elif val == 'Mme':
            val = 'Mrs'
            result.append(val)
        else:
            result.append(val)
    return result

train['title']=replace_title(train['title'])
test['title']=replace_title(test['title'])
train['title'].value_counts()
print(train['title'].unique())
print(test['title'].unique())
## bin the family size. 
def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """
    
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a

train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
train['actual_fare']=train['Fare']/train.family_size
test['actual_fare'] = test.Fare/test.family_size
train['Fare'].describe()
def fare_rnge(fare):
    val= ''
    if fare <= 4:
        val = 'very_low'
    elif fare <= 10:
        val = 'low'
    elif fare <= 20:
        val = 'mid'
    elif fare <= 45:
        val = 'high'
    else:
        val = 'very_high'
    return val

train['fare_rnge'] = train['actual_fare'].map(fare_rnge)
test['fare_rnge'] = test['actual_fare'].map(fare_rnge)
## create bins for age
#def age_group_fun(age):
#    """
#    This function creates a bin for age
#    """
#    a = ''
#    if age <= 1:
#        a = 'infant'
#    elif age <= 4: 
#        a = 'toddler'
#    elif age <= 13:
#        a = 'child'
#    elif age <= 18:
#        a = 'teenager'
#    elif age <= 35:
#        a = 'Young_Adult'
#    elif age <= 45:
#        a = 'adult'
#    elif age <= 55:
#        a = 'middle_aged'
#    elif age <= 65:
#        a = 'senior_citizen'
#    else:
#        a = 'old'
#    return a
        
## Applying "age_group_fun" function to the "Age" column.
#train['age_group'] = train['Age'].map(age_group_fun)
#test['age_group'] = test['Age'].map(age_group_fun)

## Creating dummies for "age_group" feature. 
#train = pd.get_dummies(train,columns=['age_group'], drop_first=True)
#test = pd.get_dummies(test,columns=['age_group'], drop_first=True);
train = pd.get_dummies(train, columns=['Pclass', 'Cabin', 'Embarked', 'name_len_rnge', 'title',\
                                       'fare_rnge','family_group'], drop_first=False)
test = pd.get_dummies(test, columns=['Pclass', 'Cabin', 'Embarked', 'name_len_rnge',\
                                     'title','fare_rnge','family_group'], drop_first=False)

train.drop(['family_size','name_len',\
            'Fare','Name','Ticket'], axis=1, inplace=True)
test.drop(['family_size','name_len',\
            'Fare','Name','Ticket'], axis=1, inplace=True)
def predict_age(df):
    df_not_null = df.loc[df['Age'].notnull()]
    df_null= df[df['Age'].isnull()]
    y=df_not_null['Age']
    x=df_not_null.drop('Age',axis=1)
    rf_reg=RandomForestRegressor(n_estimators=1000).fit(x,y)
    pred=rf_reg.predict(df_null.drop('Age',axis=1))
    df.loc[df.Age.isnull(), "Age"] =list(pred)
    return df
predict_age(train)
predict_age(test)
    
X = train.drop(['Survived'], axis = 1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(solver='liblinear',penalty= 'l1',random_state = 0)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred)) 
from sklearn.metrics import roc_auc_score,auc,roc_curve
fpr, tpr, _ =roc_curve(y_test,y_prob[:,1])
roc_auc= auc(fpr,tpr)
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label='ROC Curve(area = %0.2f)'%roc_auc)
plt.plot([0,1],[0,1],'k--',c='r')
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC Curve', fontsize= 18)
plt.show()
precision,recall,thre=precision_recall_curve(y_test,y_prob[:,1])
prec_recall= auc(recall,precision)
plt.figure(figsize=(10,8))
plt.plot(recall,precision,label='Precision recall Curve(area = %0.2f)'%prec_recall)
plt.xlabel('recall', fontsize = 18)
plt.ylabel('precision', fontsize = 18)
plt.title('Precision Recall', fontsize= 18)
plt.show()
cv=StratifiedShuffleSplit(n_splits=20,test_size=0.3,random_state=0)
cross_v_score= cross_val_score(LogisticRegression(),X,y,cv=cv)
print(cross_v_score)
print('mean cross validation score:{0:2.2f}'.format(np.mean(cross_v_score)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold

c = list(np.linspace(0.01,10,19))
penalties = ['l1','l2']
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .3)
param = {'penalty': penalties, 'C': c}
logreg = LogisticRegression(solver='liblinear')
grid = RandomizedSearchCV(estimator=LogisticRegression(), 
                           param_distributions = param,
                           scoring = 'accuracy',
                           cv = cv,n_iter=40
                          )
## Fitting the model
grid.fit(X, y)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_index_)
lr=grid.best_estimator_
lr.score(X,y)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)
accuracy_score(y_test,y_predict)
from sklearn.naive_bayes import MultinomialNB
g_nb=MultinomialNB()
g_nb.fit(X,y)
y_pred= g_nb.predict(X_test)
print(round(accuracy_score(y_test,y_pred),3))

from sklearn.svm import SVC
svm_n=SVC(C=3,kernel='poly',degree=3)
svm_n.fit(X_train,y_train)
y_pred= svm_n.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier
max_depth_n = range(1,10)
max_feature_n = [20,21,22,23,24,25,26,28,29,30,'auto']
criterion_n = ["gini", "entropy"]
params={'max_depth':max_depth_n,'max_features':max_feature_n,'criterion':criterion_n}
cv_n=StratifiedShuffleSplit(test_size=0.25,random_state=0)
random_cv= RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=params,cv=cv_n)
random_cv.fit(X,y)
print(random_cv.best_estimator_)
print(random_cv.best_index_)
print(random_cv.best_params_)
print(random_cv.best_score_)

dtc=random_cv.best_estimator_
dtc.score(X,y)

columns= X.columns
feature_importances = pd.DataFrame(dtc.feature_importances_,
                                   index = columns,
                                    columns=['Feature Importance'])
feature_importances.sort_values(by='Feature Importance', ascending=False).head(10)
df_temp= feature_importances.sort_values(by='Feature Importance', ascending=False).head(10)
plt.figure(figsize=(8,6))
sns.barplot(data=df_temp,y=df_temp.index, x='Feature Importance',orient='h')
#bar.set_xticklabels(bar.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
from sklearn.ensemble import RandomForestClassifier
n_estimators_n=[145,150]
max_depth_n=range(1,10)
criterion_n = ["gini", "entropy"]
params={'max_depth':max_depth_n,'criterion':criterion_n,'n_estimators':n_estimators_n}
cv_n=StratifiedShuffleSplit(test_size=0.25,random_state=0)
grid_cv= GridSearchCV(RandomForestClassifier(),param_grid=params)
grid_cv.fit(X,y)
print(grid_cv.best_estimator_)
print(grid_cv.best_params_)
print(grid_cv.best_score_)

rfc=grid_cv.best_estimator_
rfc.score(X,y)
columns= X.columns
feature_importances = pd.DataFrame(rfc.feature_importances_,
                                   index = columns,
                                    columns=['Feature Importance'])
feature_importances.sort_values(by='Feature Importance', ascending=False).head(10)
df_temp= feature_importances.sort_values(by='Feature Importance', ascending=False).head(10)
plt.figure(figsize=(8,6))
sns.barplot(data=df_temp,y=df_temp.index, x='Feature Importance',orient='h')
#bar.set_xticklabels(bar.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
from sklearn.ensemble import BaggingClassifier
n_estimators_n = [10,20,30,50,70,80,100,120, 140,150]
cv_n=StratifiedShuffleSplit(test_size=0.25,random_state=0)
params={'n_estimators':n_estimators_n}
grid_cv= GridSearchCV(BaggingClassifier(),param_grid=params,cv=cv_n)
grid_cv.fit(X,y)
print(grid_cv.best_estimator_)
print(grid_cv.best_params_)
print(grid_cv.best_score_)
bc_n=grid_cv.best_estimator_
bc_n.score(X,y)
bc=BaggingClassifier(n_estimators=30,max_features=17)
bc.fit(X_train,y_train)
y_pred=bc.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.ensemble import GradientBoostingClassifier
max_depth_n=range(2,10)
params={'max_depth':max_depth_n}
cv_n=StratifiedShuffleSplit(test_size=0.25,random_state=0)
grid_cv= GridSearchCV(GradientBoostingClassifier(),param_grid=params)
grid_cv.fit(X,y)
print(grid_cv.best_estimator_)
print(grid_cv.best_params_)
print(grid_cv.best_score_)
gbc=grid_cv.best_estimator_
gbc.score(X,y)
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
y_pred=xgbc.predict(X_test)
print(round(accuracy_score(y_test,y_pred),2))
from sklearn.ensemble import AdaBoostClassifier
n_estimators_n = [50,70,80,100]
cv_n=StratifiedShuffleSplit(test_size=0.25,random_state=0)
params = {'n_estimators':n_estimators_n}
grid_cv= GridSearchCV(AdaBoostClassifier(),param_grid=params,cv=cv_n)
grid_cv.fit(X,y)
print(grid_cv.best_estimator_)
print(grid_cv.best_score_)
print(grid_cv.best_params_)
abc_n=grid_cv.best_estimator_
abc_n.score(X,y)
abc=AdaBoostClassifier(algorithm='SAMME.R',learning_rate=1.007)
abc.fit(X_train,y_train)
y_pred=abc.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier()
etc.fit(X_train,y_train)
y_pred=etc.predict(X_test)
print(round(accuracy_score(y_test,y_pred),2))
from sklearn.gaussian_process import GaussianProcessClassifier
gpc=GaussianProcessClassifier()
gpc.fit(X_train,y_train)
y_pred=gpc.predict(X_test)
print(round(accuracy_score(y_test,y_pred),2))
from sklearn.ensemble import VotingClassifier
vc= VotingClassifier(estimators=[lr,rfc,gbc,knn,bc,abc,etc,gpc,g_nb])

vc = VotingClassifier(estimators=[
    ('lr_grid', lr),
    ('random_forest', rfc),
    ('gradient_boosting', gbc),
    ('decision_tree_grid',dtc),
    ('knn_classifier', knn),
    ('XGB_Classifier', xgbc),
    ('bagging_classifier', bc),
    ('adaBoost_classifier',abc),
    ('ExtraTrees_Classifier', etc),
    ('gaussian_process_classifier', gpc)
],voting='hard')
vc.fit(X_train,y_train)
y_pred=vc.predict(X_test)
print(round(accuracy_score(y_test,y_pred),2))
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
#Early Stop
early_stop = EarlyStopping(monitor = 'val_loss', mode = "min", verbose = 1 , patience = 25)

ann= Sequential()

ann.add(Dense(9,activation = 'relu'))
ann.add(Dropout(0.5))
ann.add(Dense(4,activation = 'relu'))
ann.add(Dropout(0.5))

ann.add(Dense(1,activation='sigmoid'))
ann.compile(loss= 'binary_crossentropy', optimizer = 'adam')

ann.fit(x=X_train, y=y_train,epochs=400,validation_data=(X_test,y_test),callbacks=[early_stop])
y_pred = (ann.predict(X_test) > 0.46).astype(int)
accuracy_score(y_test,y_pred)
models = [lr,knn,svm_n,dtc,rfc,gbc,bc,abc,etc,gpc,xgbc,vc]
c = {}
for model in models:
    pred = model.predict(X_test)
    result = accuracy_score(y_test,pred)
    c[model] = result
    
c
