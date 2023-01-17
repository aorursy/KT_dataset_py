
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score
df=pd.read_csv('bank.csv')
df.head()
df.tail()
df.columns
df.isnull().sum()
df.deposit.value_counts()
df.education.dtype
sns.distplot(df.age,bins=10)
from scipy.stats import kurtosis
from scipy.stats import skew

K=kurtosis(df['age'])
s=skew(df['age'])
print('k:',K)
print('s:',s)
df['age']=df['age'].apply(np.log)
    
sns.distplot(df['age'])
sns.boxplot(df.age)
sns.distplot(df.balance)
sns.boxplot(df.balance)
df.balance.describe()
K=kurtosis(df['balance'])
s=skew(df['balance'])
print('k:',K)
print('s:',s)
df['balance']=df['balance'].apply(np.cbrt)
    
sns.distplot(df['balance'])
sns.distplot(df.day)
sns.boxplot(df.day)
sns.distplot(df.pdays,bins=10)
df['pdays']=df['pdays'].apply(np.log)
    
sns.distplot(df['pdays'])
sns.boxplot(df.pdays)
sns.distplot(df.duration,bins=10)
df['duration']=df['duration'].apply(np.log)
    
sns.distplot(df['duration'])
sns.boxplot(df.duration)
sns.distplot(df.campaign,bins=5)
df['campaign']=df['campaign'].apply(np.log)
    
sns.distplot(df['campaign'])
sns.boxplot(df.campaign)
sns.distplot(df.previous)
df['previous']=df['previous'].apply(np.cbrt)
    
sns.distplot(df['previous'])
sns.boxplot(df.previous)
df.columns
df.poutcome.value_counts()
df.month.value_counts().plot()
df.deposit.value_counts()
depositmapping={'yes':1,'no':0}
df.deposit=df.deposit.map(depositmapping)
df.deposit.value_counts()
df[['job','deposit']].groupby('job').mean().sort_values('deposit',ascending=True)
df['job']=df['job'].replace(['management','technician','unknown','admin.','housemaid','self-employed','services',
                                'blue-collar','entrepreneur'],'rare',regex=True)
jobmapping={'student':3,'retired':2,'unemployed':1,'rare':0}
df['job']=df['job'].map(jobmapping)
df['job'].value_counts()
df.columns
df[['marital','deposit']].groupby('marital').mean().sort_values('deposit',ascending=True)
statusmapping={'married':1,'divorced':2,'single':3}
df['marital']=df['marital'].map(statusmapping)

df['marital']
df[['education','deposit']].groupby('education').mean().sort_values('deposit',ascending=True)
educationmapping={'primary':1,'secondary':2,'unknown':3,'tertiary':4}
df['education']=df['education'].map(educationmapping)
df['education']
df.deposit
df[['deposit','default']].groupby('default').mean().sort_values('deposit',ascending=True)
df.columns
defaultmapping={'no':1,'yes':2}
df['default']=df['default'].map(defaultmapping)
df.loan.value_counts()
df[['deposit','loan']].groupby('loan').mean().sort_values('deposit',ascending=True)
loanmapping={'no':1,'yes':2}
df['loan']=df['loan'].map(loanmapping)
df[['deposit','contact']].groupby('contact').mean().sort_values('deposit',ascending=True)
contactmapping={'unknown':1,'telephone':2,'cellular':3}
df['contact']=df['contact'].map(contactmapping)
df['contact'].value_counts()
df[['deposit','poutcome']].groupby('poutcome').mean().sort_values('deposit',ascending=True)
poutcomemap={'unknown':1,'failure':2,'other':3,'success':4}
df['poutcome']=df['poutcome'].map(poutcomemap)
df.head()
df[['deposit','month']].groupby('month').mean().sort_values('deposit',ascending=True)
df['month']=df['month'].replace(['mar','dec','sep','oct'],2,regex=True)
df['month']=df['month'].replace(['apr','feb','aug','jun'],1,regex=True)
df['month']=df['month'].replace(['nov','jul','jan','may'],0,regex=True)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


X = df[['age', 'job', 'marital', 'education', 'default', 'balance',
       'loan', 'contact', 'month', 'duration', 'campaign', 
        'poutcome']]  #independent columns
y = df['deposit']    #target column i.e price range
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

nrows=len(df.index)
percentage=round((nrows*70)/100)
trainingData=df.iloc[:percentage,:]
testData=df.iloc[percentage:,:]

print("Number of training data examples "+str(len(trainingData.index)))
print("Number of test examples "+str(len(testData.index)))

train_x=trainingData[['age','loan','month','poutcome','balance','campaign','contact','duration','education']]
train_y=trainingData["deposit"]

test_x=testData[['age','loan','month','poutcome','balance','campaign','contact','duration','education']]
test_y=testData["deposit"]

train_x.head()

#featureNames=["job","marital","education","age","balance","day","pdays","duration"]
#classNames=[1,0]
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(train_x, train_y)
y_pred_gini = clf_gini.predict(test_x)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf_gini,filled=True)
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(test_y, y_pred_gini)))
y_pred_train_gini = clf_gini.predict(train_x)

y_pred_train_gini
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(train_y, y_pred_train_gini)))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_y,y_pred_gini))
cm=confusion_matrix(test_y,y_pred_gini)
print(cm)
print ("Accuracy of prediction:",round((cm[0,0]+cm[1,1])/cm.sum(),3))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(train_x, train_y)

rfc_pred = rfc.predict(test_x)


print(classification_report(test_y,rfc_pred))
cm=confusion_matrix(test_y,rfc_pred)
print(cm)
print ("Accuracy of prediction:",round((cm[0,0]+cm[1,1])/cm.sum(),3))
