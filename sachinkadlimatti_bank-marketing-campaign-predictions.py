import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/bank-pred/bank.csv')
df
df.info()
df.pdays.describe()
df.pdays.median()
df1 = df[df['pdays']>0]
df1
df1.pdays.describe()
df1.pdays.median()

df.groupby('education').median()['balance']
sns.barplot(x=df['balance'],y=df['education'],estimator=np.median)
df1.pdays.value_counts()
print(sns.boxplot(y=df1['pdays']))
print(sns.boxplot(y=df['pdays']))
df.pdays.value_counts()
df=pd.get_dummies(df,columns=['response'],drop_first=True)
df
sns.scatterplot(x='poutcome',y='pdays',data=df)

df.head(2)
sns.pairplot(df[['age','salary','balance','previous','duration','campaign','response_yes']])
sns.distplot(df['age'],bins=50)

lst = [df]
for column in lst:
    column.loc[column["age"] < 30,  'age group'] = 30
    column.loc[(column["age"] >= 30) & (column["age"] <= 44), 'age group'] = 40
    column.loc[(column["age"] >= 45) & (column["age"] <= 59), 'age group'] = 50
    column.loc[column["age"] >= 60, 'age group'] = 60
agewise_response = pd.crosstab(df['response_yes'],df['age group']).apply(lambda x: x/x.sum() * 100)
agewise_response = agewise_response.transpose()
sns.countplot(x='age group', data=df, hue='response_yes')
print('Success rate and total people with different age groups contacted:')
print('People with age < 30 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 30]), df[df['age group'] == 30].response_yes.value_counts()[1]/len(df[df['age group'] == 30])))
print('People between 30 & 45 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 40]), df[df['age group'] == 40].response_yes.value_counts()[1]/len(df[df['age group'] == 40])))
print('People between 40 & 60 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 50]), df[df['age group'] == 50].response_yes.value_counts()[1]/len(df[df['age group'] == 50])))
print('People with 60+ age contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 60]), df[df['age group'] == 60].response_yes.value_counts()[1]/len(df[df['age group'] == 60])))
sns.set(rc={'figure.figsize':(20,5)})
sns.countplot(x=df['job'], data=df, hue=df['response_yes'])
plt.title('Response recieved with respect to JOB')

from prettytable import PrettyTable
counts = PrettyTable(['Job', 'Total Clients', 'Success rate'])
counts.add_row(['Blue-collar', len(df[df['job'] == 'blue-collar']), df[df['job'] == 'blue-collar'].response_yes.value_counts()[1]/len(df[df['job'] == 'blue-collar'])])
counts.add_row(['Management', len(df[df['job'] == 'management']), df[df['job'] == 'management'].response_yes.value_counts()[1]/len(df[df['job'] == 'management'])])
counts.add_row(['Technician', len(df[df['job'] == 'technician']), df[df['job'] == 'technician'].response_yes.value_counts()[1]/len(df[df['job'] == 'technician'])])
counts.add_row(['Admin', len(df[df['job'] == 'admin.']), df[df['job'] == 'admin.'].response_yes.value_counts()[1]/len(df[df['job'] == 'admin.'])])
counts.add_row(['Services', len(df[df['job'] == 'services']), df[df['job'] == 'services'].response_yes.value_counts()[1]/len(df[df['job'] == 'services'])])
counts.add_row(['Retired', len(df[df['job'] == 'retired']), df[df['job'] == 'retired'].response_yes.value_counts()[1]/len(df[df['job'] == 'retired'])])
counts.add_row(['Self-employed', len(df[df['job'] == 'self-employed']), df[df['job'] == 'self-employed'].response_yes.value_counts()[1]/len(df[df['job'] == 'self-employed'])])
counts.add_row(['Entrepreneur', len(df[df['job'] == 'entrepreneur']), df[df['job'] == 'entrepreneur'].response_yes.value_counts()[1]/len(df[df['job'] == 'entrepreneur'])])
counts.add_row(['Unemployed', len(df[df['job'] == 'unemployed']), df[df['job'] == 'unemployed'].response_yes.value_counts()[1]/len(df[df['job'] == 'unemployed'])])
counts.add_row(['Housemaid', len(df[df['job'] == 'housemaid']), df[df['job'] == 'housemaid'].response_yes.value_counts()[1]/len(df[df['job'] == 'housemaid'])])
counts.add_row(['Student', len(df[df['job'] == 'student']), df[df['job'] == 'student'].response_yes.value_counts()[1]/len(df[df['job'] == 'student'])])
counts.add_row(['Unknown', len(df[df['job'] == 'unknown']), df[df['job'] == 'unknown'].response_yes.value_counts()[1]/len(df[df['job'] == 'unknown'])])
print(counts)
sns.countplot(x=df['poutcome'], data=df, hue=df['response_yes'])
plt.title('Count Plot of poutcome for target variable')
sns.countplot(x=df['salary'], data=df, hue=df['response_yes'])
plt.title('Salary wise and response')
sns.countplot(x=df['education'], data=df, hue=df['response_yes'])
plt.title('Respinse received based on education')

df.education.value_counts()
sns.countplot(x=df['default'], data=df, hue=df['response_yes'])
plt.title('Response received against defaulters')
df.default.value_counts()
df[df['default']=='yes'].response_yes.count()
sns.countplot(x=df['loan'], data=df, hue=df['response_yes'])
plt.title('Count plot of loan for target variable y')
sns.countplot(x=df['contact'], data=df, hue=df['response_yes'])
plt.title('Modes of Communication:')
df.contact.value_counts()
sns.countplot(x=df['month'], data=df, hue=df['response_yes'])
plt.title('MOnth wise communication and response')
df.head(1)
categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical = [x for x in df.columns.to_list() if x not in categorical]
numerical.remove('response_yes')
numerical.remove('age group')

corr_data = df[numerical + ['response_yes']]
corr = corr_data.corr()
plt.close()
cor_plot = sns.heatmap(corr,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':10})
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.xticks(fontsize=10,rotation=-30)
plt.yticks(fontsize=10)
plt.title('Correlation Matrix')
plt.show()
pd.crosstab(df['pdays'],df['poutcome'])
pd.crosstab(df['pdays'],df['poutcome'],values=df['response_yes'],aggfunc='count',margins=True,normalize=True)
pd.crosstab(df['pdays'],df['previous'],values=df['response_yes'],aggfunc='count',margins=True,normalize=True)
df=pd.get_dummies(df,drop_first=True)
df
df.columns
X=df.drop('response_yes', axis=1)
Y=df['response_yes']

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
logm=LogisticRegression()
from sklearn.feature_selection import RFE
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 30% of the data will be used for testing
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=0)
import warnings
warnings.filterwarnings('ignore')
logm.fit(X_train,Y_train)
rfe = RFE(logm, 10)
rfe = rfe.fit(X_train, Y_train)
rfe_ = X_train.columns[rfe.support_]
rfe_
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)
X_new_train=X_train[rfe_]
checkVIF(X_new_train)
X_new=df[['housing_yes', 'contact_unknown','month_aug','month_jan',
       'month_jul', 'month_mar', 'month_oct', 'month_sep', 'poutcome_success']]
Y=df['response_yes']
X_new_train, X_new_test, Y_train, Y_test= train_test_split(X_new, Y, test_size=0.3, random_state=0)
z=logm.fit(X_new_train,Y_train)
z
auc=[X_train,X_new_train]
models = []
models.append(('LogisticRegression', LogisticRegression()))
for i in auc:
        kfold = KFold(n_splits=10, random_state=0)    
        # train the model
        cv_results = cross_val_score(LogisticRegression(), i, Y_train, cv=kfold, scoring='accuracy')    
        msg = "%s: %f (%f)" % (LogisticRegression, cv_results.mean(), cv_results.std())
        print(msg)
Y_pred=z.predict(X_new_test)
Y_pred
Y_pred.shape
# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestClassifier(n_estimators=30,max_depth=30)
rfc.fit(X_train, Y_train)

y_pred=rfc.predict(X_test)
y_pred
print(confusion_matrix(Y_test,y_pred))

print(classification_report(Y_test, y_pred))
p=[X_train,X_new_train]
for i in p:
    kf = KFold(n_splits=10)    
    cross_v = cross_val_score(RandomForestClassifier(), i, Y_train, cv=kfold, scoring='accuracy')  
    print('Cross validation score:',cross_v.mean())
model_new = RandomForestClassifier(n_estimators=45,max_depth=10)
model_new.fit(X_new_train, Y_train)
y1_pred=model_new.predict(X_new_test)
y1_pred
print('For all features')
print(accuracy_score(Y_test, y1_pred))
print('For selected features')
print(accuracy_score(Y_test, y1_pred))
print(classification_report(Y_test, y1_pred))
print(confusion_matrix(Y_test,y1_pred))
