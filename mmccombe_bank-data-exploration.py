import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
import scipy as sp
from scipy import stats
### Upload the files from your computer:
#### bank-train and bank-test.csv (and samp-submission.csv)
from google.colab import files
uploaded = files.upload()
train = pd.read_csv('bank-train.csv')
test = pd.read_csv('bank-test.csv')

train.head() # lots of categorical variables
train.describe() # pdays, previous, y is very skewed
sns.catplot(y='id', x='y', data=train, kind='bar')
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10, 2), sharey=False, dpi=100)
sns.distplot(train['pdays'] , color="dodgerblue", ax=axes[0], axlabel='Pdays')
sns.distplot(train['previous'] , color="deeppink", ax=axes[1], axlabel='Previous')
print(train.isnull().apply(sum), '\n') # no null values
print(train.groupby('y').count()['id'], '\n') # 29245=0, 3705=1
print('There are',len(test),'testing observations') # 8238 testing observations
'''categorical: 
job: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown'
marital: 'divorced', 'married', 'single', 'unknown'
education: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown'
default: 'no', 'unknown', 'yes'
housing: 'no', 'unknown', 'yes'
loan: 'no', 'unknown', 'yes'
contact: 'cellular', 'telephone'
month: 'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct',
       'sep'
day_of_week: 'fri', 'mon', 'thu', 'tue', 'wed'
poutcome: 'failure', 'nonexistent', 'success'
''' 

def yes_no_uk(x):
  '''used to transform default, housing, and loan'''
  if x=='yes':
    return(1)
  elif x=='no':
    return(0)
  elif x=='unknown':
    return(2)
  
def poutcome(x):
  '''used to transform poutcome'''
  if x=='success':
    return(1)
  elif x=='failure':
    return(0)
  elif x=='nonexistent':
    return(2)

def day_of_week(x):
  '''used to transform day_of_week'''
  if x=='mon':
    return(1)
  elif x=='tue':
    return(2)
  elif x=='wed':
    return(3)
  elif x=='thu':
    return(4)
  elif x=='fri':
    return(5)
  
def month(x):
  '''used to transform month'''
  if x=='jan':
    return(1)
  elif x=='feb':
    return(2)
  elif x=='mar':
    return(3)
  elif x=='apr':
    return(4)
  elif x=='may':
    return(5)
  elif x=='jun':
    return(6)
  elif x=='jul':
    return(7)
  elif x=='aug':
    return(8)
  elif x=='sep':
    return(9)
  elif x=='oct':
    return(10)
  elif x=='nov':
    return(11)
  elif x=='dec':
    return(12)

# transforming ordinal variables
default_labels = train['default'].apply(yes_no_uk)
housing_labels = train['housing'].apply(yes_no_uk)
loan_labels = train['loan'].apply(yes_no_uk)
month_labels = train['month'].apply(month)
day_labels = train['day_of_week'].apply(day_of_week)
poutcome_labels = train['poutcome'].apply(poutcome)

# transforming test data
default_labels2 = test['default'].apply(yes_no_uk)
housing_labels2 = test['housing'].apply(yes_no_uk)
loan_labels2 = test['loan'].apply(yes_no_uk)
month_labels2 = test['month'].apply(month)
day_labels2 = test['day_of_week'].apply(day_of_week)
poutcome_labels2 = test['poutcome'].apply(poutcome)
# transforming categorical variables
marital_labels = pd.get_dummies(train['marital'])
job_labels = pd.get_dummies(train['job'])
education_labels = pd.get_dummies(train['education'])
contact_labels = pd.get_dummies(train['contact'])

# making sure the unknowns have a specific label
marital_labels.columns = ['divorced', 'married', 'single', 'unknown.marital']
job_labels.columns = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown.job']
education_labels.columns = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown.education']



# transforming test data
marital_labels2 = pd.get_dummies(test['marital'])
job_labels2 = pd.get_dummies(test['job'])
education_labels2 = pd.get_dummies(test['education'])
contact_labels2 = pd.get_dummies(test['contact'])

# making sure the unknowns have a specific label
marital_labels2.columns = ['divorced', 'married', 'single', 'unknown.marital']
job_labels2.columns = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown.job']
education_labels2.columns = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown.education']
train_y = train['y']
train2 = train[['id', 'age', 'duration', 'campaign','pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']]

train2['default'] = default_labels
train2['housing'] = housing_labels
train2['loan'] = loan_labels
train2['month'] = month_labels
train2['day'] = day_labels
train2['poutcome'] = poutcome_labels

train2 = pd.concat([train2, marital_labels], axis=1)
train2 = pd.concat([train2, job_labels], axis=1)
train2 = pd.concat([train2, education_labels], axis=1)
train2 = pd.concat([train2, contact_labels], axis=1)

train2['y'] = train_y
train2.head()

test2 = test[['id', 'age', 'duration', 'campaign','pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']]

test2['default'] = default_labels2
test2['housing'] = housing_labels2
test2['loan'] = loan_labels2
test2['month'] = month_labels2
test2['day'] = day_labels2
test2['poutcome'] = poutcome_labels2

test2 = pd.concat([test2, marital_labels2], axis=1)
test2 = pd.concat([test2, job_labels2], axis=1)
test2 = pd.concat([test2, education_labels2], axis=1)
test2 = pd.concat([test2, contact_labels2], axis=1)


test2.head()
train2.describe()
#test2.describe()
corr = pd.DataFrame()
for a in list('y'):
    for b in list(train2.columns.values):
        corr.loc[b, a] = train2.corr().loc[a, b]
sns.heatmap(corr)
#print(corr['y'].sort_values())
train2.columns
#corr = pd.DataFrame()
#for a in list('y'):
#    for b in list(train2.columns.values):
#        corr.loc[b, a] = train2.corr().loc[a, b]
        
#sns.heatmap(corr)
#print(corr['y'].sort_values())

# variables with abs(corr)<0.01
'''
basic.4y              -0.009658
high.school           -0.009604
housemaid             -0.008696
self-employed         -0.008180
unknown.job           -0.003743
technician            -0.001249
management            -0.000280
loan                   0.000409
professional.course    0.000415
unknown.marital        0.002550
illiterate             0.007441
day                    0.008814
'''
train3 = train2[['age', 'duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'housing', 'month', 'poutcome',
       'divorced', 'married', 'single', 'admin.',
       'blue-collar', 'entrepreneur', 'retired',
       'services', 'student', 'unemployed',
       'basic.6y', 'basic.9y', 'university.degree',
       'unknown.education', 'cellular', 'telephone']]
test3 = test2[['age', 'duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'housing', 'month', 'poutcome',
       'divorced', 'married', 'single', 'admin.',
       'blue-collar', 'entrepreneur', 'retired',
       'services', 'student', 'unemployed',
       'basic.6y', 'basic.9y', 'university.degree',
       'unknown.education', 'cellular', 'telephone']]

# variables with abs(corr)<0.05
"""
basic.9y              -0.043711
married               -0.042574
services              -0.031471
basic.6y              -0.024711
entrepreneur          -0.016653
divorced              -0.010230
basic.4y              -0.009658
high.school           -0.009604
housemaid             -0.008696
self-employed         -0.008180
unknown.job           -0.003743
technician            -0.001249
management            -0.000280
loan                   0.000409
professional.course    0.000415
unknown.marital        0.002550
illiterate             0.007441
day                    0.008814
housing                0.011729
unemployed             0.014542
unknown.education      0.016053
age                    0.027631
admin.                 0.030412
month                  0.036602
"""
train4 = train2[['duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'poutcome', 'single', 'blue-collar','retired',
       'student', 'university.degree', 'cellular', 'telephone']]

test4 = test2[['duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'poutcome', 'single', 'blue-collar','retired',
       'student', 'university.degree', 'cellular', 'telephone']]
train2_lite = train2.iloc[:-2000, :-1]
trainy_lite = train2.iloc[:-2000, -1]
train2_test = train2.iloc[-2000:, :-1]
trainy_test = train2.iloc[-2000:, -1]

train3_lite = train3.iloc[:-2000, :-1]
train3_test = train3.iloc[-2000:, :-1]

train4_lite = train4.iloc[:-2000, :-1]
train4_test = train4.iloc[-2000:, :-1]
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(solver='liblinear',fit_intercept=True)

logis_test = logis.fit(train2_lite, trainy_lite)
preds = logis_test.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, preds))
print(sklearn.metrics.accuracy_score(trainy_test, preds))


log_test2 = logis.fit(train3_lite, trainy_lite)
preds2 = log_test2.predict(train3_test)
print(sklearn.metrics.f1_score(trainy_test, preds2))
print(sklearn.metrics.accuracy_score(trainy_test, preds2))


log_test3 = logis.fit(train4_lite, trainy_lite)
preds3 = log_test3.predict(train4_test)
print(sklearn.metrics.f1_score(trainy_test, preds3))
print(sklearn.metrics.accuracy_score(trainy_test, preds3))


from sklearn.feature_selection import f_regression
(F_vals, p_vals) = f_regression(train2_lite, trainy_lite)

cols = list(train2_lite.columns[p_vals<0.01])
trainF = train2[cols]

trainF_lite = trainF.iloc[:-2000, :-1]
trainF_test = trainF.iloc[-2000:, :-1]
log_testF = logis.fit(trainF_lite, trainy_lite)
predsF = log_testF.predict(trainF_test)
print(sklearn.metrics.f1_score(trainy_test, predsF))
print(sklearn.metrics.accuracy_score(trainy_test, predsF))
# no better than original
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


lr = LinearRegression()
linreg = lr.fit(train2_lite, trainy_lite)
predsLR = linreg.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsLR))
print(sklearn.metrics.accuracy_score(trainy_test, predsLR))

rr = Ridge(alpha=0.000001, normalize=True)
ridge = rr.fit(train2_lite, trainy_lite)
predsR = ridge.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsR))
print(sklearn.metrics.accuracy_score(trainy_test, predsR))

lasso = Lasso(alpha=0.00000000001, normalize=True)
lass = lasso.fit(train2_lite, trainy_lite)
predsLass = lass.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsLass))
print(sklearn.metrics.accuracy_score(trainy_test, predsLass))
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree = DecisionTreeClassifier()
treeD = tree.fit(train2_lite, trainy_lite)
predsTD = treeD.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsTD))
print(sklearn.metrics.accuracy_score(trainy_test, predsTD))


print(pd.DataFrame({'Gain': treeD.feature_importances_}, index = train2_lite.columns).sort_values('Gain', ascending = False))
forest = RandomForestClassifier(criterion = 'entropy')
forestR = forest.fit(train2_lite, trainy_lite)
predsF = forestR.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsF))
print(sklearn.metrics.accuracy_score(trainy_test, predsF))


print(pd.DataFrame({'Importance': forestR.feature_importances_}, index = train2_lite.columns).sort_values('Importance', ascending = False))

'''
duration               0.302235*
id                     0.153253*
age                    0.070679*
euribor3m              0.053408*
nr.employed            0.037945*
pdays                  0.034321*
campaign               0.032640*
day                    0.030208*
emp.var.rate           0.024689
month                  0.021207

duration             0.327945*
nr.employed          0.154874*
id                   0.116405*
age                  0.074104*
euribor3m            0.037914*
campaign             0.032394*
cons.conf.idx        0.023777
day                  0.023379*
pdays                0.022481*
housing              0.013133
'''
trainT = train2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign', 'day', 'cons.conf.idx', 'housing',
                'emp.var.rate']]
trainT_lite = trainT.iloc[:-2000,:]
trainT_test = trainT.iloc[-2000:, :]

testT = test2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign', 'day', 'cons.conf.idx', 'housing',
                'emp.var.rate']]

treeD2 = tree.fit(trainT_lite, trainy_lite)
predsTD2 = treeD2.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsTD2))
print(sklearn.metrics.accuracy_score(trainy_test, predsTD2))


forest2 = RandomForestClassifier(criterion = 'gini')
forestR2 = forest2.fit(trainT_lite, trainy_lite)
predsF2 = forestR2.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsF2))
print(sklearn.metrics.accuracy_score(trainy_test, predsF2))
linregT = lr.fit(trainT_lite, trainy_lite)
predsLRT = linregT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLRT))


rr = Ridge(alpha=0.000001, normalize=True)
ridgeT = rr.fit(trainT_lite, trainy_lite)
predsRT = ridgeT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsRT))


lassoT = Lasso(alpha=0.000001, normalize=True)
lassT = lassoT.fit(trainT_lite, trainy_lite)
predsLassT = lassT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLassT))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(trainT_lite, trainy_lite)
predsG = model.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsG))
print(sklearn.metrics.accuracy_score(trainy_test, predsG))


linregT = lr.fit(trainT, train.iloc[:, -1]) # best so far
prediction1 = linregT.predict(testT)>0.3

def TF(x):
  if x==True:
    return(1)
  elif x==False:
    return(0)

submission = pd.concat([test.id, pd.Series(prediction1)], axis = 1)
submission.columns = ['id', 'Predicted']
submission['Predicted'] = submission['Predicted'].apply(TF)
submission.to_csv('submission.csv', index=False)

submission
from google.colab import files
files.download('submission.csv')
logis_test2 = logis.fit(train2_lite, trainy_lite)
predsL = logis_test2.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsL))
print(sklearn.metrics.accuracy_score(trainy_test, predsL))
from sklearn.svm import SVC
from sklearn.linear_model import  LogisticRegression
classifier = SVC(kernel="linear")
svm = classifier.fit(trainT_lite, trainy_lite)
predsSVM = svm.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsSVM))
print(sklearn.metrics.accuracy_score(trainy_test, predsSVM))
predictions3 = svm.predict(testT)
submission = pd.concat([test.id, pd.Series(predictions3)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)
submission
from google.colab import files
files.download('submission.csv')
classifier2 = SVC()
#svm2 = classifier2.fit(trainT_lite, trainy_lite)
predsSVM2 = svm2.predict(trainT_test)
print(sklearn.metrics.accuracy_score(trainy_test, predsSVM2))
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()
knn.fit(trainT_lite, trainy_lite)
predKNN = knn.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predKNN))
print(sklearn.metrics.accuracy_score(trainy_test, predKNN))

knn.fit(trainT_lite, trainy_lite)
predKNN2 = knn.predict(testT)
submission = pd.concat([test.id, pd.Series(predKNN2)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)
submission
from google.colab import files
files.download('submission.csv')
classifier2 = SVC(kernel="rbf")
svm2 = classifier2.fit(trainT_lite, trainy_lite)
predsSVM2 = svm2.predict(trainT_test)
print(sklearn.metrics.fbeta_score(trainy_test, predsSVM2))
print(sklearn.metrics.accuracy_score(trainy_test, predsSVM2))
trainT2 = train2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign']]
trainT2_lite = trainT.iloc[:-2000,:]
trainT2_test = trainT.iloc[-2000:, :]

testT2 = test2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign']]

linregT = lr.fit(trainT2_lite, trainy_lite)
predsLRT = linregT.predict(trainT2_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLRT))

rr = Ridge(alpha=0.000001, normalize=True)
ridgeT = rr.fit(trainT2_lite, trainy_lite)
predsRT = ridgeT.predict(trainT2_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsRT))

lassoT = Lasso(alpha=0.000001, normalize=True)
lassT = lassoT.fit(trainT2_lite, trainy_lite)
predsLassT = lassT.predict(trainT2_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLassT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLassT))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(trainT2_lite, trainy_lite)
predsG = model.predict(trainT2_test)
print(sklearn.metrics.f1_score(trainy_test, predsG))
print(sklearn.metrics.accuracy_score(trainy_test, predsG))

logis_test2 = logis.fit(trainT_lite, trainy_lite)
predsL = logis_test2.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsL))
print(sklearn.metrics.accuracy_score(trainy_test, predsL))

logis_test3 = logis.fit(trainT2_lite, trainy_lite)
predsL2 = logis_test3.predict(trainT2_test)
print(sklearn.metrics.f1_score(trainy_test, predsL2))
print(sklearn.metrics.accuracy_score(trainy_test, predsL2))
train1 = []
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list2='var'+'_'+var
    cat_list2 = pd.get_dummies(test[var], prefix=var)
    test1 = test.join(cat_list2)
    test=test1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars2=test.columns.values.tolist()
to_keep=[i for i in data_vars2 if i not in cat_vars]
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(train[var], prefix=var)
    train1=train.join(cat_list)
    train=train1
    cat_list2='var'+'_'+var
    cat_list2 = pd.get_dummies(test[var], prefix=var)
    test1 = test.join(cat_list2)
    test=test1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=train.columns.values.tolist()
data_vars2=test.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=train[to_keep]
test_final = test[to_keep]
data_final.columns.values
from sklearn.model_selection import train_test_split

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(train2_lite, trainy_lite)
print(rfe.support_)
print(rfe.ranking_)
train_small = train2_lite[train2_lite.columns[rfe.support_]]
test_small = train2_test[train2_lite.columns[rfe.support_]]
# trainy_lite, trainy_test

linregT = lr.fit(train_small, trainy_lite)
predsLRT = linregT.predict(test_small)>0.22
print(sklearn.metrics.f1_score(trainy_test, predsLRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLRT))

logis_test2 = logis.fit(train_small, trainy_lite)
predsL = logis_test2.predict(test_small)
print(sklearn.metrics.f1_score(trainy_test, predsL))
print(sklearn.metrics.accuracy_score(trainy_test, predsL))

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=LogisticRegression()
result=logit_model.fit(X, y)
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=LogisticRegression()
result=logit_model.fit(X, y)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
test2 = test_final[cols]
print(X.columns)
print(test2.columns)
print(y)
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(test2)
submission = pd.concat([test.id, pd.Series(y_pred)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))