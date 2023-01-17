
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))




import matplotlib.pyplot as plt
import seaborn as sns

# Display options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option("display.max_colwidth", 120)
pd.set_option('mode.chained_assignment', None)
pd.set_option('io.hdf.default_format','table')

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=2)
#  first look
interviews = pd.read_csv('../input/Interview.csv')
interviews.head()
interviews.describe()
#remove empty columns
interviews = interviews.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis=1)
#check column names
interviews.columns
# let's do a quick look into the features 
for c in interviews.columns:
    print(c)
    print(interviews[c].unique())
#rename col
newnames = {
    'Have you obtained the necessary permission to start at the required time':'permission',
    'Hope there will be no unscheduled meetings':'meetings',
    'Can I Call you three hours before the interview and follow up on your attendance for the interview':'follow up call',
    'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much':'alternative number',
    'Have you taken a printout of your updated resume. Have you read the JD and understood the same':'printout',
    'Are you clear with the venue details and the landmark.':'venue details',
    'Has the call letter been shared':'shared letter',
    'Nature of Skillset':'skillset',
    'Position to be closed':'position'
}
interviews.rename(columns=newnames, inplace=True)

interviews.columns
interviews.fillna('NA', inplace=True)
#unify criteria , YES, NO, NA
# quick and dirty solution
def unify_ans(row):
    
    string = row.upper()
        
    if 'NA' in string:
        return 'NA'

    if 'NOT SURE' in string or 'CANT SAY' in string or 'YET' in string:
        return 'UNCERTAIN'
    
    if 'NO' in string or 'CHECK' in string:
        return 'NO'
    if 'YES' in string:
        return 'YES'
    
    print(row)
    return 'NA'
    
# test cleaning function
temp = interviews.copy()
temp['permission'] = temp['permission'].apply(unify_ans)
newnames.values()

for c in ['Observed Attendance','permission', 'meetings', 'follow up call', 'alternative number', 'printout', 'venue details', 'shared letter']:
    interviews[c] = interviews[c].apply(unify_ans)
# let's do a quick look into the features 
for c in interviews.columns:
    print(c)
    print(interviews[c].unique())
# more specific cleaning
#Interview Type
#['Scheduled Walkin' 'Scheduled ' 'Walkin' 'Scheduled Walk In'
# 'Sceduled walkin' 'Walkin ' 'NA']
for idx,row in enumerate(interviews['Interview Type']):
    string = row.upper()
    
    if 'ULED WALK' in string:
        interviews['Interview Type'].iloc[idx] = 'SCHEDULED WALKIN'
    elif 'WALK' in string:
        interviews['Interview Type'].iloc[idx] = 'WALKIN'
    else:
        interviews['Interview Type'].iloc[idx] = string
interviews['Interview Type'].value_counts()
for c in ['Location', 'Candidate Current Location', 'Candidate Job Location', 'Interview Venue' , 'Candidate Native location']:
    for idx,row in enumerate(interviews[c]):
        string = row.upper().strip('-').strip()
        interviews[c].iloc[idx] = string
        
for c in ['Location', 'Candidate Current Location', 'Candidate Job Location', 'Interview Venue' , 'Candidate Native location']:
    print(interviews[c].unique())
for idx,row in enumerate(interviews['Industry']):
    string = row.upper()
    
    if 'IT' in string:
        interviews[c].iloc[idx] = 'IT'
interviews['Date of Interview'].unique()
import datetime
a = datetime.date(2018,3, 4)
a.weekday()
# I belive the most important factor in the date is day of the week mounth and year. 
# So I going to extract only those
# there are some date set on the future... not sure if those are typos
# 
def fix_dates(row):
    
    if row == 'NA':
        return 'NA','NA','NA','NA'
    else:
        pass

        string  = row.replace(" ", "")
        string  = string.replace("–", "-")

        if '&' in string:
            string = string.split('&')[0]
            
        if '.' in string:
            d, m , y = string.split('.')
        elif '/' in string:
            d, m , y = string.split('/')
        elif '-' in string:
            d, m , y = string.split('-')
        else:
            d, y = string.split('Apr')
            m = 4

        d = int(d)
        y = int(y)
        if y < 2000:
            y = y + 2000
        if m =='Apr':
             m = 4
        else:
            m = int(m)

        a = datetime.date(y,m, d)
        w = a.weekday()

        return w,d,m,y

    
interviews['weekday'],interviews['day'],interviews['month'],interviews['year'] = zip(*interviews['Date of Interview'].apply(fix_dates))
interviews.head(5)
interviews.shape
mldata = interviews.copy()
mldata['Marital Status'].replace({'Single':0, 'Married':1, 'NA':np.nan}, inplace=True)
mldata['Gender'].replace({'Female':0, 'Male':1, 'NA':np.nan}, inplace=True)

mldata['Gender'].value_counts()
# remove missing attendace
mldata = mldata[mldata['Observed Attendance']!='NA']
mldata.replace('NA', np.nan, inplace=True)
#  missing values per candidates
mldata.isnull().sum(axis=1)
# missing values per features
mldata.isnull().sum(axis=0)
trimdataset = mldata.copy()
for c in ['permission', 'meetings', 'follow up call','alternative number','printout', 'venue details', 'shared letter']:
    del trimdataset[c]
# missinig 
# 'Observed Attendance','permission', 'meetings', 'follow up call', 'alternative number', 'printout', 'venue details', 'shared letter']
# category columns
cat = ['Client name', 
       'Industry', 
       'Location', 
       'position',
       'Interview Type',
       'Candidate Current Location', 
       'Candidate Job Location', 
       'Interview Venue', 
       'Candidate Native location',
      'weekday','month',]

pref = ['Clnt','Ind', 'Loc', 'Pos', 'Itype', 'Cc_loc', 'Cj_loc', 'Iv', 'Cn_loc','W','M' ]

#DUMMYFY
trimdataset = pd.get_dummies(trimdataset, columns=cat, prefix=pref)
trimdataset['Observed Attendance'].replace({'YES':1,'NO':0}, inplace=True)
sel_features = list()
for c in trimdataset:
    if c in ['Date of Interview', 'skillset', 'Name(Cand ID)',   'Expected Attendance', 'Observed Attendance', 'weekday','month','year' ]:
        pass
    else:
        sel_features.append(c)
len(sel_features)
# shuffle and split
trimdataset = trimdataset.sample(frac=1)
msk = np.random.rand(len(trimdataset)) < 0.8
train = trimdataset[msk]
test = trimdataset[~msk]
# quick function to eval models

def show_performance_model(model, X_train, y_train, X_test, y_test):
    # check classification scores of logistic regression
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(model.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(model.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    print(model.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")


    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
          "and a specificity of %.3f" % (1-fpr[idx]) + 
          ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
    return
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.model_selection import  cross_val_score

X_train = train[sel_features]
y_train = train['Observed Attendance']
X_test = test[sel_features]
y_test  = test['Observed Attendance']

LR = LogisticRegression(C=0.1, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001)
show_performance_model(LR, X_train, y_train, X_test, y_test )
Xtotal = trimdataset[sel_features]
ytotal = trimdataset['Observed Attendance']

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(Xtotal, ytotal)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(Xtotal.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (Acc)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
mldata = mldata.replace(np.nan,'NA')

# category columns
cat = ['Client name', 
       'Industry', 
       'Location', 
       'position',
       'Interview Type',
       'Candidate Current Location', 
       'Candidate Job Location', 
       'Interview Venue', 
       'Candidate Native location',
       'weekday',
       'month',
       'permission',
       'meetings',
       'follow up call',
       'alternative number',
       'printout',
       'venue details',
       'shared letter']

pref = ['Clnt','Ind', 'Loc', 'Pos', 'Itype', 'Cc_loc', 'Cj_loc', 'Iv', 'Cn_loc','W','M', 'Per', 'Mt', 'Fc',
'Al', 'Pr', 'V', 'Shl' ]

#DUMMYFY
alldataset = pd.get_dummies(mldata, columns=cat, prefix=pref)
alldataset['Observed Attendance'].replace({'YES':1,'NO':0}, inplace=True)
sel_features = list()
for c in alldataset:
    if c in ['Date of Interview', 'skillset', 'Name(Cand ID)',   'Expected Attendance', 'Observed Attendance', 'weekday','month','year' ]:
        pass
    else:
        sel_features.append(c)

        
 # use same mask  than before
train = alldataset[msk]
test = alldataset[~msk]

X_train = train[sel_features]
y_train = train['Observed Attendance']
X_test = test[sel_features]
y_test  = test['Observed Attendance']
#y = final_train['Survived']

X_train.head(10)
X_train.isnull().values.any()
sel_features

LR = LogisticRegression(C=0.1, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001)
show_performance_model(LR, X_train, y_train, X_test, y_test )
Xtotal = alldataset[sel_features]
ytotal = alldataset['Observed Attendance']

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(Xtotal, ytotal)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(Xtotal.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (Acc)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 1000)
show_performance_model(RF, X_train, y_train, X_test, y_test )
featlist = list(zip(sel_features, RF.feature_importances_))
featlist.sort(key=lambda x: x[1], reverse=True)
featlist
# last try
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
show_performance_model(gnb, X_train, y_train, X_test, y_test )
df = mldata[mldata['permission']!='NA']
df = pd.get_dummies(df, columns=cat, prefix=pref)

df.shape

 # use same mask  than before
sel_features = list()
for c in df:
    if c in ['Date of Interview', 'skillset', 'Name(Cand ID)',   'Expected Attendance', 'Observed Attendance', 'weekday','month','year' ]:
        pass
    else:
        sel_features.append(c)

# I need a new mask
m = np.random.rand(len(df)) < 0.8
df['Observed Attendance'].replace({'YES':1,'NO':0}, inplace=True) 
train = df[m]
test = df[~m]

X_train = train[sel_features]
y_train = train['Observed Attendance']
X_test = test[sel_features]
y_test  = test['Observed Attendance']
#y = final_train['Survived']
LR = LogisticRegression(C=0.1, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001)
show_performance_model(RF, X_train, y_train, X_test, y_test )
