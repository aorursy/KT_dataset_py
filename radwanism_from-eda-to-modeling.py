# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fullpath = os.path.join(dirname,filename)

data = pd.read_csv(fullpath)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
#%%timeit

%matplotlib inline

import sklearn 

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



from sklearn.pipeline import Pipeline

from collections import Counter

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit,cross_val_score,KFold,cross_val_predict,RepeatedStratifiedKFold,StratifiedShuffleSplit



from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score,confusion_matrix,precision_recall_curve,plot_precision_recall_curve,average_precision_score,balanced_accuracy_score,classification_report,plot_confusion_matrix,roc_auc_score,plot_roc_curve,average_precision_score







import numpy as np 

import matplotlib.pyplot as plt 

import yellowbrick as yb # For the styles



from sklearn.base import clone 

from sklearn.model_selection import KFold

from sklearn.metrics.classification import _check_targets

from sklearn.utils.multiclass import unique_labels

from sklearn.utils.validation import check_consistent_length

from sklearn.externals.joblib import Parallel, delayed
data.head(3)
#data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'],errors='coerce')

#data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'],errors='coerce')

data.ScheduledDay =  data.ScheduledDay.apply(np.datetime64)

data.AppointmentDay =  data.AppointmentDay.apply(np.datetime64)



data['appointment_day'] = data['AppointmentDay'].dt.day_name()

data['scaduled_day'] = data['ScheduledDay'].dt.day_name()

data["waiting_days"] = abs(data.AppointmentDay - data.ScheduledDay).dt.days

data.waiting_days.head(20)
data=data.rename(columns={

                         "Handcap":"Handicap",

                         "No-show": "Show"

                         })
data.head(3)
df=data.copy()
days={'Saturday':0,'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Friday':6}

gender={'F':0 , 'M':1 }

data['scaduled_day'] = data.scaduled_day.map(days)

data['appointment_day'] = data.appointment_day.map(days)

data['Gender']=data.Gender.map(gender)
y = data['Show']

y.replace('No',1,inplace=True)

y.replace('Yes',0,inplace=True)
print('Age:',sorted(data.Age.unique()))

print('Gender:',data.Gender.unique())

print('scaduled_day:',data.scaduled_day.unique())

print('appointment_day:',data.appointment_day.unique())

print('waiting_days',sorted(data.waiting_days.unique()))

%matplotlib inline

sns.stripplot(data = data, y = 'waiting_days', jitter = True)
data = data[data.waiting_days < 350]
df.head(2)


def probStatus(dataset, group_by):

    # count no-show /showup with respct to age

    df = pd.crosstab(index=dataset[group_by], columns=dataset.Show).reset_index()

    df['probShowUp'] = df['No'] / (df['No'] + df['Yes'])

    return df[[group_by, 'probShowUp']]
dfprop.head(1)
dfprop = probStatus(df,'Age')

ageplot = sns.lmplot(data = dfprop, x='Age', y='probShowUp', fit_reg=True)

ageplot.set(xlim=(0, 100), title='Probability of showing up with respect to Age')
def calculateHour(timestamp):

    timestamp = str(timestamp)

    hour = int(timestamp[11:13])

    minute = int(timestamp[14:16])

    second = int(timestamp[17:])

    return round(hour + minute/60 + second/3600)



df['HourOfTheDay'] = df.ScheduledDay.apply(calculateHour)
data.head(1)
df.head(1)
#dfhour=probStatus(data, 'HourOfTheDay')



H = sns.lmplot(data = probStatus(df, 'HourOfTheDay'), x = 'HourOfTheDay', 

           y = 'probShowUp', fit_reg = True)

H.set(title=('Probability of showing up with respect to HourOfTheDay'))





T = sns.lmplot(data = probStatus(df, 'waiting_days'), x = 'waiting_days', 

           y = 'probShowUp', fit_reg = True)



T.set(title=('Probability of showing up with respect to AwaitingTime'),

     ylim=(0, 1))

sns.countplot(x='Show', data=df)

plt.show()
sns.countplot(x='Gender', data=df, palette = 'plasma')

plt.show()

plt.figure(figsize=(20,50))

ct = pd.crosstab(data.Age, data['Show'])

ct.plot.bar(stacked=True)

plt.show()
ct = pd.crosstab(df.Gender, df['Show'])

ct.plot.bar(stacked=True)

plt.show()

x=data['Gender']

print(Counter(x))



Scholarship = sns.countplot(x = 'Gender', hue = 'Show', data = df)

Scholarship.set_title('Gender Attendence')

plt.xlabel('Gender')

plt.ylabel('No of Visits')

plt.show()
ct = pd.crosstab(df.Scholarship, df['Show'])

ct.plot.bar(stacked=True)

plt.show()

x=data['Scholarship']

print(Counter(x))
Scholarship = sns.countplot(x = 'Scholarship', hue = 'Show', data = data)

Scholarship.set_title('Scholarship status for patients')

plt.xlabel('Scholarship received Status')

plt.ylabel('No of Visits')

plt.show()
cat_list = ['Scholarship','Gender',  'Hipertension', 'Diabetes', 'Alcoholism']

for col in cat_list:

    col1 =pd.crosstab(df[col],df['Show'])

    x=df[col]

    print(Counter(x))

    #ct.plot.bar(stacked=True)

   # plt.show()

    col1.div(col1.sum(1).astype(float), axis=0).plot(kind="bar",  stacked=True, figsize=(4,4));

categories = pd.Series(['same day: 0', 'week: 1-7', 'month: 8-30', 'quarter: 31-90', 'semester: 91-180', 'a lot of time: >180'])

df['waiting_days_categories'] = pd.cut(df.waiting_days, bins = [-1, 0, 7, 30, 90, 180, 500], labels=categories)

plt.figure(figsize=(1,6))

ct = pd.crosstab(df.waiting_days_categories, df['Show'])

ct.plot.bar(stacked=True)

plt.show()

x=df['waiting_days_categories']

print(Counter(x))
plt.figure(figsize=(1,6))

ct = pd.crosstab(df.SMS_received, df['Show'])

ct.plot.bar(stacked=True)

plt.show()
plt.figure(figsize=(1,6))

ct = pd.crosstab(df.appointment_day, df['Show'])

ct.plot.bar(stacked=True)

plt.show()

from collections import Counter

x=df['appointment_day']

print(Counter(x))
plt.figure(figsize=(1,6))

ct = pd.crosstab(df.scaduled_day , df['Show'])

ct.plot.bar(stacked=True)

plt.show()

from collections import Counter

x=df['scaduled_day']

print(Counter(x))
data=data.drop(['AppointmentID','ScheduledDay','AppointmentDay','Neighbourhood'],axis=1)

data.head(3)


data.drop(data[(data.Age < 0) | (data.Age > 100)].index, inplace = True)
scaler=StandardScaler()

data['Age'] = scaler.fit_transform(data[['Age']])

data['PatientId'] = scaler.fit_transform(data[['PatientId']])
print("Columns with Missing Values::",data.columns[data.isnull().any()].tolist())

print("Number of rows with Missing Values::",len(pd.isnull(data).any(1).to_numpy().nonzero()[0].tolist()))

print("Sample Indices with missing data::",pd.isnull(data).any(1).to_numpy().nonzero()[0].tolist()[0:5] )
print(df['Show'].value_counts())

print('No show', round(df['Show'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('show', round(df['Show'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
osa=data.copy()

osa.head(2)
from sklearn.utils import resample



# Separate input features and target

y = osa.Show

X = osa.drop('Show', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)



# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)



# separate minority and majority classes

no_show = X[X.Show==0]

show = X[X.Show==1]



# upsample minority

noshow_upsampled = resample(show,

                          replace=True, # sample with replacement

                          n_samples=len(no_show), # match number in majority class

                          random_state=27) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([no_show, noshow_upsampled])



# check new class counts

print(upsampled.Show.value_counts())

print(upsampled.Show.shape)

   
upsampled.head(2)
y = upsampled.Show

X = upsampled.drop('Show', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

#clf=LogisticRegression(solver='liblinear')

#clf=SVC(kernel='linear')

#clf=DecisionTreeClassifier(max_depth=5,min_samples_leaf=12)

clf= RandomForestClassifier(n_estimators=1000, random_state=42)

#clf=LinearSVC()

#upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

clf.fit(X_train, y_train)

upsampled_pred = clf.predict(X_test)



# Checking accuracy

accuracy_score(y_test, upsampled_pred)

#    0.9807

    

# f1 score

f1_score(y_test, upsampled_pred)

 #   0.1437

    

recall_score(y_test, upsampled_pred)

#    0.8712

# Checking unique values

predictions = pd.DataFrame(upsampled_pred)

cl=classification_report(y_test,upsampled_pred)

print(cl)

cm=confusion_matrix(y_test, upsampled_pred)

print(cm)
CV = StratifiedShuffleSplit(n_splits=7, test_size=0.2, random_state=100)

score=['accuracy','f1']

scores=cross_validate(clf, X, y, cv=CV, scoring=score)

print(sorted(scores.keys()))

print(scores['test_accuracy'])

print(scores[ 'test_f1'])
#plot ROC curve

clf_disp = plot_roc_curve(clf, X_test, y_test)

plt.show()
y_pred = clf.predict(X_test)

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
#Plot precision and recall curve

disp = plot_precision_recall_curve(clf, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))

rfc = clf

pred=rfc.fit(X_train, y_train)

ax = plt.gca()

rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)

clf_disp.plot(ax=ax, alpha=0.8)

plt.show()
cm = confusion_matrix(y_test, y_pred)

print(cm)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['no-show', 'Show']); ax.yaxis.set_ticklabels(['no-show', 'Show']);
plot_confusion_matrix(clf, X_test, y_test,)  

plt.show()  
#make a function to plot result and compare predicted and actual 

def plot_target(y_true, y_pred, labels=None, ax=None, width=0.35, **kwargs):

    # Validate the input 

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    

    if y_type not in ("binary", "multiclass"):

        raise ValueError("%s is not supported" % y_type)

    



    # This is probably not necessary 

    check_consistent_length(y_true, y_pred)

    

    # Manage the labels passed in (yb might use classes for this arg)

    if labels is None:

        labels = unique_labels(y_true, y_pred)

    else:

        labels = np.asarray(labels)

        if np.all([l not in y_true for l in labels]):

            raise ValueError("At least one label specified must be in y_true")





    # Count the values of y_true and y_pred for each class

    indices = np.arange(0, labels.shape[0]) 



    # This expects labels to be numerically encoded, not strings 

    # YB needs to handle either case better, though _check_targets 

    # may deal with this, I'm not sure - need to review the code. 

    # Needless to say this is a HACK that needs to be addressed. 

    t_counts = np.array([(y_true==label).sum() for label in indices])

    p_counts = np.array([(y_pred==label).sum() for label in indices])

    

    # Begin the figure 

    if ax is None:

        _, ax = plt.subplots()



    b1 = ax.bar(indices, t_counts, width, color='b', label="actual")

    b2 = ax.bar(indices+width, p_counts, width, color='g', label="predicted")



    ax.set_xticks(indices + width/2)

    ax.set_xticklabels(labels)

    ax.set_xlabel("class")

    ax.set_ylabel("number of instances")

    ax.legend(loc='best', frameon=True)

    ax.grid(False, axis='x')



    return ax







#Plot result on test data 

plot_target(y_test, y_pred) 

plt.show()


X_list=list(X_test)

y_list=list(y_test)

print(pred)

#print(X_test)

#print(X_list)

#print(y_list)



listt1=list(y_test)

listt2=list(y_pred)

#print(listt1)

print(listt2)
print(listt1.count(1))

print(listt1.count(0))

print(listt2.count(1))

print(listt2.count(0))