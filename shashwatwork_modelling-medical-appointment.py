# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import datetime

from time import strftime



from sklearn.metrics import accuracy_score,precision_score,roc_curve,roc_auc_score,classification_report

from sklearn.model_selection import GridSearchCV,KFold,train_test_split,learning_curve

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler

import seaborn as se

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import keras

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings('ignore')



from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.linear_model import RidgeClassifier
data = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
print("The shape of the DataFrame {}".format(data.shape))
data.head()
data['PatientId'] = data['PatientId'].astype('int64')



data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay']).dt.date.astype('datetime64[ns]')

data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay']).dt.date.astype('datetime64[ns]')



data = data.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})
data.info()
percent_missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)

percent_missing.head()
data.head()
print(f"Total of Unique Patients is {data.PatientId.nunique()} and Appointments is {data.AppointmentID.nunique()}")
null_feat = pd.DataFrame(len(data['PatientId']) - data.isnull().sum(), columns = ['Count'])



trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'lightblue',

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  "Missing Values")

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
# Print Unique Values

print("Unique Values in `Gender` => {}".format(data.Gender.unique()))

print("Unique Values in `Scholarship` => {}".format(data.Scholarship.unique()))

print("Unique Values in `Hypertension` => {}".format(data.Hypertension.unique()))

print("Unique Values in `Diabetes` => {}".format(data.Diabetes.unique()))

print("Unique Values in `Alcoholism` => {}".format(data.Alcoholism.unique()))

print("Unique Values in `Handicap` => {}".format(data.Handicap.unique()))

print("Unique Values in `SMSReceived` => {}".format(data.SMSReceived.unique()))
data['Scholarship'] = data['Scholarship'].astype('object')

data['Hypertension'] = data['Hypertension'].astype('object')

data['Diabetes'] = data['Diabetes'].astype('object')

data['Alcoholism'] = data['Alcoholism'].astype('object')

data['Handicap'] = data['Handicap'].astype('object')

data['SMSReceived'] = data['SMSReceived'].astype('object')
data.info()
print("Patients with `Age` less than -1 - {}".format(data[data.Age == -1].shape[0]))
data = data[(data.Age >= 0 ) & (data.Age <= 100)]

print("Unique Values in `Age` => {}".format(np.sort(data.Age.unique())))
# Get Day of the Week for ScheduledDay and AppointmentDay

data['ScheduledDay_DOW'] = data['ScheduledDay'].dt.weekday_name

data['AppointmentDay_DOW'] = data['AppointmentDay'].dt.weekday_name
data['AppointmentDay'] = np.where((data['AppointmentDay'] - data['ScheduledDay']).dt.days < 0, data['ScheduledDay'], data['AppointmentDay'])



# Get the Waiting Time in Days of the Patients.

data['Waiting_Time_days'] = data['AppointmentDay'] - data['ScheduledDay']

data['Waiting_Time_days'] = data['Waiting_Time_days'].dt.days
data
M = data[(data['NoShow'] == 'Yes')]

B = data[(data['NoShow'] == 'No')]

trace = go.Bar(x = (len(M), len(B)), y = ['Yes','No'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['green','red'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Count of NoShow variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
print("NoShow and Show '%' of Patients")

show = data.groupby(['NoShow']).size()[0]/(data.groupby(['NoShow']).size()[0]+data.groupby(['NoShow']).size()[1])

print("Percent of Patients who `Showed Up`  {}%".format(show*100))

noshow = data.groupby(['NoShow']).size()[1]/(data.groupby(['NoShow']).size()[0]+data.groupby(['NoShow']).size()[1])

print("Percent of Patients who Did `Not Showed Up` {}%".format(noshow*100))
data.groupby(['NoShow']).size()[0]
g = sns.distplot(data['Age'])

g.set_title("Age Count Distribuition", fontsize=18)

g.set_xlabel("")

g.set_ylabel("Probability", fontsize=12)
x = data.groupby('PatientId')['AppointmentDay'].nunique()

print('Mean number of appointments per patient:\t%s' %np.mean(x))

print('Median number of appointments per patient:\t%s' %np.median(x))



plt.figure(1)

plt.hist(x, bins = x.nunique())

plt.title("Number of Appointments per patient")

plt.show
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax  = se.countplot(data['Age'],hue = data['NoShow'])

ax.set_title('Appointment by Age')

plt.show()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax  = se.countplot(data['Neighbourhood'],hue = data['NoShow'])

ax.set_title('Appointment by Neighbourhood')

plt.show()
data
plt.figure(figsize=(16,4))

ax = sns.countplot(x=data.Waiting_Time_days, order=data.Waiting_Time_days.value_counts(ascending=True).iloc[:55].index)

ax.set_title("Waiting Time in Days")

plt.show()
data_viz = data.copy()

bin_ranges = [-1, 2, 8, 16, 18, 25, 40, 50, 60, 75]

bin_names = ["Baby", "Children", "Teenager", 'Young', 'Young-Adult', 'Adult', 'Adult-II', 'Senior', 'Old']



data_viz['age_bin'] = pd.cut(np.array(data_viz['Age']),

                               bins=bin_ranges, labels=bin_names)

# now stack and reset

show_prob_age = pd.crosstab(data_viz['age_bin'], data_viz['NoShow'], normalize='index')
stacked = show_prob_age.unstack().reset_index().rename(columns={0:'value'})

plt.figure(figsize=(16,12))

ax1 = sns.countplot(x="age_bin", data=data_viz)

ax1.set_title("Age Bins Count", fontsize=22)

ax1.set_xlabel("Age Categories", fontsize=18)

ax1.set_ylabel("Count", fontsize=18)
plt.figure(figsize=(10,4))

ax  = se.countplot(data_viz['Hypertension'],hue = data_viz['NoShow'])

ax.set_title('Plot for  Hypertension')

plt.show()
plt.figure(figsize=(10,4))

plt.xticks(rotation=90)

ax  = se.countplot(data_viz['Alcoholism'],hue = data_viz['NoShow'])

ax.set_title('Plot for  Alcoholism')

plt.show()
plt.figure(figsize=(10,4))

plt.xticks(rotation=90)

ax  = se.countplot(data_viz['Diabetes'],hue = data_viz['NoShow'])

ax.set_title('Plot for  Diabetes')

plt.show()
week_key = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

plt.figure(figsize=(16,4))

ax = sns.countplot(x=data_viz.AppointmentDay_DOW, hue=data_viz.NoShow, order=week_key)

ax.set_title("Show/NoShow for Appointment Day of the Week")

plt.show()
def dayToNumber(day):

    if day == 'Monday': 

        return 0

    if day == 'Tuesday': 

        return 1

    if day == 'Wednesday': 

        return 2

    if day == 'Thursday': 

        return 3

    if day == 'Friday': 

        return 4

    if day == 'Saturday': 

        return 5

    if day == 'Sunday': 

        return 6



data.Gender = data.Gender.apply(lambda x: 1 if x == 'M' else 0)

data.ScheduledDay_DOW = data.ScheduledDay_DOW.apply(dayToNumber)

data.AppointmentDay_DOW = data.AppointmentDay_DOW.apply(dayToNumber)

data.NoShow = data.NoShow.apply(lambda x: 1 if x == 'Yes' else 0)
data['ScheduledDay_Y'] = data['ScheduledDay'].dt.year

data['ScheduledDay_M'] = data['ScheduledDay'].dt.month

data['ScheduledDay_D'] = data['ScheduledDay'].dt.day

data.drop(['ScheduledDay'], axis=1, inplace=True)



data['AppointmentDay_Y'] = data['AppointmentDay'].dt.year

data['AppointmentDay_M'] = data['AppointmentDay'].dt.month

data['AppointmentDay_D'] = data['AppointmentDay'].dt.day

data.drop(['AppointmentDay'], axis=1, inplace=True)
col_to_drop = ['PatientId', 'AppointmentID']

data = data.drop(col_to_drop,axis=1)
le = LabelEncoder()

data['Neighbourhood'] = le.fit_transform(data['Neighbourhood'])
data
data.columns
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix



X = data.drop('NoShow',1)

y = data.NoShow



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from imblearn.over_sampling import ADASYN

balancer = ADASYN(random_state=42)

x_resampled, y_resampled = balancer.fit_sample(X_train, y_train)



print('Normal Data: ', Counter(y_train))

print('Resampled: ', Counter(y_resampled))
scaler = MinMaxScaler(feature_range=(0, 1))

x_resampled = scaler.fit_transform(x_resampled)

X_test = scaler.transform(X_test)
def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# Show metrics 

def show_metrics():

    tp = cm[1,1]

    fn = cm[1,0]

    fp = cm[0,1]

    tn = cm[0,0]

    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))

    print('Precision =     {:.3f}'.format(tp/(tp+fp)))

    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))

    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/

                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))
def plot_precision_recall():

    plt.step(recall, precision, color = 'b', alpha = 0.2,

             where = 'post')

    plt.fill_between(recall, precision, step ='post', alpha = 0.2,

                 color = 'b')



    plt.plot(recall, precision, linewidth=2)

    plt.xlim([0.0,1])

    plt.ylim([0.0,1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('Precision Recall Curve')

    plt.show();

def plot_roc():

    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)

    plt.plot([0,1],[0,1], 'k--', linewidth = 2)

   # plt.xlim([0.0,0.001])

   # plt.ylim([0.0,1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show();
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,

                        n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):

    """

    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel('Training examples')

    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis = 1)

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.std(test_scores, axis = 1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha = 0.1, color = "g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",

             label = "Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color = "g",

             label = "Cross-validation score")

    plt.legend(loc = "best")

    return plt
def cross_val_metrics(model) :

    scores = ['accuracy', 'precision', 'recall']

    for sc in scores:

        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)

        print('[%s] : %0.5f (+/- %0.5f)'%(sc, scores.mean(), scores.std()))
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(random_state = 42)

param_grid = {

            'penalty' : ['l2','l1'],  

            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]

            }



CV_log_clf = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'accuracy', verbose = 1, n_jobs = -1)

CV_log_clf.fit(x_resampled, y_resampled)



best_parameters = CV_log_clf.best_params_

print('The best parameters for using this model is', best_parameters)
CV_log2_clf = LogisticRegression(C = best_parameters['C'], 

                                 penalty = best_parameters['penalty'], 

                                 random_state = 42)





CV_log2_clf.fit(x_resampled, y_resampled)



y_pred = CV_log2_clf.predict(X_test)

y_score = CV_log2_clf.decision_function(X_test)

# Confusion maxtrix & metrics

cm = confusion_matrix(y_test, y_pred)

class_names = [0,1]
show_metrics()
cross_val_metrics(CV_log2_clf)
# ROC curve

fpr, tpr, t = roc_curve(y_test, y_score)

plot_roc()
clfs = []

seed = 3



clfs.append(("LogReg", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBClassifier())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsClassifier())]))) 



clfs.append(("DecisionTreeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeClassifier())]))) 



clfs.append(("RandomForestClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestClassifier())]))) 



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, 

                                                                       n_estimators=600))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))





clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreeClassifier())])))



scoring = 'accuracy'

n_folds = 10

msgs = []

results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, x_resampled, y_resampled, 

                                 cv=kfold, scoring=scoring, n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  

                               cv_results.std())

    msgs.append(msg)

    print(msg)

    

# creating a model

model = RandomForestClassifier()



# feeding the training set into the model

model.fit(x_resampled, y_resampled)



# predicting the test set results

y_pred = model.predict(X_test)



# Calculating the accuracies

print("Training accuracy :", model.score(x_resampled, y_resampled))

print("Testing accuarcy :", model.score(X_test, y_test))



# classification report

cr = classification_report(y_test, y_pred)

print(cr)



# confusion matrix 

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

se.heatmap(cm, annot = True, cmap = 'winter')

plt.title('Confusion Matrix', fontsize = 20)

plt.show()
model = Sequential()

model.add(Dense(64,input_dim = 18,activation='relu'))

model.add(Dense(32,activation='relu',init = 'uniform'))

model.add(Dense(16,activation='relu',init = 'uniform'))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history=model.fit(x_resampled,y_resampled ,epochs=50,batch_size=128, validation_data=(X_test,y_test))
# evaluate the keras model

_, accuracy = model.evaluate(x_resampled, y_resampled)

print('Accuracy: %.2f' % (accuracy*100))
from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()