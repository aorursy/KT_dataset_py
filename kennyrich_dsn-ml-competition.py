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
import matplotlib.pyplot as plt

import seaborn as sns
df_train=pd.read_csv("/kaggle/input/intercampusai2019/train.csv")

df_test=pd.read_csv("/kaggle/input/intercampusai2019/test.csv")

df_train.head(5)
df_train.describe()
df_train.shape
df_train.columns
df_train.describe(include=[np.object, pd.Categorical]).T
import missingno as msno
missingData = df_train.columns[df_train.isnull().any()].tolist()

msno.matrix(df_train[missingData])
msno.bar(df_train[missingData], color="blue", log=True, figsize=(30,18))
msno.heatmap(df_train[missingData], figsize=(20,20))
cat_cols=['Division', 'Qualification', 'Gender', 'Channel_of_Recruitment', 'Previous_Award', 'State_Of_Origin',

         'Foreign_schooled', 'Marital_Status',  'Past_Disciplinary_Action', 'Previous_IntraDepartmental_Movement']



def count_unique(df, cols):

    for col in cols:

        print('\n' + 'For column ' + col)

        print(df[col].value_counts())

        

count_unique(df_train, cat_cols)
def plot_bars(df, cols):

    for col in cols:

        fig = plt.figure(figsize=(6,6)) # define plot area

        ax = fig.gca() # define axis

        counts = df[col].value_counts() # find the counts for each unique categor

        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the

        ax.set_title(col) # Give the plot a main title

        ax.set_xlabel('Count by' + col) # Set text for the x axis

        ax.set_ylabel('Count')# Set text for y axis

        plt.show()

        

plot_bars(df_train, cat_cols)
dataset=[df_train, df_test]

for data in dataset:

    data['Age']= 2019-data['Year_of_birth']

    

df_train.head(5)
dataset=[df_train, df_test]

for data in dataset:

    data['No_of_years_at_work']= 2019-data['Year_of_recruitment']

    

df_train.head(5)
dataset = [df_train, df_test]

zones = {'LAGOS':"SW","FCT":"NC", "OGUN":"SW","RIVERS":"SS", "ANAMBRA":"SE","KANO":"NW", "DELTA":"SS", "OYO":"SW", 

         "KADUNA":"NW","IMO":"SE", "EDO":"SS", "ENUGU":"SE", "ABIA":"SE", "OSUN":"SW","ONDO":"SW", "NIGER":"NC", 

         "KWARA":"NC", "PLATEAU":"NC", "AKWA IBOM":"SS", "NASSARAWA":"NC", "KATSINA":"NW", "ADAMAWA":"NE","BENUE":"NC",

         "BAUCHI":"NE", "KOGI":"NC", "SOKOTO":"NW", "CROSS RIVER":"SS", "EKITI":"SW", "BORNO":"NE", "TARABA":"NE",

         "KEBBI":"NW", "BAYELSA":"SS", "EBONYI":"SE", "GOMBE":"NE", "ZAMFARA":"NW","JIGAWA":"NW", "YOBE":"NE"}

for data in dataset:

    data['Zones']=data['State_Of_Origin'].replace(zones)

    

df_train.head(5)
df_train['Qualification'] = df_train['Qualification'].fillna(method="bfill")

df_test['Qualification'] = df_test['Qualification'].fillna(method="bfill")
print(df_train['Last_performance_score'].mean())

print(df_train['Last_performance_score'].max())

print(df_train['Last_performance_score'].median())

print(df_train['Last_performance_score'].min())
print(df_train.Age.mean())

print(df_train.Age.min())

print(df_train.Age.max())

print(df_train.Age.median())
print(df_train.Trainings_Attended.mean())

print(df_train.Trainings_Attended.min())

print(df_train.Trainings_Attended.max())

print(df_train.Trainings_Attended.median())
"""dataset = [df_train, df_test]

for data in dataset:

    data["Last_performance_score"] = pd.cut(data["Last_performance_score"], 3, labels=["low", "medium", "high"])

    data["Training_score_average"] = pd.cut(data["Training_score_average"], 3, labels=["low", "medium", "high"])  

    data["Trainings_Attended"] = pd.cut(data["Trainings_Attended"], 3, labels=["low", "medium", "high"])"""
df_train['Zones'].value_counts()
df_train.head(5)
df_train['No_of_previous_employers'].value_counts()
del df_train['Year_of_birth']

del df_test['Year_of_birth']

del df_train['Year_of_recruitment']

del df_test['Year_of_recruitment']

del df_test['State_Of_Origin']

del df_train['State_Of_Origin']
cat_cols=['Gender', 'Previous_Award','Foreign_schooled', 'Marital_Status',  'Qualification', 'Division','Channel_of_Recruitment', 'Zones',

          'Past_Disciplinary_Action', 'Previous_IntraDepartmental_Movement', 'No_of_previous_employers']



train=pd.get_dummies(df_train, prefix=cat_cols,columns=cat_cols)

test=pd.get_dummies(df_test, prefix=cat_cols, columns=cat_cols)
print("This is the shape of the training set ",train.shape)

print("This is the shape of the test set ", test.shape)
del train['EmployeeNo']

del test['EmployeeNo']
#Heat map

corrmat= train.corr()

f, ax =plt.subplots(figsize=(40,40))

sns.heatmap(corrmat, annot= True,square=True)
del train['Promoted_or_Not']

print("This is the shape of the training set ",train.shape)

print("This is the shape of the test set ", test.shape)
df_train['Promoted_or_Not'].value_counts()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

import sklearn.metrics as sklm

from math import sqrt
feature=train.columns

target=['Promoted_or_Not']

X=train[feature].values

y=df_train[target].values 

split_test_size=0.30

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=split_test_size, random_state=42)
print("{0:0.2f}% in training set".format((len(X_train)/len(train.index)) * 100))

print("{0:0.2f}% in test set".format((len(X_test)/len(train.index)) * 100))
from sklearn.preprocessing import RobustScaler

ss=RobustScaler()

X_train=ss.fit_transform(X_train)

X_test=ss.transform(X_test)

test1= ss.transform(test)
'''from fancyimpute import KNN

knn=KNN(k=3)

X_train = knn.fit_transform(X_train)

X_test=  knn.transfrom(X_test)

test1=knn.transform(test1)

'''
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_train, y_train = smote.fit_sample(X_train, y_train)
#Logistic Regression without parameter tuning

lr = LogisticRegression()

lr.fit(X_train, y_train)
def plot_auc(labels, probs):

    ## Compute the false positive rate, true positive rate

    ## and threshold along with the AUC

    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])

    auc = sklm.auc(fpr, tpr)

    

    ## Plot the result

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, color = 'blue', label = 'AUC = %0.2f' % auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
probabilities = lr.predict_proba(X_test)



def score_model(probs, threshold):

    return np.array([1 if x > threshold else 0 for x in probs[:,1]])



def print_metrics(labels, probs, threshold):

    scores = score_model(probs, threshold)

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))

    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:,1]))

    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))

    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

    



print_metrics(y_test, probabilities, 0.5)
probabilities = lr.predict_proba(X_train)

print_metrics(y_train, probabilities, 0.5)
plot_auc(y_train, probabilities)
gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)
plt.bar(range(len(gb.feature_importances_)), gb.feature_importances_)

plt.show()
probabilities = gb.predict_proba(X_train)

print_metrics(y_train, probabilities, 0.5)
probabilities = gb.predict_proba(X_test)

print_metrics(y_test, probabilities, 0.5)
plot_auc(y_test, probabilities)
gb = GradientBoostingClassifier(max_depth=2, n_estimators=500, max_features='auto', loss='exponential' )

gb.fit(X_train, y_train)
probabilities = gb.predict_proba(X_train)

print_metrics(y_train, probabilities, 0.5)
probabilities = gb.predict_proba(X_test)

print_metrics(y_test, probabilities, 0.5)
plot_auc(y_test, probabilities)
import xgboost as xgb
xgb=xgb.XGBClassifier(max_depth=3, n_estimators=500, n_jobs=-1, scale_pos_weight=4)

xgb.fit(X_train, y_train)
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)

plt.show()
probabilities = xgb.predict_proba(X_train)

print_metrics(y_train, probabilities, 0.5)
probabilities = xgb.predict_proba(X_test)

print_metrics(y_test, probabilities, 0.5)
from sklearn.metrics import f1_score

y_pred=xgb.predict(X_test)

f1_score(y_test,y_pred, average='weighted')
plot_auc(y_test, probabilities)
print(xgb.predict(test1))
solution=xgb.predict(test1)

my_submission=pd.DataFrame({'EmployeeNo':df_test['EmployeeNo'],'Promoted_or_Not': solution})

my_submission.to_csv('xgboostFirstSubmission.csv', index=False)
from catboost import CatBoostClassifier
model = CatBoostClassifier (iterations= 1700, learning_rate=0.2, depth=5, verbose=True, scale_pos_weight=2)

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_train)

print_metrics(y_train, probabilities, 0.5)
probabilities = model.predict_proba(X_test)

print_metrics(y_test, probabilities, 0.5)
from sklearn.metrics import f1_score

y_pred=model.predict(X_test)

f1_score(y_test,y_pred, average='weighted')
plot_auc(y_test, probabilities)
print(model.predict(test1))
solution=model.predict(test1)

my_submission=pd.DataFrame({'EmployeeNo':df_test['EmployeeNo'],'Promoted_or_Not': solution})

my_submission.to_csv('CatBoostSubmission.csv', index=False)