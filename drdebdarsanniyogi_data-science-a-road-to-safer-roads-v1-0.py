import numpy as np 
import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
# Data file name
filename = "../input/seattle-sdot-collisions-data/Collisions.csv"

# Read the whole data file
df = pd.read_csv(filename, low_memory=False)

## ---------------------------------------------------------------------------------------------------------------
## Read first maxrows of data
## ---------------------------------------------------------------------------------------------------------------

## maxrows = 20000
## df = pd.read_csv(filename, nrows=nrows, skiprows=0, low_memory=False)

## ---------------------------------------------------------------------------------------------------------------
## Read percent_rows % of the data rows
## ---------------------------------------------------------------------------------------------------------------

#percent_rows = 0.50  # 20% of the lines
## keep the header, then take only 20% of lines
## if random from [0,1] interval is greater than 0.01 the row will be skipped

#df = pd.read_csv(filename,header=0, 
#         skiprows=lambda i: i>0 and np.random.random() > percent_rows, low_memory=False)

# shuffle the DataFrame rows 
#df = df.sample(frac = 1) 
#!pip install -q pandas_profiling

#from pandas_profiling import ProfileReport

# Generate profile report
# profile = ProfileReport(df, title="Data Profile Report", explorative=True)
# profile.to_notebook_iframe()
# Convert INCDTTM to date type

df['INCDTTM'] = pd.to_datetime(df['INCDTTM'], errors='coerce')

# Extract month, weekday, hour information

df['Month']=df['INCDTTM'].dt.month
df['Weekday']=df['INCDTTM'].dt.weekday
df['Hour']=df['INCDTTM'].dt.hour
df.drop(['INCDATE', 'INCDTTM'], axis=1, inplace=True)
# Unique ID columns are not predictors, hence drop:
# OBJECTID, INCKEY, COLDETKEY, INTKEY, SEGLANEKEY, CROSSWALKKEY

df.drop(['OBJECTID', 'INCKEY', 'COLDETKEY', 'INTKEY', 'SEGLANEKEY', 'CROSSWALKKEY'], axis=1, inplace=True)
# Undefined Columns:
# X, Y, EXCEPTRSNCODE, EXCEPTRSNDESC, REPORTNO, STATUS, SDOTCOLNUM

df.drop(['X', 'Y', 'EXCEPTRSNCODE', 'REPORTNO', 'STATUS', 'SDOTCOLNUM'], axis=1, inplace=True)
# Drop columns having descriptions corresponding to codes:
# EXCEPTRSNDESC, SEVERITYDESC, SDOT_COLDESC, ST_COLDESC

df.drop(['EXCEPTRSNDESC', 'SEVERITYDESC', 'SDOT_COLDESC', 'ST_COLDESC', 'LOCATION'], axis=1, inplace=True)
df.drop(['SDOT_COLCODE'], axis=1, inplace=True)
# Export the intermediate semi-processed data file
filename = "Collisions_100_initial_processing_before_EDA.csv"
df.to_csv(filename,index=False)
# Read the intermediate semi-processed data file
filename = "Collisions_100_initial_processing_before_EDA.csv"
df = pd.read_csv(filename)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
df['SEVERITYCODE'].value_counts(normalize=True, dropna=False).round(5)
df['SEVERITYCODE'].replace('0', np.nan, inplace=True)
df.dropna(axis=0, how='any',thresh=None, subset=['SEVERITYCODE'], inplace=True)
df['SEVERITYCODE'].replace('3', '4', inplace=True)
df['SEVERITYCODE'].replace('2b', '3', inplace=True)
df['SEVERITYCODE'] =  df['SEVERITYCODE'].astype('int64')
df['SEVERITYCODE'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
countseverity = df.SEVERITYCODE.unique()
count_by_severity=[]
for i in df.SEVERITYCODE.unique():
    count_by_severity.append(df.loc[df.SEVERITYCODE == i, 'SEVERITYCODE'].count())
fig, ax = plt.subplots(figsize=(5,5))
plt.title('Count of Accidents by Severity', y=1.05)
ax.set(xlabel='Severity Code', ylabel='Count')
sns.barplot(countseverity, count_by_severity)
plt.figure(figsize=(10,5))
sns.countplot(x='Hour', hue='SEVERITYCODE', data=df, palette="Set1")
plt.legend(loc='best', prop={'size': 10})
plt.title('Count of Accidents by Hour', y=1.05)
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x='Weekday', hue='SEVERITYCODE', data=df, palette="Set1")
plt.legend(loc='best', prop={'size': 10})
plt.title('Count of Accidents by Weekday', y=1.05)
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x='Month', hue='SEVERITYCODE', data=df, palette="Set1")
plt.legend(loc='best', prop={'size': 10})
plt.title('Count of Accidents by Month', y=1.05)
plt.show()
df['WEATHER'].value_counts(dropna=False, ascending=False)
df.replace({'WEATHER' : {np.nan : 'Unknown'}}, inplace=True)
plt.figure(figsize=(30,5))
sns.countplot(x='WEATHER', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Weather Condition', size=10, y=1.05)
plt.legend(loc='best', prop={'size': 10})
plt.show()
df['ADDRTYPE'].value_counts(normalize=True, ascending=False, dropna=False).round(5)
df.replace({'ADDRTYPE' : {np.nan : 'Unknown'}}, inplace=True)
plt.figure(figsize=(10,5))
sns.countplot(x='ADDRTYPE', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Address Type', size=10, y=1.05)
plt.legend(loc='best', prop={'size': 10})
plt.show()
df['COLLISIONTYPE'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'COLLISIONTYPE' : {np.nan : 'Unknown'}}, inplace=True)
plt.figure(figsize=(15,5))
sns.countplot(x='COLLISIONTYPE', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Collision Type', size=10, y=1.05)
plt.legend(loc='best', prop={'size': 10})
plt.show()
df['JUNCTIONTYPE'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'JUNCTIONTYPE' : {np.nan : 'Unknown'}}, inplace=True)
df.replace({'JUNCTIONTYPE' : {'Mid-Block (not related to intersection)': 'Mid-Block (Not Intersect)', 
                              'At Intersection (intersection related)': 'At Intersection (Intersect)',
                              'Mid-Block (but intersection related)': 'Mid-Block (Intersect)',
                              'At Intersection (but not related to intersection)': 'At Intersection (Intersect)'}}, inplace=True)
plt.figure(figsize=(25,5))
sns.countplot(x='JUNCTIONTYPE', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Junction Type', size=10, y=1.05)
plt.legend(loc='best', prop={'size': 10})
plt.show()
df['UNDERINFL'].value_counts(normalize=True, ascending=False, dropna=False)
df.replace({'UNDERINFL' : {np.nan : 'Unknown', '0': 'N', '1': 'Y'}}, inplace=True)
df['UNDERINFL'].value_counts(normalize=True, ascending=False, dropna=False)
underinfluence = df.UNDERINFL.unique()
count_by_underinfluence=[]
for i in df.UNDERINFL.unique():
    count_by_underinfluence.append(df.loc[df.UNDERINFL == i, 'UNDERINFL'].count())
fig, ax = plt.subplots(figsize=(5,3))
ax.set(xlabel='Under Influence Flag', ylabel='Count', title='Under Influence Flag')
sns.barplot(underinfluence, count_by_underinfluence)
df['ROADCOND'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'ROADCOND' : {np.nan: 'Unknown'}}, inplace=True)
plt.figure(figsize=(10,5))
sns.countplot(x='ROADCOND', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Road Condition Type', size=10, y=1.05)
plt.legend(loc='upper right', prop={'size': 10})
plt.show()
df['LIGHTCOND'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'LIGHTCOND' : {np.nan: 'Unknown'}}, inplace=True)
plt.figure(figsize=(20,5))
sns.countplot(x='LIGHTCOND', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Light Condition Type', size=10, y=1.05)
plt.legend(loc='upper right', prop={'size': 10})
plt.show()
df['HITPARKEDCAR'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
plt.figure(figsize=(10,5))
sns.countplot(x='HITPARKEDCAR', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents where parked car was hit', size=10, y=1.05)
plt.legend(loc='upper right', prop={'size': 10})
plt.show()
df.replace({'ST_COLCODE' : {np.nan: '31', ' ': '31'}}, inplace=True)
df['ST_COLCODE'].value_counts(normalize=True, ascending=False,dropna=False).round(5).head()
plt.figure(figsize=(20,5))
sns.countplot(x='ST_COLCODE', hue='SEVERITYCODE', data=df, palette="Set1")
plt.title('Count of Accidents by Collision Code', size=10, y=1.05)
plt.legend(loc='upper left', prop={'size': 10})
plt.show()
df.groupby(
     ['SEVERITYCODE']
 ).agg(
     sum_INJURIES =            ('INJURIES','sum'),
     sum_SERIOUSINJURIES =     ('SERIOUSINJURIES','sum'),
     sum_FATALITIES =          ('FATALITIES','sum'),
 ).reset_index()
x_cols = [col for col in df.columns if col not in ['SEVERITYCODE'] if ((col =='INJURIES') or (col == 'SERIOUSINJURIES')
                                                                       or (col == 'FATALITIES'))]

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df['SEVERITYCODE'].values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(5,2))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient", fontsize=10)
ax.set_title("Correlation coefficient of the injury variables", fontsize=10)
plt.show()
df['PEDROWNOTGRNT'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'PEDROWNOTGRNT' : {np.nan: 'N'}}, inplace=True)
df['SPEEDING'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'SPEEDING' : {np.nan: 'N'}}, inplace=True)
df['INATTENTIONIND'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df.replace({'INATTENTIONIND' : {np.nan: 'N'}}, inplace=True)
df['PERSONCOUNT'].value_counts(normalize=True, ascending=False,dropna=False).round(5).head()
df['PEDCYLCOUNT'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df['PEDCOUNT'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df['VEHCOUNT'].value_counts(normalize=True, ascending=False,dropna=False).round(5).head()
# Target variable
target='SEVERITYCODE'

df_sev_1 = df.loc[df[target] == 1]
df_sev_1 = df_sev_1.drop(target, axis=1)
df_sev_1 = df_sev_1.mode().T
new_header = ['Mode (S = 1)']
df_sev_1 = df_sev_1[1:]           
df_sev_1.columns = new_header

df_sev_2 = df.loc[df[target] == 2]
df_sev_2 = df_sev_2.drop(target, axis=1)
df_sev_2 = df_sev_2.mode().T
new_header = ['Mode (S = 2)']
df_sev_2 = df_sev_2[1:]           
df_sev_2.columns = new_header

df_sev_3 = df.loc[df[target] == 3]
df_sev_3 = df_sev_3.drop(target, axis=1)
df_sev_3 = df_sev_3.mode().T
new_header = ['Mode (S = 3)']
df_sev_3 = df_sev_3[1:]           
df_sev_3.columns = new_header

df_sev_4 = df.loc[df[target] == 3]
df_sev_4 = df_sev_4.drop(target, axis=1)
df_sev_4 = df_sev_4.mode().T
new_header = ['Mode (S = 4)']
df_sev_4 = df_sev_4[1:]           
df_sev_4.columns = new_header

df_res = pd.concat([df_sev_1, df_sev_2, df_sev_3, df_sev_4], axis=1)
df_res
filename = "Collisions_100_after_EDA_unbalanced.csv"
df.to_csv(filename, index=False)
filename = "Collisions_100_after_EDA_unbalanced.csv"
df = pd.read_csv(filename)
df['SEVERITYCODE'] = df['SEVERITYCODE'].astype('category')
#features = ['ADDRTYPE','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES', 
#            'SERIOUSINJURIES', 'FATALITIES', 'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 
#            'ROADCOND', 'LIGHTCOND', 'PEDROWNOTGRNT', 'SPEEDING', 
#            'ST_COLCODE', 'HITPARKEDCAR', 'Month', 'Weekday', 'Hour']

all_cols = ['SEVERITYCODE','ADDRTYPE','COLLISIONTYPE', 
            'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 
            'ROADCOND', 'LIGHTCOND', 'PEDROWNOTGRNT', 'SPEEDING', 
            'ST_COLCODE', 'HITPARKEDCAR']

all_features = ['ADDRTYPE','COLLISIONTYPE',
            'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 
            'ROADCOND', 'LIGHTCOND', 'PEDROWNOTGRNT', 'SPEEDING', 
            'ST_COLCODE', 'HITPARKEDCAR']

df_sel = df.loc[:, all_cols]    
df_sel[all_features] = df_sel[all_features].astype('category')
df_sel = pd.get_dummies(df_sel, columns=all_features, drop_first=True, dtype='int64')

df_sel.head()
df_sel.drop(df_sel.columns[df_sel.columns.str.contains('Unknown')], axis=1, inplace=True)
# Check if all the data types are int64 or not

df.select_dtypes(exclude=['int64'])
filename = "Collisions_100_after_Feature_Selected_unbalanced.csv"
df_sel.to_csv(filename, index=False)
filename = "Collisions_100_after_Feature_Selected_unbalanced.csv"
df = pd.read_csv(filename)
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

# --------------------------------------------------------------------------------
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
# --------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Target variable
target='SEVERITYCODE'

# set X and y
y = df[target]
X = df.drop(target, axis=1)

X = StandardScaler().fit(X).transform(X)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
bagging = BaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
balanced_bagging = BalancedBaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
brf = BalancedRandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
eec = EasyEnsembleClassifier(n_estimators=10, n_jobs=-1)

bagging.fit(X_train, y_train)
balanced_bagging.fit(X_train, y_train)
brf.fit(X_train, y_train)
eec.fit(X_train, y_train)

y_pred_bc = bagging.predict(X_test)
y_pred_bbc = balanced_bagging.predict(X_test)
y_pred_brf = brf.predict(X_test)
y_pred_eec = eec.predict(X_test)

fig, ax = plt.subplots(ncols=4, figsize=(20,20))

cm_bagging = confusion_matrix(y_test, y_pred_bc)
plot_confusion_matrix(cm_bagging, classes=np.unique(df[target]), ax=ax[0],
                      title='Bagging\nBalanced accuracy: {:.2f}'.format(balanced_accuracy_score(y_test, y_pred_bc)))

cm_balanced_bagging = confusion_matrix(y_test, y_pred_bbc)
plot_confusion_matrix(cm_balanced_bagging, classes=np.unique(df[target]), ax=ax[1],
                      title='Balanced bagging\nBalanced accuracy: {:.2f}'.format(balanced_accuracy_score(y_test, y_pred_bbc)))

cm_brf = confusion_matrix(y_test, y_pred_brf)
plot_confusion_matrix(cm_brf, classes=np.unique(df[target]), ax=ax[2],
                      title='Balanced Random Forest\nBalanced accuracy: {:.2f}'.format(balanced_accuracy_score(y_test, y_pred_brf)))

cm_eec = confusion_matrix(y_test, y_pred_eec)
plot_confusion_matrix(cm_eec, classes=np.unique(df[target]), ax=ax[3],
                      title='Balanced EasyEnsemble\nBalanced accuracy: {:.2f}'.format(balanced_accuracy_score(y_test, y_pred_eec)))


plt.show()
filename = "Collisions_100_after_Feature_Selected_unbalanced.csv"
df = pd.read_csv(filename)
df['SEVERITYCODE'].value_counts(normalize=True, ascending=False,dropna=False).round(5)
df['Severity 4'] = 0
df.loc[df['SEVERITYCODE'] == 4, 'Severity 4'] = 1
df['Severity 4'].value_counts()
df = pd.concat([df[df['Severity 4']==1].sample(10000, replace = True),
                   df[df['Severity 4']==0].sample(10000)], axis=0)
print('Resampled data:', df['Severity 4'].value_counts())
df['Severity 4'].value_counts(normalize=True, ascending=False, dropna=False).round(5)
df.drop(['SEVERITYCODE'], axis=1, inplace=True)
filename = "Collisions_100_after_Feature_Selected_balanced.csv"
df.to_csv(filename,index=False)
filename = "Collisions_100_after_Feature_Selected_balanced.csv"
df = pd.read_csv(filename)
import math
import warnings

warnings.simplefilter(action='ignore', category=Warning)

df_sel = df.drop(df.columns[df.columns.str.contains('ST_COLCODE')], axis=1)

x_cols = [col for col in df_sel.columns if col not in ['Severity 4'] if (df[col].dtype=='int64')]

labels = []
values = []
for col in x_cols:
    if not (math.isnan(np.corrcoef(df_sel[col].values, df_sel['Severity 4'].values)[0,1])):
        labels.append(col)
        values.append(np.corrcoef(df_sel[col].values, df_sel['Severity 4'].values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(10,12))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='g')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient", fontsize=10)
ax.set_title("Correlation coefficient of the variables with respect to Severity Code", fontsize=10)
plt.show()
filename = "Collisions_100_after_Feature_Selected_balanced.csv"
df = pd.read_csv(filename)
from sklearn import preprocessing

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# Target variable
target='Severity 4'

# set X and y
y = df[target]
X = df.drop(target, axis=1)

X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

# List of classification algorithms
algorithm_list=['Logistic Regression', 'k-Nearest Neighbors', 'Decision Trees', 'Random Forest']

# Initialize an empty list for the accuracy for each algorithm
accuracy_list=[]
# Logistic regression with default setting.
from sklearn.linear_model import LogisticRegression

# Classifier Model = Logistic Regression
lreg_clf = LogisticRegression(max_iter=10000, random_state=42)

lreg_clf.fit(X_train, y_train)

lreg_accuracy_train = lreg_clf.score(X_train, y_train)
print("Training Accuracy: %.1f%%"% (lreg_accuracy_train*100))

lreg_accuracy_test = lreg_clf.score(X_test, y_test)
print("Testing Accuracy: %.1f%%"% (lreg_accuracy_test*100))
#Grid Search
from sklearn.model_selection import GridSearchCV

LR_grid = {
           'C':        [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25],
           'max_iter': [1000, 10000, 100000]
          }

lr_cv = GridSearchCV(estimator=LogisticRegression(random_state=42), param_grid = LR_grid, scoring = 'accuracy', cv = 5)

lr_cv.fit(X_train, y_train)
print('Best Parameters: ', lr_cv.best_params_)
%%time
from sklearn.metrics import confusion_matrix, accuracy_score

# Classifier Model = Logistic Regression
lreg_clf = LogisticRegression(C=5, max_iter=1000, penalty='l2')

lreg_clf.fit(X_train, y_train)

lreg_accuracy_train = lreg_clf.score(X_train, y_train)
print("Training Accuracy: %.1f%%"% (lreg_accuracy_train*100))

lreg_accuracy_test = lreg_clf.score(X_test, y_test)
print("Testing Accuracy: %.1f%%"% (lreg_accuracy_test*100))

# Append to the accuracy list
accuracy_list.append(lreg_accuracy_test)
from sklearn.metrics import confusion_matrix

y_pred = lreg_clf.predict(X_test)

lreg_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

df_conf = pd.DataFrame(data=lreg_cm, columns=['Predicted: 0','Predicted: 1'], index=['Actual: 0','Actual: 1'])

plt.figure(figsize = (5,3))

sns.heatmap(df_conf, annot=True, fmt='d', cmap="YlGnBu").set_title(
            "Logistic Regression\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                lreg_accuracy_train*100, lreg_accuracy_test*100), fontsize=12)

plt.show()
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create a k-NN classifier with 7 neighbors
knn_clf = KNeighborsClassifier(n_neighbors=7)

knn_clf.fit(X_train, y_train)

knn_accuracy_train = knn_clf.score(X_train, y_train)
print("Train Accuracy: %.1f%%"% (knn_accuracy_train*100))

knn_accuracy_test = knn_clf.score(X_test,y_test)
print("Test Accuracy: %.1f%%"% (knn_accuracy_test*100))

# Append to the accuracy list
accuracy_list.append(knn_accuracy_test)
from sklearn.metrics import confusion_matrix

y_pred = knn_clf.predict(X_test)

knn_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

df_conf = pd.DataFrame(data=knn_cm, columns=['Predicted: 0','Predicted: 1'], index=['Actual: 0','Actual: 1'])

plt.figure(figsize = (5,3))
sns.heatmap(df_conf, annot=True,fmt='d',cmap="YlGnBu").set_title(
    "k-NN\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                knn_accuracy_train*100, knn_accuracy_test*100), fontsize=12)
plt.show()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

DT_grid = {'min_samples_split': [5, 10, 20, 30], 
           'max_features': [None, 'log2', 'sqrt']}

CV_DT = GridSearchCV(DecisionTreeClassifier(random_state=42), DT_grid, verbose=1, cv=3)
CV_DT.fit(X_train, y_train)

print('Best Parameters: ', CV_DT.best_params_)
%%time
from sklearn import tree

# Training step, on X_train with y_train
tree_clf = tree.DecisionTreeClassifier(min_samples_split = 5, max_features = 'log2', 
                                       class_weight='balanced', random_state=42)
tree_clf = tree_clf.fit(X_train, y_train)

tree_accuracy_train = tree_clf.score(X_train, y_train)
print("Train Accuracy: %.1f%%"% (tree_accuracy_train*100))

tree_accuracy_test = tree_clf.score(X_test,y_test)
print("Test Accuracy: %.1f%%"% (tree_accuracy_test*100))

# Append to the accuracy list
accuracy_list.append(tree_accuracy_test)
y_pred = tree_clf.predict(X_test)

tree_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

df_conf = pd.DataFrame(data=tree_cm, columns=['Predicted: 0','Predicted: 1'], index=['Actual: 0','Actual: 1'])

plt.figure(figsize=(5, 3))
sns.heatmap(df_conf, annot=True, fmt='d',cmap="YlGnBu").set_title(
    "Decision Tree\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                tree_accuracy_train*100, tree_accuracy_test*100), fontsize=12)
plt.show()
fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(tree_clf, max_depth=4, fontsize=10,
               feature_names=df.drop('Severity 4', axis=1).columns.to_list(), class_names=True, filled=True)
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid ={'bootstrap': [True, False],
 'max_depth': [5, 10, 20, 30, 40],
 'max_features': ['auto', 'sqrt'],
 'n_estimators': [10, 20, 30]
            }

CV_RF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid,cv=4)
CV_RF.fit(X_train, y_train)
print('Best Parameters: ', CV_RF.best_params_)
%%time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(bootstrap=False, max_depth=40, max_features='sqrt', n_estimators=20, random_state=42)
rf_clf.fit(X_train,y_train)

f = lambda x: 1 if x>=0.5 else 0
train_pred = np.array(list(map(f, rf_clf.predict(X_train))))
test_pred = np.array(list(map(f, rf_clf.predict(X_test))))

rf_train_accuracy = accuracy_score(y_train, train_pred)
print("Train Accuracy: %.1f%%"% (rf_train_accuracy*100))

rf_test_accuracy = accuracy_score(y_test, test_pred)
print("Test Accuracy: %.1f%%"% (rf_test_accuracy*100))

# Append to the accuracy list
accuracy_list.append(rf_test_accuracy)
rf_cm = confusion_matrix(y_true=y_test, y_pred=test_pred)

df_conf = pd.DataFrame(data=rf_cm, columns=['Predicted: 0','Predicted: 1'], index=['Actual: 0','Actual: 1'])

plt.figure(figsize = (5, 3))
sns.heatmap(df_conf, annot=True,fmt='d',cmap="YlGnBu").set_title(
    "Random Forest\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                rf_train_accuracy*100, rf_test_accuracy*100), fontsize=12)
plt.show()
# Generate a list of ticks for y-axis
y_ticks = np.arange(len(algorithm_list))

# Combine the list of algorithms and list of accuracy scores into a dataframe, sort the value based on accuracy score
df_accuracy = pd.DataFrame(list(zip(algorithm_list, accuracy_list)), 
                    columns=['Algorithm','Accuracy_Score']).sort_values(by=['Accuracy_Score'], ascending = True)

# Make a plot
ax = df_accuracy.plot.barh('Algorithm', 'Accuracy_Score', align='center', legend=False, color='g')

# Add the data label on to the plot
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+0.02, i.get_y()+0.2, str(round(i.get_width(),2)), fontsize=10)

# Set the limit, lables, ticks and title
plt.xlim(0, 1.1)
plt.xlabel('Accuracy Score')
plt.yticks(y_ticks, df_accuracy['Algorithm'], rotation=0)
plt.title('Algorithm performance')

plt.show()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))

plot_confusion_matrix(lreg_cm, classes=np.unique(df[target]), ax=ax[0],
                      title="Logistic Regression\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                      lreg_accuracy_train*100, lreg_accuracy_test*100))

plot_confusion_matrix(knn_cm, classes=np.unique(df[target]), ax=ax[1],
                      title="k-NN\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                             knn_accuracy_train*100, knn_accuracy_test*100))

plot_confusion_matrix(tree_cm, classes=np.unique(df[target]), ax=ax[2],
                      title="Decision Tree\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                      tree_accuracy_train*100, tree_accuracy_test*100))

plot_confusion_matrix(rf_cm, classes=np.unique(df[target]), ax=ax[3],
                      title="Random Forest\nTrain Acc %: {:.2f} Test Acc %: {:.2f}".format(
                      rf_train_accuracy*100, rf_test_accuracy*100))


plt.show()
importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=['importance'], 
                           index=df.drop('Severity 4',axis=1).columns)

importances.iloc[:,0] = tree_clf.feature_importances_

importances.sort_values(by='importance', inplace=True, ascending=False)
importancestop = importances.head(30)

plt.figure(figsize=(10, 10))
sns.barplot(x='importance', y=importancestop.index, data=importancestop)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.title('Feature Importance (using Decision Tree Classifier)', size=15)

plt.show()
importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=['importance'], index=df.drop('Severity 4',axis=1).columns)

importances.iloc[:,0] = rf_clf.feature_importances_

importances.sort_values(by='importance', inplace=True, ascending=False)
importancestop = importances.head(30)

plt.figure(figsize=(10, 10))
sns.barplot(x='importance', y=importancestop.index, data=importancestop)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.title('Feature Importance (using Random Forest Classifier)', size=15)

plt.show()
