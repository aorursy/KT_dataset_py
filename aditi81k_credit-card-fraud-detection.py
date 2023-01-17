#importing library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

%matplotlib inline
train = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# print shape of dataset with rows and columns

print(train.shape)
# print the top5 records

train.head()
# to print the full summary

train.info()
categorical_features=[feature for feature in train.columns if train[feature].dtypes=='O' or train[feature].dtypes== 'int64']

print(categorical_features)
discrete_feature=[feature for feature in train.columns if len(train[feature].unique())<25 and feature not in ['Class']]

print("Discrete Variables Count: {}".format(len(discrete_feature)))
# step make the list of features which has missing values

features_with_na=[features for features in train.columns if train[features].isnull().sum()>1]

# print the missing features list

print(len(features_with_na))
sns.countplot(train['Class'])

plt.show()

print('Percent of fraud transaction: ',len(train[train['Class']==1])/len(train['Class'])*100,"%")

print('Percent of normal transaction: ',len(train[train['Class']==0])/len(train['Class'])*100,"%")
# Lets analyse the continuous values by creating histograms to understand the distribution

data=train.copy()

data.drop(columns='Class', inplace = True)



for feature in data.columns:

    train[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()
#figure_factory module contains dedicated functions for creating very specific types of plots

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class_0 = train.loc[train['Class'] == 0]["Time"]

class_1 = train.loc[train['Class'] == 1]["Time"]

hist_data = [class_0, class_1]

group_labels = ['Not Fraud', 'Fraud']



fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))

iplot(fig, filename='dist_only')
plt.figure(figsize = (14,14))

plt.title('Feature correlation')

corr = train.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")

plt.show()
# Transaction amount 

data=train.copy()

data.drop(columns=['Class'], inplace = True)

for i in data.columns:

  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

  s = sns.boxplot(ax = ax1, x="Class", y=i, hue="Class",data=train, palette="PRGn",showfliers=True)

  s = sns.boxplot(ax = ax2, x="Class", y=i, hue="Class",data=train, palette="PRGn",showfliers=False)

  plt.show();
#Features density plot

col = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',

       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',

       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',

       'V28']



i = 0

t0 = train.loc[train['Class'] == 0]

t1 = train.loc[train['Class'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(8,4,figsize=(16,30))



for feature in col:

    i += 1

    plt.subplot(7,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0", color='b')

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1", color='r')

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
pca_vars = ['V%i' % k for k in range(1,29)]

plt.figure(figsize=(12,4), dpi=80)

sns.barplot(x=pca_vars, y=t0[pca_vars].skew(), color='darkgreen')

plt.xlabel('Column')

plt.ylabel('Skewness')

plt.title('V1-V28 Skewnesses for Class 0')
plt.figure(figsize=(12,4), dpi=80)

sns.barplot(x=pca_vars, y=t1[pca_vars].skew(), color='darkgreen')

plt.xlabel('Column')

plt.ylabel('Skewness')

plt.title('V1-V28 Skewnesses for Class 1')
sns.set_style("whitegrid")

sns.FacetGrid(train, hue="Class", height = 6).map(plt.scatter, "Time", "Amount").add_legend()

plt.show()
FilteredData = train[['Time','Amount', 'Class']]

countLess = FilteredData[FilteredData['Amount'] < 2500]

countMore = train.shape[0] - len(countLess)

percentage = round((len(countLess)/train.shape[0])*100,2)

Class_1 = countLess[countLess['Class'] == 1]

print('Total number for transaction less than 2500 is {}'.format(len(countLess)))

print('Total number for transaction more than 2500 is {}'.format(countMore))

print('{}% of transactions having transaction amount less than 2500' .format(percentage))

print('{} fraud transactions in data where transaction amount is less than 2500' .format(len(Class_1)))
sns.boxplot(x = "Class", y = "Amount", data = train)

plt.ylim(0, 5000)

plt.show()
Amount_0 = train.loc[train['Amount'] == 0]

print(Amount_0['Class'].value_counts())
from sklearn.preprocessing import StandardScaler, RobustScaler

data1=train.copy()

std_scaler = StandardScaler()

rob_scaler = RobustScaler()



train['scaled_amount'] = std_scaler.fit_transform(train['Amount'].values.reshape(-1,1))

train['scaled_time'] = std_scaler.fit_transform(data1['Time'].values.reshape(-1,1))



train.drop(['Amount', 'Time'], axis=1, inplace = True)

scaled_amount = train['scaled_amount']

scaled_time = train['scaled_time']



train.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

train.insert(0, 'scaled_amount', scaled_amount)

train.insert(1, 'scaled_time', scaled_time)

print(train.head())
population = data1[data1['Class'] == 0].Amount

sample = data1[data1['Class'] == 1].Amount

sampleMean = sample.mean()

populationStd = population.std()

populationMean = population.mean()

z_score = (sampleMean - populationMean) / (populationStd / sample.size ** 0.5)

z_score
print(train.head())
from sklearn.model_selection import train_test_split

X = train.drop(['Class'], axis=1)

Y = train['Class']

# This is explicitly used for with data imbalance

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



print(X.shape, Y.shape)
print('X train shape: ', X_train.shape)

print('X test shape: ', X_test.shape)

print('y train shape: ', y_train.shape)

print('y test shape: ', y_test.shape)
print(y_test.value_counts())
# Classifier Libraries

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

# from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections

from sklearn.metrics import make_scorer, precision_score, recall_score, confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score, accuracy_score, average_precision_score, roc_auc_score, precision_recall_fscore_support

from sklearn import metrics

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import  f_classif

from xgboost import XGBClassifier

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier(),

    "RandomForestClassifier": RandomForestClassifier(),

    "BaggingClassifier": BaggingClassifier(n_estimators=10, random_state=0),

    "SGDClassifier" : SGDClassifier(),

    "GradientBoostingClassifier" : GradientBoostingClassifier(),

    "xgb" : XGBClassifier()

}
def plot(df):

  fraud = df[df['class']==1]

  normal = df[df['class']==0]

  fraud.drop(['class'],axis=1,inplace=True)

  normal.drop(['class'],axis=1,inplace=True)

  fraud = fraud.set_index('classifier')

  normal = normal.set_index('classifier')

  plt.figure(figsize = (8,2))

  sns.heatmap(fraud.iloc[:, :], annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"),linewidth=2)

  plt.title('class 1')

  plt.show()

  plt.figure(figsize = (8,2))

  sns.heatmap(normal.iloc[:, :], annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"),linewidth=2)

  plt.title('class 0')

  plt.show()
def roc_curve(y_test, rdict):

  sns.set_style('whitegrid')

  plt.figure()

  i=0

  fig, ax = plt.subplots(4,2,figsize=(16,30))

  for key,val in rdict.items():

    fpr, tpr, thresholds = metrics.roc_curve( y_test, val,

                                                  drop_intermediate = False )

    auc_score = metrics.roc_auc_score( y_test, val)

    i+= 1

    plt.subplot(4,2,i)

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title(key)

    plt.legend(loc="lower right")

  plt.show()
def training(models, x, y, x_t, y_t):

    conf = []

    comp = []

    rdict = {}

    for key, model in models.items():

      model = model.fit(x, y)

      y_pred = model.predict(x_t)

      rdict[key] = y_pred

      tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()

      precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_t, y_pred)

      r1 = {'Classifier': key, 'TN': tn, 'TP': tp, 'FN': fn, 'FP': fp}

      conf.append(r1)

      MCC = matthews_corrcoef(y_t, y_pred)

      AUROC = roc_auc_score(y_t, y_pred)

      Cohen_kappa = cohen_kappa_score(y_t, y_pred)

      accuracy = metrics.accuracy_score(y_t, y_pred)

      r2 = {'classifier': key,'matthews_corrcoef':MCC,'Cohen_kappa':Cohen_kappa,'accuracy': accuracy,'AUROC':AUROC, 'precision': precision[0],'recall':recall[0],'f1':fscore[0], 'class':0}

      r3 = {'classifier': key,'matthews_corrcoef':MCC,'Cohen_kappa':Cohen_kappa,'accuracy': accuracy,'AUROC':AUROC, 'precision': precision[1],'recall':recall[1],'f1':fscore[1], 'class':1}

      comp.append(r2)

      comp.append(r3)

    r11 = (pd.DataFrame(conf).to_markdown())

    r12 = pd.DataFrame(comp)

    print(f'\n\nRoc curve \n\n')

    roc_curve(y_t, rdict)

    print(f'\n\n confusion matrixs comparison \n\n')

    print(r11)

    print(f'\n\n Performance comparison \n\n')

    plot(r12)

    
training(classifiers, X_train, y_train, X_test, y_test)
bestfeatures = SelectKBest(score_func=f_classif, k=10)

fit = bestfeatures.fit(X,Y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score'] 

featureScores_df = featureScores.sort_values(['Score', 'Specs'], ascending=[False, True])  #naming the dataframe columns

print(featureScores_df)
col = ['V17', 'V14', 'V12','V10','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6','V21','V19','V20','V8','V27','scaled_time','V28','V24']
training(classifiers, X_train[col], y_train, X_test[col], y_test)
xgb = XGBClassifier()

# X_train[col], y_train, X_test[col], y_test

xgb.fit(X_train[col],y_train)

y_pred_final = xgb.predict(X_test[col])
submission = pd.DataFrame({'ID':X_test['V17'],'Prediction':y_pred_final})
submission.shape
submission.to_csv('/kaggle/working/submission.csv',index=False)