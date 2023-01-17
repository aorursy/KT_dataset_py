import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

import pandas as pd

full_data = pd.read_csv("../input/gimana/GIM_Dataset.csv")

#full_data

y = full_data['Target']

sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(full_data, y, test_size = 0.2, random_state = 42)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)
na_columns = X_train.isna().sum()

print(na_columns[na_columns!=0])
sns.distplot(X_train['emi'],color='g')

s = 'emi'

l = X_train[s].mean()-3*X_train[s].std()

u = X_train[s].mean()+ 3*X_train[s].std()

ind = X_train[X_train[s]>u].index

X_train[X_train[s]>u].shape
X_train= X_train.drop(ind)

sns.distplot(X_train['emi'],color='g')
X_train
X_train['pinCode'].nunique()
X_test['pinCode'].nunique()
"""

X_train['pinModelCode'] = X_train['pinCode'].astype(str)+"_"+X_train['modelCode'].astype(str)

X_train['pinModelCode'] = X_train['pinCode'].astype(str)+"_"+X_train['modelCode'].astype(str)

le = LabelEncoder()

X_train['pinModelCode'] = le.fit_transform(X_train['pinModelCode'])

X_test['pinModelCode'] = le.fit_transform(X_test['pinModelCode'])

"""
X_train["emi_to_mean_timesFlowedTotal"] = X_train.groupby(['timesFlowedTotal'])['emi'].transform('mean')

userID_mean_dict = X_train.groupby(['timesFlowedTotal'])['emi'].mean().to_dict()

X_test['emi_to_mean_timesFlowedTotal'] = X_test['timesFlowedTotal'].apply(lambda x:userID_mean_dict.get(x,0))
X_train["emi_to_mean_dealerCode"] = X_train.groupby(['dealerCode'])['emi'].transform('mean')

userID_mean_dict = X_train.groupby(['dealerCode'])['emi'].mean().to_dict()

X_test['emi_to_mean_dealerCode'] = X_test['dealerCode'].apply(lambda x:userID_mean_dict.get(x,0))
X_train["emi_to_mean_modelCode"] = X_train.groupby(['modelCode'])['emi'].transform('mean')

userID_mean_dict = X_train.groupby(['dealerCode'])['emi'].mean().to_dict()

X_test['emi_to_mean_modelCode'] = X_test['dealerCode'].apply(lambda x:userID_mean_dict.get(x,0))
numericCols = ['emi_to_mean_timesFlowedTotal','emi_to_mean_dealerCode','emi_to_mean_modelCode','timesFlowedTotalbyMOB','timesFlowed3M','timesBounced3M','averageDelayEMI','timesFlowedTotal','numberEnquired9M','numberEnquired1M','emi','maxContMobFlowed', 'tenureinMonths','allocatedToColAgent', 'allocatedToFC','numberLiveUnSecAcc']

categoricalCols = ['Target','flowedInLastMonth','pinCode','dealerCode','bounceReason','aadharAvailable','modelCode','schemeType','surrogate','qualification','productCode','panVoteIdAvailable','residentType']
X_train_num = X_train[numericCols]

correlationMatrix = X_train_num.corr().abs()

plt.figure(figsize=(30,16))

heat = sns.heatmap(data=correlationMatrix, annot=True,annot_kws={'size':16},cbar=False,cmap='YlGnBu')

plt.title('Heatmap of Pearson Correlation')

plt.yticks(fontsize=16)

plt.xticks(fontsize=16,rotation='vertical')
upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

to_drop

for i_var in to_drop:

    numericCols.remove(i_var)

numericCols
to_drop
import scipy.stats as stats

from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
cT = ChiSquare(X_train)

#Feature Selection

for var in categoricalCols:

    cT.TestIndependence(colX=var,colY="Target" ) 
categoricalCols.remove('aadharAvailable')
features = numericCols + categoricalCols

X_train = X_train.loc[:,features]
enc = ['pinCode','dealerCode','bounceReason','panVoteIdAvailable','modelCode','schemeType','surrogate','qualification','productCode','residentType']

from sklearn.preprocessing import LabelEncoder

for c in enc:

    lb = LabelEncoder()

    X_train[c] = lb.fit_transform(X_train[c])
X_train
X_test
features = numericCols + categoricalCols

X_test = X_test.loc[:,features]



enc = ['pinCode','dealerCode','bounceReason','panVoteIdAvailable','modelCode','schemeType','surrogate','qualification','productCode','residentType']

from sklearn.preprocessing import LabelEncoder

for c in enc:

    lb = LabelEncoder()

    X_test[c] = lb.fit_transform(X_test[c])

X_test
y_train = X_train['Target']

X_train.drop(['Target'],1,inplace=True)
X_test.drop(['Target'],1,inplace=True)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 42, class_weight='balanced')

classifier.fit(X_train, y_train)

predictions = classifier.predict_proba(X_test)[:,1]
logreg_predictionsTest = classifier.predict_proba(X_train)[:,1]
y_pred = classifier.predict(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score

print(f1_score(y_test, y_pred, average="macro"))

print(precision_score(y_test, y_pred, average="macro"))

print(recall_score(y_test, y_pred, average="macro"))
y1 = pd.DataFrame(predictions,columns=['prob'])

y1['hard'] = np.where(y1['prob']<0.5,0,1)

print(y1.groupby(['hard'])['prob'].count())

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y1['hard'])

print(cm)

logreg_real_auc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print("CM Accuracy : "+str(logreg_real_auc))

logreg_pred = pd.DataFrame(predictions,columns=['Target'])

#a = cm[1,1]/(cm[1,1]+cm[1,0])#maximize

#print("Recall: ", a)



print("Cost Savings : ",155*cm[1,1]+0*cm[0,0]-180*cm[1,0]-25*cm[0,1])
fprs_classifier, tprs_classifier, scores_classifier = [], [], []

from sklearn.metrics import roc_curve, auc

y_predict = classifier.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve( y_test, y_predict)

auc_score = auc(fpr, tpr)

logreg_auc_score = auc_score

scores_classifier.append(logreg_auc_score)

fprs_classifier.append(fpr)

tprs_classifier.append(tpr)

print("AUC : "+str(logreg_auc_score))

plt.figure(figsize=(16, 6))

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.5f)' % logreg_auc_score)  

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show() 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=5,class_weight = 'balanced',random_state = 0)

classifier.fit(X_train, y_train)

predictions = classifier.predict_proba(X_test)[:,1]
dt_predictionsTest = classifier.predict_proba(X_train)[:,1]
y_pred = classifier.predict(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score

print(f1_score(y_test, y_pred, average="macro"))

print(precision_score(y_test, y_pred, average="macro"))

print(recall_score(y_test, y_pred, average="macro"))
y1 = pd.DataFrame(predictions,columns=['prob'])

y1['hard'] = np.where(y1['prob']<0.5,0,1)

print(y1.groupby(['hard'])['prob'].count())

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y1['hard'])

print(cm)

rf_real_auc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print("CM Accuracy : "+str(rf_real_auc))

rf_pred = pd.DataFrame(predictions,columns=['Target'])

#a = cm[1,1]/(cm[1,1]+cm[1,0])#maximize

#print("Recall: ", a)

print("Cost Savings : ",155*cm[1,1]+0*cm[0,0]-180*cm[1,0]-25*cm[0,1])
from sklearn.metrics import roc_curve, auc

y_predict = classifier.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve( y_test, y_predict)

auc_score = auc(fpr, tpr)

rf_auc_score = auc_score

scores_classifier.append(rf_auc_score)

fprs_classifier.append(fpr)

tprs_classifier.append(tpr)

print("AUC : "+str(rf_auc_score))

plt.figure(figsize=(16, 6))

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.5f)' % rf_auc_score)  

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show() 
from lightgbm import LGBMClassifier

feature_importances = pd.DataFrame()

feature_importances['feature'] = X_train.columns

from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.metrics import roc_curve, auc

cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

for (train, test), i in zip(cv.split(X_train, y_train), range(5)):

    classifier = LGBMClassifier(random_state = 42)

    classifier.fit(X_train.iloc[train,:], y_train.iloc[train])

    feature_importances['fold_{}'.format(i + 1)] = classifier.feature_importances_

feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(cv.n_splits)]].mean(axis=1)

plt.figure(figsize=(15, 8))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False), x='average', y='feature');

plt.title(' TOP feature importance over {} folds average'.format(cv.n_splits),fontsize=15)

plt.xticks(fontsize=15)

plt.ylabel('Features',fontsize=20)

plt.yticks(fontsize=15)
rankings = classifier.feature_importances_.tolist()

features = list(X_train)

d = dict(zip(features,rankings))

d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])

d.sort_values(["ranking"], ascending=False)
plt.figure(figsize=(16, 6))

lw = 2

model_list = ['Logistic Regression','Random Forest']

colors = ['darkorange','green']

for i in range(0,2):

    plt.plot(fprs_classifier[i], tprs_classifier[i], color=colors[i],

             lw=lw, label="{} (area = {})".format(model_list[i],round(scores_classifier[i],4)))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate',fontsize=15)

plt.title('ROC Area Under Curve',fontsize=20)

plt.ylabel('True Positive Rate',fontsize=15)

plt.legend(loc="lower right",fontsize=17)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.show()