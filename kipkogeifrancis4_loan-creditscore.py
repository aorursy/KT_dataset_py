import scipy



import pandas as pd

import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.manifold import TSNE

from IPython.core.interactiveshell import InteractiveShell

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

warnings.filterwarnings('ignore') # to supress seaborn warnings

pd.options.display.max_columns = None # Remove pandas display column number limit

#InteractiveShell.ast_node_interactivity = "all" # Display all values of a jupyter notebook cell

import sys

#import savReaderWriter as sav

#import the evaluatation metric

from sklearn.metrics import balanced_accuracy_score

#pandas library for reading data

import pandas as pd

#numpy library for computation with matrices and arrays

import numpy as np

#matplotlib library for visualization

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore')

#command for displaying visualizations within the notebook

%matplotlib inline



from sklearn.metrics import confusion_matrix, recall_score, precision_score



import seaborn as sns
# Read the data into DataFrames.

LOANDATA=pd.read_csv("../input/LOANDATA1.csv")

# Breif look at the data

LOANDATA.head()
#Check feaures of the data , data types and number of rows for each column

LOANDATA.info()
#View some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values

LOANDATA.describe()
#Check total number of rows and columns within dataframe

LOANDATA.shape
import pandas_profiling as pp

pp.ProfileReport(LOANDATA)
#Checking further infomation about CreditScoreGroup

LOANDATA.CreditScoreGroup.value_counts()
#Checking further infomation about Province

LOANDATA.Province.value_counts()
#Checking further infomation about Class

LOANDATA.Class.value_counts()
LOANDATA.isna().sum()
sns.heatmap(LOANDATA.isnull(),cmap="winter_r")
#Making a copy of loandata so that we can utilize the copy for further feature enginneering

LOANDATA1=LOANDATA.copy()


cor = LOANDATA1.corr()

plt.figure(figsize=(18,18))

sns.heatmap(cor, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},

            xticklabels=cor.columns.values,

            yticklabels=cor.columns.values)
#replacing spaced words in culumns (' ', '_')  with underscore for ease 

LOANDATA1.columns =LOANDATA1.columns.str.strip().str.replace(' ', '_')

LOANDATA1.head(1)
#LOANDATA1['Unnamed:_0']

LOANDATA1.drop('Unnamed_0', axis=1,inplace=True)



LOANDATA1.set_index(['Customer_Id'],inplace=True)
LOANDATA1.Class.value_counts()
LOANDATA1.head()
Creditscore_count =LOANDATA1.CreditScoreGroup.value_counts()

print('Class 0:', Creditscore_count[0])

print('Class 1:', Creditscore_count[1])

print('Class 2:', Creditscore_count[2])

print('Proportion:', round(Creditscore_count[0] / (Creditscore_count[1]+Creditscore_count[2]), 3), ': 1')



Creditscore_count.plot(kind='bar', title='Count (Creditscore)');
LOANDATA1.drop('Class',axis=1,inplace=True)
LOANDATA1["CreditScoreGroup"]=LOANDATA1["CreditScoreGroup"].map({"A": 0, "B": 1, "C": 2}).astype(int)
LOANDATA1=LOANDATA1.fillna(value={"Province":"Umujyi wa Kigali"})
LOANDATA1["Province"]=LOANDATA1["Province"].map({"Umujyi wa Kigali": 0, "Iburasirazuba": 1, "Iburengerazuba": 2,'Amajyepfo':3,"Amajyaruguru":4,"Diaspora - A":5}).astype(int)
LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].str.strip().str.replace(' ', '_')
LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].map({"Completely_Repaid":0,"Partially_Repaid":1,"In_arrears":2,"Not_yet":3}).astype(int)
LOANDATA1["ReturningCustomer"]=(LOANDATA1["ReturningCustomer"]).astype(int)
LOANDATA4=LOANDATA1.drop(['District','DOB','Effective_Date','Maturity_Date','Date_of_Birth'], axis=1)

#Set df4 equal to a set of a sample of 1000 deafault and 1000 non-default observations.

df1 = LOANDATA4[LOANDATA1.CreditScoreGroup == 0].sample(n = 1000)

df2 = LOANDATA4[LOANDATA1.CreditScoreGroup == 1].sample(n = 1000)

df3 = LOANDATA4[LOANDATA1.CreditScoreGroup == 2].sample(n = 1000)

df4 = pd.concat([df1,df2,df3], axis = 0)



#Scale features to improve the training ability of TSNE.

standard_scaler = StandardScaler()

df4_std = standard_scaler.fit_transform(df4)



#Set y equal to the target values.

y = df4.CreditScoreGroup



tsne = TSNE(n_components=2, random_state=0)

x_test_2d = tsne.fit_transform(df4_std)



Creditscore_count =LOANDATA1.CreditScoreGroup.value_counts()

print('Class 0:', Creditscore_count[0])

print('Class 1:', Creditscore_count[1])

print('Class 2:', Creditscore_count[2])

print('Proportion:', round(Creditscore_count[0] / (Creditscore_count[1]+Creditscore_count[2]), 3), ': 1')



Creditscore_count.plot(kind='bar', title='Count (Creditscore)');
#Build the scatter plot with the three types of transactions.

color_map = {0:'red', 1:'blue',2:'yellow'}

plt.figure()

for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x = x_test_2d[y==cl,0], y = x_test_2d[y==cl,1], c = color_map[idx], label = cl)

    #plt.scatter(x = x_test_2d[y==cl,0], y = x_test_2d[y==cl,2], c = color_map[idx], label = cl)

plt.xlabel('X in t-SNE')

plt.ylabel('Y in t-SNE')

plt.legend(loc='upper right')

plt.title('t-SNE visualization of train data')

plt.show()
plt.figure(figsize=(10,10));

LOANDATA4.hist(figsize=(10,10));


cor = LOANDATA4.corr()

plt.figure(figsize=(18,18))

sns.heatmap(cor, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},

            xticklabels=cor.columns.values,

            yticklabels=cor.columns.values)
PaymentStatus_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['PaymentStatus'], margins=True, normalize=False)

new_index = {0: 'A', 1:'B',2: 'C', }

new_columns = {0:"Completely_Repaid",1:"Partially_Repaid",2:"In_arrears",3:"Not_yet"}

PaymentStatus_crosstab.rename(index=new_index, columns=new_columns, inplace=True)

PaymentStatus_crosstab/PaymentStatus_crosstab.loc['All']
ReturningCustomer_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['ReturningCustomer'], margins=True, normalize=False)

new_index = {0: 'A', 1:'B',2: 'C', }

new_columns = {0: 'False', 1:'True'}

ReturningCustomer_crosstab.rename(index=new_index, columns=new_columns, inplace=True)

ReturningCustomer_crosstab/ReturningCustomer_crosstab.loc['All']
pen_interest_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['Due_pen_interest'], margins=True, normalize=False)

new_index = {0: 'A', 1:'B',2: 'C', }

new_columns = {}

pen_interest_crosstab.rename(index=new_index, columns=new_columns, inplace=True)

pen_interest_crosstab/pen_interest_crosstab.loc['All']
age_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['age'], margins=True, normalize=False)

new_index = {0: 'A', 1:'B',2: 'C', }

new_columns = {}

age_crosstab.rename(index=new_index, columns=new_columns, inplace=True)

age_crosstab/age_crosstab.loc['All']
Duration_crosstab = pd.crosstab(LOANDATA4['CreditScoreGroup'], LOANDATA4['Duration'], margins=True, normalize=False)

new_index = {0: 'A', 1:'B',2: 'C', }

new_columns = {}

Duration_crosstab.rename(index=new_index, columns=new_columns, inplace=True)

Duration_crosstab/Duration_crosstab.loc['All']
#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import preprocessing, metrics

from xgboost import XGBClassifier

warnings.filterwarnings('ignore') # to supress warnings

from sklearn.model_selection import learning_curve, GridSearchCV

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import preprocessing, metrics

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore') # to supress warnings

#Import libraries:

import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

#from sklearn import cross_validation, metrics   #Additional scklearn functions

#from sklearn.grid_search import GridSearchCV   #Perforing grid search



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
x= LOANDATA4.drop(columns=['CreditScoreGroup','Due_Principal','Due_interest',

                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)

Y = LOANDATA4.CreditScoreGroup

scaler=StandardScaler()

X=scaler.fit(x).transform(x)

# train is now 75% of the entire data set

# the _junk suffix means that we drop that variable completely

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)



# list of different classifiers we are going to test

clfs = {

'LogisticRegression' : LogisticRegression(),

'GaussianNB': GaussianNB(),

'RandomForest': RandomForestClassifier(),

'DecisionTreeClassifier': DecisionTreeClassifier(),

'SVM': SVC(),

'KNeighborsClassifier': KNeighborsClassifier(),

'GradientBoosting': GradientBoostingClassifier(),

'XGBClassifier': XGBClassifier()

}
# code block to test all models in clfs and generate a report

models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    y_score = clf.score(x_test,y_test)

    

    #print('Calculating {}'.format(clf_name))

    t = pd.Series({ 

                     'Model': clf_name,

                     'Precision_score': metrics.precision_score(y_test, y_pred,average='macro'),

                     'Recall_score': metrics.recall_score(y_test, y_pred,average='macro'),

                     'F1_score': metrics.f1_score(y_test, y_pred,average='macro'),

                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}

                   )



    models_report = models_report.append(t, ignore_index = True)



models_report
# Function to optimize model using gridsearch 

def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):

    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)

    gs.fit(x_train, y_train)

    print ('Best params: ', gs.best_params_)

    print ('Best AUC on Train set: ', gs.best_score_)

    print( 'Best AUC on Test set: ', gs.score(x_test, y_test))



# Function to generate confusion matrix

def confmat(pred, y_test):

    conmat = np.array(confusion_matrix(y_test, pred, labels=[0,1,2]))

    conf = pd.DataFrame(conmat, index=['A', 'B','C'],

                             columns=['Predicted A', 'Predicted B','Predicted C'])

    print( conf)



# Function to plot roc curve

def roc(prob, y_test):

    y_score = prob

    fpr = dict()

    tpr = dict()

    roc_auc=dict()

    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)

    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure(figsize=[7,7])

    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)

    plt.plot([1,0], [1,0], 'k--', linewidth=4)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive rate', fontsize=15)

    plt.ylabel('True Positive rate', fontsize=15)

    plt.title('ROC curve for Credit Default', fontsize=16)

    plt.legend(loc='Lower Right')

    plt.show()

    

def model(md, x_train, y_train,x_test, y_test):

    md.fit(x_train, y_train)

    pred = md.predict(x_test)

    #prob = md.predict_proba(x_test)[:,1]

    print( ' ' )

    print ('Accuracy on Train set: ', md.score(x_train, y_train))

    print( 'Accuracy on Test set: ', md.score(x_test, y_test))

    print( ' ')

    print(classification_report(y_test, pred))

    print( ' ')

    print('Confusion Matrix',confmat(pred, y_test))

    

    #roc(prob, y_test)

    return md
x= LOANDATA4.drop(columns=['CreditScoreGroup','Due_Principal','Due_interest',

                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)

Y = LOANDATA4.CreditScoreGroup

train_ratio = 0.75

validation_ratio = 0.15

test_ratio = 0.10

pca = PCA(n_components = 23)

kfold = 5



scaler=StandardScaler()

X=scaler.fit(x).transform(x)

# train is now 75% of the entire data set

# the _junk suffix means that we drop that variable completely

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)

StratifiedKFold(n_splits=kfold, random_state=42)

# test is now 10% of the initial data set

# validation is now 15% of the initial data set

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 



#print(x_train, x_val, x_test
clfs = {

'LogisticRegression' : LogisticRegression(),

'GaussianNB': GaussianNB(),

'RandomForest': RandomForestClassifier(),

'DecisionTreeClassifier': DecisionTreeClassifier(),

'SVM': SVC(),

'KNeighborsClassifier': KNeighborsClassifier(),

'GradientBoosting': GradientBoostingClassifier(),

'XGBClassifier': XGBClassifier()

}


 # code block to test all models in clfs and generate a report

models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_val)

    y_score = clf.score(x_val,y_val)

    

    #print('Calculating {}'.format(clf_name))

    t = pd.Series({ 

                     'Model': clf_name,

                     'Precision_score': metrics.precision_score(y_val, y_pred,average='macro'),

                     'Recall_score': metrics.recall_score(y_val, y_pred,average='macro'),

                     'F1_score': metrics.f1_score(y_val, y_pred,average='macro'),

                     'Accuracy': metrics.accuracy_score(y_val, y_pred)}

                   )



    models_report = models_report.append(t, ignore_index = True)



models_report
# code block to test all models in clfs and generate a report

models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    y_score = clf.score(x_test,y_test)

    

    #print('Calculating {}'.format(clf_name))

    t = pd.Series({ 

                     'Model': clf_name,

                     'Precision_score': metrics.precision_score(y_test, y_pred,average='macro'),

                     'Recall_score': metrics.recall_score(y_test, y_pred,average='macro'),

                     'F1_score': metrics.f1_score(y_test, y_pred,average='macro'),

                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}

                   )



    models_report = models_report.append(t, ignore_index = True)



models_report
# Function to optimize model using gridsearch 

def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):

    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)

    gs.fit(x_train, y_train)

    print ('Best params: ', gs.best_params_)

    print ('Best AUC on Train set: ', gs.best_score_)

    print( 'Best AUC on Test set: ', gs.score(x_test, y_test))



# Function to generate confusion matrix

def confmat(pred, y_test):

    conmat = np.array(confusion_matrix(y_test, pred, labels=[0,1,2]))

    conf = pd.DataFrame(conmat, index=['A', 'B','C'],

                             columns=['Predicted A', 'Predicted B','Predicted C'])

    print( conf)



# Function to plot roc curve

def roc(prob, y_test):

    y_score = prob

    fpr = dict()

    tpr = dict()

    roc_auc=dict()

    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)

    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure(figsize=[7,7])

    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)

    plt.plot([1,0], [1,0], 'k--', linewidth=4)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive rate', fontsize=15)

    plt.ylabel('True Positive rate', fontsize=15)

    plt.title('ROC curve for Credit Default', fontsize=16)

    plt.legend(loc='Lower Right')

    plt.show()

    

def model(md, x_train, y_train,x_test, y_test):

    md.fit(x_train, y_train)

    pred = md.predict(x_test)

    #prob = md.predict_proba(x_test)[:,1]

    print( ' ' )

    print ('Accuracy on Train set: ', md.score(x_train, y_train))

    print( 'Accuracy on Test set: ', md.score(x_test, y_test))

    print( ' ')

    print(classification_report(y_test, pred))

    print( ' ')

    print('Confusion Matrix',confmat(pred, y_test))

    

    #roc(prob, y_test)

    return md
# feature selection with the best model from grid search

gb = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 8,n_estimators=1000, max_features = 0.9,min_samples_leaf = 100)

model_gb = model(gb, x_train, y_train,x_test, y_test)
# Use gridsearch to fine tune the parameters

xgb = XGBClassifier()

xgb_params = {'n_estimators':[200,300],'learning_rate':[0.05,0.02], 'max_depth':[4],'min_child_weight':[0],'gamma':[0]}

gridsearch(xgb, xgb_params,x_train, x_test, y_train, y_test,5)
# feature selection with the best model from grid search

xgb = XGBClassifier(

 learning_rate =0.2,

 n_estimators=200,

 max_depth=7,

 eta=0.025,

 min_child_weight=10,

 gamma=0.65,

 max_delta_step=1.8,

 subsample=0.9,

 colsample_bytree=0.4,

 objective= 'binary:logistic',

 nthread=1,

 scale_pos_weight=1,

 thresh = 0.5,

 reg_lambda=1,

 booster='gbtree',

 n_jobs=1,

 num_boost_round=700,

 silent=True,

 seed=30)

model_xgb = model(xgb, x_train, y_train,x_test, y_test)
LOANDATA4['PREDICTED_STATUS']=np.int_(model_gb.predict(LOANDATA4.drop(['CreditScoreGroup','Due_Principal','Due_interest',

                            'Due_Fee','Paid_Fee','Due_TAX','Paid_Tax'],axis = 1)))

LOANDATA4.index.names =['Customer_Id']
LOANDATA4[20:30]
LOANDATA4.loc[851947]
#LOANDATA4['PREDICTED_STATUS'].to_csv("LOANDATA4_Predict.csv")