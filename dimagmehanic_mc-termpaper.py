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
### import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as sns



from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression, Perceptron

from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, precision_recall_curve, average_precision_score 

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier

from sklearn.neural_network import MLPClassifier



%matplotlib inline

plt.style.use('ggplot')
def read_data(tp = "Train", N = 1542865627584):

    target = pd.read_csv("/kaggle/input/healthcare-provider-fraud-detection-analysis/{}-{}.csv".format(tp.title(), N))

    pt = pd.read_csv("/kaggle/input/healthcare-provider-fraud-detection-analysis/{}_Beneficiarydata-{}.csv".format(tp.title(), N))

    in_pt = pd.read_csv("/kaggle/input/healthcare-provider-fraud-detection-analysis/{}_Inpatientdata-{}.csv".format(tp.title(), N))

    out_pt = pd.read_csv("/kaggle/input/healthcare-provider-fraud-detection-analysis/{}_Outpatientdata-{}.csv".format(tp.title(), N))

    return (in_pt, out_pt, pt, target)
### Load Train data

in_pt, out_pt, asl, target = read_data()
asl = asl.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,

                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 

                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 

                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2, 'Gender': 2 }, 0)

asl = asl.replace({'RenalDiseaseIndicator': 'Y'}, 1).astype({'RenalDiseaseIndicator': 'int64'})
print(asl.shape)

asl.head()
print(target.shape)

target.head()
plt.title("Potential Fraud Test distribution")

target.groupby( ["PotentialFraud"] ).Provider.count().plot(kind = "bar", figsize = (10,6))

plt.xlabel('Status')

plt.ylabel('Count')

plt.show()
print(in_pt.shape)

in_pt.head()
print(out_pt.shape)

out_pt.head()
asl['WhetherDead']= 0

asl.loc[asl.DOD.notna(),'WhetherDead'] = 1
target["target"] = np.where(target.PotentialFraud == "Yes", 1, 0) 
MediCare = pd.merge(in_pt, out_pt, left_on = [ x for x in out_pt.columns if x in in_pt.columns], right_on = [ x for x in out_pt.columns if x in in_pt.columns], how = 'outer')

MediCare.shape
data = pd.merge(MediCare, asl,left_on='BeneID',right_on='BeneID',how='inner')

data.shape
### Check Physicians columns for stange records and value length.

def len_check(data , l):

    S = dict()

    for i in data.columns:

         S[i] = [x for x in data.loc[ np.any(data[[i]].notnull().to_numpy(), axis = 1)][i].unique() if (len(str(x)) < l | len(str(x)) > l ) ]

    

    print(S)



len_check(data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']], len('PHY388358'))  
def uniq(a):

    return np.array([len(set([i for i in x[~pd.isnull(x)]])) for x in a.values])
### Create new variable and drop 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'

data['NumPhysicians'] = uniq(data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']]) 

data = data.drop(['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'], axis = 1)
ClmProcedure_vars = ['ClmProcedureCode_{}'.format(x) for x in range(1,7)]

### Create new variable 

data['NumProc'] = data[ClmProcedure_vars].notnull().to_numpy().sum(axis = 1)
keep = ['BeneID', 'ClaimID', 'ClmAdmitDiagnosisCode', 'NumProc' ] + ClmProcedure_vars

### Checking if procedures is unique

print(data[keep].loc[data['NumProc'] != uniq( data[ClmProcedure_vars])])



data = data.drop(ClmProcedure_vars, axis = 1)
ClmDiagnosisCode_vars =['ClmAdmitDiagnosisCode'] + ['ClmDiagnosisCode_{}'.format(x) for x in range(1, 11)]



### Create new variable 

data['NumClaims'] = data[ClmDiagnosisCode_vars].notnull().to_numpy().sum(axis = 1)
keep = ['BeneID', 'ClaimID', 'ClmAdmitDiagnosisCode', 'NumClaims'] + ClmDiagnosisCode_vars



### Create new variable 

data['NumClaims'] = data[ClmDiagnosisCode_vars].notnull().to_numpy().sum(axis = 1)



print(data[keep].loc[data['NumClaims'] != uniq( data[ClmDiagnosisCode_vars])].head())

### if checking result of unique claims is not missing, we are going to add number of unique claims.
data['NumUniqueClaims'] = uniq(data[ClmDiagnosisCode_vars])



data['ExtraClm'] = data['NumClaims'] - data['NumUniqueClaims']



data = data.drop(ClmDiagnosisCode_vars, axis = 1)

data = data.drop(['NumClaims'], axis = 1)
### 

data['AdmissionDt'] = pd.to_datetime(data['AdmissionDt'] , format = '%Y-%m-%d')

data['DischargeDt'] = pd.to_datetime(data['DischargeDt'],format = '%Y-%m-%d')



data['ClaimStartDt'] = pd.to_datetime(data['ClaimStartDt'] , format = '%Y-%m-%d')

data['ClaimEndDt'] = pd.to_datetime(data['ClaimEndDt'],format = '%Y-%m-%d')



data['DOB'] = pd.to_datetime(data['DOB'] , format = '%Y-%m-%d')

data['DOD'] = pd.to_datetime(data['DOD'],format = '%Y-%m-%d')



### Number of hospitalization days

data['AdmissionDays'] = ((data['DischargeDt'] - data['AdmissionDt']).dt.days) + 1

### Number of claim days 

data['ClaimDays'] = ((data['ClaimEndDt'] - data['ClaimStartDt']).dt.days) + 1



data['Age'] = round(((data['ClaimStartDt'] - data['DOB']).dt.days + 1)/365.25)
data['Hospt'] = np.where(data.DiagnosisGroupCode.notnull(), 1, 0)

data = data.drop(['DiagnosisGroupCode'], axis = 1)
### Check if there were any actions after death. 

data['DeadActions'] = np.where(np.any(np.array([ data[x] > data['DOD'] for x in ['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt']]), axis = 0), 1, 0)



print(data.loc[data['DeadActions'] > 0])



### If there is no actions after death date, we will drop this variable. 

data = data.drop(['AdmissionDt', 'DeadActions', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt', 'DOD', 'DOB'], axis = 1)
data.describe(exclude = ['object'])
data.shape
data.isnull().sum()
## Fill missing results using 0

data = data.fillna(0).copy()

data.columns
### Sum all results

df1 = data.groupby(['Provider'], as_index = False)[['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator', 

                                                  'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',

                                                  'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 

                                                  'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 

                                                  'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 

                                                  'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',

                                                  'ChronicCond_stroke', 'WhetherDead', 'NumPhysicians', 

                                                  'NumProc','NumUniqueClaims', 'ExtraClm', 'AdmissionDays',

                                                  'ClaimDays', 'Hospt']].sum()

### Count number of records

df2 = data[['BeneID', 'ClaimID']].groupby(data['Provider']).nunique().reset_index()

### Calculate mean

df3 = data.groupby(['Provider'], as_index = False)[['NoOfMonths_PartACov', 'NoOfMonths_PartBCov',

                                                    'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',

                                                    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age']].mean()

### Combine all together

df = df2.merge(df1, on='Provider', how='left').merge(df3, on='Provider', how='left')

print(df.shape, target.shape)
df1 = df.merge(target, on='Provider', how='left').drop(['Provider', 'target'], axis = 1)

df2 = df.merge(target, on='Provider', how='left').drop(['Provider', 'PotentialFraud'], axis = 1)

print(df.shape, target.shape)
g = sns.pairplot(df1, hue = 'PotentialFraud', markers="+")

g.fig.suptitle('Plot pairwise relationships in a dataset')

plt.show()
plt.figure(figsize=(20, 20))

plt.title('Correlation heatmap')

sns.heatmap(df2.corr())

plt.show()
countFraud = target.target.value_counts()

print('No:', countFraud[0])

print('Yes:', countFraud[1])

print('Proportion:', round(countFraud[1] / countFraud[0], 2))

### We should keep in mind that we are using unbalanced data
### Only Train dataset is labeled that why we split it to two sets train and validation

X_train, X_val, y_train, y_val = train_test_split(df.drop(['Provider'], axis = 1), target.target.to_numpy(), test_size=0.25, random_state=1)



cols = X_train.columns



X_train = StandardScaler().fit_transform(X_train)

X_val = StandardScaler().fit_transform(X_val)



print("Train obs: {}; Features Number: {}".format(X_train.shape[0], X_train.shape[1]))

print("Validation obs: {};".format(X_val.shape[0]))
## write Master Learn class which we are going to use for our analysis

class MasterL:

    

    def __init__(self, model, #### model is a method which we are going to use for detecting FRAUDS. For example: sklearn.svm

                 X= X_train, y= y_train, test= X_val, ### data

                 **kvars  #### additional key parameters for model

                ):

        self.clf = model( **kvars)

        self.methodname = model.__name__

        self.X_train = X

        self.y_train = y

        self.X_test = test

        self.fit(self.X_train, self.y_train)

        self.predicted = self.predict(test)

        

    def fit (self, X, y):

        self.clf.fit(X, y)

    

    def predict(self, x):

        return self.clf.predict(x)

       

    def get_score(self, y = y_val, roc = True, params = False):

        accuracy = accuracy_score(self.predicted, y)

        if params:

            print(self.clf.get_params())

        print(self.methodname+ " metrics:\n")

        print(" Accuracy Score: %.2f%%" % (accuracy * 100.0))

        print(" Confusion matrix:", "\n",confusion_matrix(y_true=y, y_pred=self.predicted))

        print( 'Classification report:\n', classification_report(y, self.predicted))

        if roc:

            print(" ROC Score: %.2f%%" % (roc_auc_score(y, self.clf.predict_proba(self.X_test)[:,1])))

        

    def plot_curves(self, y = y_val):   

        plt.figure(figsize=(17, 5))

        plt.subplot(131)

        # Plot the recall precision tradeoff        

        self.plot_pr_curve(y)

        plt.subplot(132)        

        self.plot_lern_curve(accuracy_score)     

        plt.subplot(133)

        self.plot_lern_curve(roc_auc_score)

        plt.show()

        

    def plot_pr_curve(self, y = y_val):

        

        plt.subplot(122)

        # Calculate average precision and the PR curve

        average_precision = average_precision_score(y, self.predicted)



        # Obtain precision and recall 

        precision, recall, _ = precision_recall_curve(y, self.clf.predict_proba(self.X_test)[:,1])

        

        plt.step(recall, precision, where='post')

        plt.xlabel('Recall')

        plt.ylabel('Precision')

        plt.ylim([0.0, 1.05])

        plt.xlim([0.0, 1.05])

        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision))

    

    def plot_lern_curve(self, metrics):

        plt.title(self.methodname + " Learning Curves")

        plt.xlabel("Training examples")

        plt.ylabel("{}".format(' '.join(metrics.__name__.split('_')).title()))

        

        train_sizes, train_scores, test_scores = learning_curve(self.clf, self.X_train, self.y_train, n_jobs=-1, 

                                                                cv = ShuffleSplit(n_splits=5, test_size=.25 , random_state = 5), 

                                                                train_sizes=np.linspace(0.5, 1.0, 10), scoring = make_scorer(metrics))

        train_scores_mean = np.mean(train_scores, axis=1) 

        test_scores_mean = np.mean(test_scores, axis=1) 

        #plt.grid()



        plt.plot(train_sizes,  train_scores_mean, 'o-', color="r", label="Training score")

        plt.plot(train_sizes,  test_scores_mean, 'o-', color="g", label="Cross-validation score")

        

        plt.legend(loc="best")

    

    def plot_roc_curve(self, y = y_val, models = None, fig = None):

        fig = plt.figure(figsize=(15, 7))

        ax = fig.add_subplot(121)

        

        self.roc_curves(ax, y, models)

        

        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

        

        ax.set_xlabel('False Positive Rate')

        ax.set_ylabel('True Positive Rate')

        plt.title('Receiver Operating Characteristic (ROC) Curve')

        

        plt.legend(loc="best")

        

        #if fig != None:

            #plt.savefig( fig, bbox_inches = 'tight')

       

    def roc_curves(self, p, y, M):

        if M == None:

            fpr, tpr, thresholds = roc_curve(y, self.clf.predict_proba(self.X_test)[:,1] )

            p.plot(fpr, tpr,  label=self.methodname )

        else:

            fpr, tpr, thresholds = roc_curve(y, self.clf.predict_proba(self.X_test)[:,1] )

            p.plot(fpr, tpr,  label=self.methodname )

            for i in M:

                fpr, tpr, thresholds = roc_curve(y, i.clf.predict_proba(i.X_test)[:,1] )

                p.plot(fpr, tpr,  label=i.methodname )



#### Function for serching best parameters which is fiting the model and shows best results for specified method.               

def grid(method, parameters):

    

    grid_1 = GridSearchCV(method, parameters, scoring = make_scorer(accuracy_score), cv=5, n_jobs = -1)

    grid_2 = GridSearchCV(method, parameters, scoring = make_scorer(roc_auc_score), cv=5, n_jobs = -1)

    

    grid_1.fit(X_train, y_train)

    print('Best parameters using accuracy score:')

    print(grid_1.best_params_)



    grid_2.fit(X_train, y_train)

    print('Best parameters usin ROC accuracy score:')

    print(grid_2.best_params_)
### Logistic regression 

### Balanced Weight and Scaled data

ML1 = MasterL(LogisticRegression, 

              penalty= 'l1',

              solver= 'liblinear', class_weight='balanced', random_state = 5 , C = 0.001)

# Get your performance metrics

ML1.get_score()
ML1.plot_roc_curve()

ML1.plot_pr_curve()
# SVM(scaled data)

ML2 = MasterL(SVC, 

              gamma = 'auto', probability = True, random_state= 5, class_weight= 'balanced', C=1 )



# Get your performance metrics

ML2.get_score()
ML2.plot_roc_curve(models = [ML1])

ML2.plot_pr_curve()
### Random Forest Clasifier

# Continue fitting the model and obtain predictions



ML3 = MasterL(RandomForestClassifier, 

              n_estimators = 60, n_jobs = -1, random_state = 5, class_weight = 'balanced_subsample', 

              min_samples_split = 0.25

             )

 

# Get your performance metrics

ML3.get_score() 
ML3.plot_roc_curve(models = [ML1, ML2])

ML3.plot_pr_curve()
features = ML3.clf.feature_importances_

Features_score = pd.DataFrame(np.array([cols, features]).T, columns = ["VarName", "Importamce"]).sort_values(by=["Importamce"], ascending=False)



Features_score.head()
### Generate ensemble

ML4 = MasterL(VotingClassifier, 

              estimators=[ ('lr', ML1.clf), ("rf", ML3.clf)], voting='soft', n_jobs = -1

             )

 

# Get your performance metrics

ML4.get_score()
ML4.methodname = "log-reg + RandomForestCl"

ML4.plot_roc_curve(models = [ML1, ML2, ML3])

ML4.plot_pr_curve()
### Multy Layer Perceptron

ML5 = MasterL( MLPClassifier, 

              activation = 'logistic',

              hidden_layer_sizes = (1, 3),random_state = 5, max_iter= 1000 )

# Get your performance metrics 

ML5.get_score()
ML5.plot_roc_curve(models = [ML1, ML2, ML3, ML4])

ML5.plot_pr_curve()