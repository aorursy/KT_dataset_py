# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
admission = pd.read_excel('../input/admission_details.xlsx')
admission.head(2)
admission.info()
admission.isnull().values.any()
admission.payer_code.value_counts()
admission.medical_specialty.value_counts()
admission = admission.drop(['medical_specialty','payer_code'],axis = 1)
admission.time_in_hospital.value_counts()
admission.columns
admission.head(2)
admission.shape
diabetic = pd.read_csv('../input/diabetic_data.csv')
diabetic.columns
diabetic.info()
diabetic.readmitted.value_counts()
diabetic.diabetesMed.value_counts()
diabetic.change.value_counts()
diabetic.isnull().values.any()
Diabetic_transform=diabetic.replace(['No','Steady','Up','Down'],[0,1,1,1])

Diabetic_transform.set_index('encounter_id',inplace=True)

Diabetic_transform.head()
Diabetic_transform.sum(axis=1).value_counts()
#diabetic.set_index('encounter_id',inplace=True)
i1 = Diabetic_transform[Diabetic_transform['insulin']==1].sum(axis = 1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])
i1.value_counts()
i0=Diabetic_transform[Diabetic_transform['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['no med','other','other','other','other','other','other'])
i0.value_counts()
treatments=pd.concat([i1,i0])

treatments = pd.DataFrame({'treatments':treatments})
treatments.head()
diabetic=diabetic.join(treatments,on='encounter_id') #setting index as encounter_id
diabetic.head(2)
diabetic.shape
diabetic.columns
lab_sessions = pd.read_excel('../input/Lab-session.xlsx')
lab_sessions.columns
lab_sessions.info()
lab_sessions.columns
lab_sessions.isnull().values.any()
lab_sessions.shape
patient_details = pd.read_excel('../input/Paitent_details.xlsx')
patient_details.columns
patient_details.isnull().values.any()
patient_details = patient_details.drop(['weight'],axis = 1)
patient_details.race.value_counts()
patient_details['race']=patient_details.race.replace('?',np.nan)

patient_details['race'].fillna(patient_details['race'].mode()[0], inplace=True)

patient_details.race.isnull().sum()
patient_details.shape
print("Admission" , admission.shape)

print("Diabetic" ,diabetic.shape)

print("Lab Sessions",lab_sessions.shape)

print("Patient_details",patient_details.shape)
data = pd.concat([patient_details,admission,lab_sessions,diabetic],axis=1)
data.shape
#data = pd.read_csv('Final_Diabetes_withallrows.csv')
#data[['admission_source_id','time_in_hospital']] = admission[['admission_source_id','time_in_hospital']]
#data[['num_lab_procedures','num_medication','number_outpatient','number_emergency','number_inpatient','num_procedures']] = lab_sessions[['num_lab_procedures','num_medications',

                                                                                                                        #'number_outpatient','number_emergency','number_inpatient','num_procedures']]
data.columns
df1=data.iloc[:,:2]



df1.head()
df2=data.drop(['encounter_id','patient_nbr'],axis=1)



df2.head()
data_final=pd.concat([df1,df2],axis=1)



data_final.shape



data_final.head()


data_diamed_yes=data_final[data_final.diabetesMed=='Yes']

data_diamed_yes.shape
data_readmit_no=data_diamed_yes[data_diamed_yes.readmitted=='NO']

data_readmit_no.shape
data_new=data_readmit_no[~data_readmit_no.discharge_disposition_id.isin([11,13,14,19,20])]

data_new.shape
data_model=data_new[data_new.treatments!='other']
data_model.info()
data_model.shape
data_model.head().T
#data_cat = data_model.select_dtypes(include=['object']).copy()

data_model.treatments.value_counts()
data_model = data_model.drop(['metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',

       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',

       'tolazamide', 'examide', 'citoglipton', 'insulin',

       'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone'],axis = 1)

data_model.shape
data_model.columns
data_model.num_procedures.plot(kind='hist')

plt.xlabel("No.of Lab Procedures")
import seaborn as sns

sns.barplot(data_model.discharge_disposition_id)
data_model.num_medications.plot(kind='hist')

plt.xlabel("No.of Medications")
data_onehot = pd.get_dummies(data_model, columns=['race', 'gender','max_glu_serum', 'A1Cresult', 'change',

       'diabetesMed', 'readmitted'])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_onehot['age']=data_onehot.apply(le.fit_transform)
data_onehot[['discharge_disposition_id','admission_type_id', 'admission_source_id','num_lab_procedures',

       'num_procedures','time_in_hospital',

       'num_medications', 'number_outpatient', 'number_emergency',

       'number_inpatient']]=  data_model[['discharge_disposition_id','admission_type_id','admission_source_id','num_lab_procedures',

       'num_procedures','time_in_hospital',

       'num_medications', 'number_outpatient', 'number_emergency',

       'number_inpatient']]
data_onehot.shape
data_onehot.info()
data_onehot.head()
#data_onehot = data_onehot.drop(['Unnamed: 0'],axis = 1)
data_onehot.shape
data_onehot.columns
import pandas as pd

import numpy as np

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
data_onehot['dummyCat'] = np.random.choice([0, 1], size=(len(data_onehot),), p=[0.5, 0.5])



data_onehot.dummyCat.value_counts()
#Initialize ChiSquare Class

cT = ChiSquare(data_onehot)



#Feature Selection

testColumns = ['encounter_id', 'patient_nbr', 'age', 'admission_type_id',

       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',

       'num_lab_procedures', 'num_procedures', 'num_medications',

       'number_outpatient', 'number_emergency', 'number_inpatient',

        'race_AfricanAmerican', 'race_Asian', 'race_Caucasian',

       'race_Hispanic', 'race_Other', 'gender_Female', 'gender_Male',

       'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_None',

       'max_glu_serum_Norm', 'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_None',

       'A1Cresult_Norm', 'change_Ch', 'change_No', 'diabetesMed_Yes',

       'readmitted_NO']

for var in testColumns:

    cT.TestIndependence(colX=var,colY="treatments" ) 
from scipy.stats import chisquare,chi2_contingency



cat_col = []

chi_pvalue = []

chi_name = []



def chi_sq(i):

    ct = pd.crosstab(data_onehot['treatments'],data_onehot[i])

    chi_pvalue.append(chi2_contingency(ct)[1])

    chi_name.append(i)



for i in testColumns:

    chi_sq(i)



chi_data = pd.DataFrame()

chi_data['Pvalue'] = chi_pvalue

chi_data.index = chi_name



plt.figure(figsize=(11,8))

plt.title('P-Values of Chisquare with ''Treatments'' as Target Categorical Attribute',fontsize=16)

x = chi_data.Pvalue.sort_values().plot(kind='barh')

x.set_xlabel('P-Values',fontsize=15)

x.set_ylabel('Independent Categorical Attributes',fontsize=15)

plt.show()
# Import `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier
# Isolate Data, class labels and column values

X = data_onehot.drop(['treatments'],axis=1)

Y = data_onehot['treatments']

Y=Y.replace(['insulin','io'],[0,1])

names = data_onehot.columns.values
X.head()
Y.head()
Y.shape
# Build the model

rfc = RandomForestClassifier()



# Fit the model

rfc.fit(X, Y)
#Finding the feature importance using Random Forest

feature_imp=pd.DataFrame({'Features':X.columns,'Importance':rfc.feature_importances_})

feature_imp.sort_values(by = 'Importance',ascending=True)
data_onehot.columns
X = data_onehot.drop(['encounter_id','patient_nbr','age','num_lab_procedures','number_outpatient','number_emergency',

                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',

                      'readmitted_NO','dummyCat','treatments'],axis=1)

X.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
y_p=[]

for i in range(y_test.shape[0]):

    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest

len(y_p) 
y_pred=pd.Series(y_p)
accuracy_score(y_test,y_pred)
#Logistic Regression

m1=LogisticRegression()

m1.fit(X_train,y_train)

y_pred_lr=m1.predict(X_test)

Train_Score_lr = m1.score(X_train,y_train)

Test_Score_lr = accuracy_score(y_test,y_pred_lr)
print('Training Accuracy is:',Train_Score_lr)

print('Testing Accuracy is:',Test_Score_lr)

print(classification_report(y_test,y_pred_lr))
from sklearn.metrics import roc_curve, auc

fpr,tpr, _ = roc_curve(y_test,y_pred_lr)

roc_auc_lr = auc(fpr, tpr)



print('Auc for Logistic Regression is:',roc_auc_lr)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
m2 = KNeighborsClassifier()

m2.fit(X_train,y_train)

y_pred_knn = m2.predict(X_test)

Train_Score_knn = m2.score(X_train,y_train)

Test_Score_knn = accuracy_score(y_test,y_pred_knn)
print('Training Accuracy is :',Train_Score_knn)

print('Testing Accuracy is:',Test_Score_knn)

print(classification_report(y_test,y_pred_knn))
fpr,tpr, _ = roc_curve(y_test,y_pred_knn)

roc_auc_knn = auc(fpr, tpr)



print('Auc for KNN is:',roc_auc_knn)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
m3=BernoulliNB()

m3.fit(X_train,y_train)

y_pred_bnb=m3.predict(X_test)

Train_Score_bnb = m3.score(X_train,y_train)

Test_Score_bnb = accuracy_score(y_test,y_pred_bnb)
print('Training Accuracy :',Train_Score_bnb)

print('Testing Accuracy  :',Test_Score_bnb)

print(classification_report(y_test,y_pred_bnb))
fpr,tpr, _ = roc_curve(y_test,y_pred_bnb)

roc_auc_bnb = auc(fpr, tpr)



print('Auc for Bernoulli Naive Bayes is:',roc_auc_bnb)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
m4 = DecisionTreeClassifier()

m4.fit(X_train,y_train)

y_pred_dt=m4.predict(X_test)

Train_Score_dt = m4.score(X_train,y_train)

Test_Score_dt = accuracy_score(y_test,y_pred_dt)
print('Training Accuracy :',Train_Score_dt)

print('Testing Accuracy :',Test_Score_dt)

print(classification_report(y_test,y_pred_dt))
fpr,tpr, _ = roc_curve(y_test,y_pred_dt)

roc_auc_dt = auc(fpr, tpr)



print('Auc for Decision Tree is:',roc_auc_dt)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
m5 = RandomForestClassifier()

m5.fit(X_train,y_train)

y_pred_rf=m5.predict(X_test)

Train_Score_rf = m5.score(X_train,y_train)

Test_Score_rf = accuracy_score(y_test,y_pred_rf)
print('Training Accuracy :',Train_Score_rf)

print('Testing Accuracy :',Test_Score_rf)

print(classification_report(y_test,y_pred_rf))
fpr,tpr, _ = roc_curve(y_test,y_pred_rf)

roc_auc_rf = auc(fpr, tpr)



print('Auc for Random Forest is:',roc_auc_rf)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 3



# parameters to build the model on

parameters = {'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]}



# instantiate the model

dtree = DecisionTreeClassifier(random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
tree.best_params_
m6 = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=100,min_samples_split=50)

m6.fit(X_train,y_train)

y_pred_tdt=m6.predict(X_test)

Train_Score_tdt = m6.score(X_train,y_train)

Test_Score_tdt = accuracy_score(y_test,y_pred_tdt)
print('Training Accuracy :',Train_Score_tdt)

print('Testing Accuracy  :',Test_Score_tdt)

print(classification_report(y_test,y_pred_tdt))

fpr,tpr, _ = roc_curve(y_test,y_pred_tdt)

roc_auc_tdt = auc(fpr, tpr)



print('Auc for Tuned Decision Tree is:',roc_auc_tdt)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
#Gridsearch CV to find Optimal K value for KNN model

grid = {'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X_train,y_train)





print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))
m7 = KNeighborsClassifier(n_neighbors=45)

m7.fit(X_train,y_train)

y_pred_tknn=m7.predict(X_test)

Train_Score_tknn = m7.score(X_train,y_train)

Test_Score_tknn = accuracy_score(y_test,y_pred_tknn)
print('Training Accuracy :',Train_Score_tknn)

print('Testing Accuracy  :',Test_Score_tknn)

print(classification_report(y_test,y_pred_tknn))
fpr,tpr, _ = roc_curve(y_test,y_pred_tknn)

roc_auc_tknn = auc(fpr, tpr)



print('Auc for Tuned KNN is:',roc_auc_tknn)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
parameter={'n_estimators':np.arange(1,101)}

gs = GridSearchCV(m5,parameter,cv=3)

gs.fit(X_train,y_train)

gs.best_params_
m8 = RandomForestClassifier(n_estimators=71)

m8.fit(X_train,y_train) 

y_pred_trf=m8.predict(X_test)

Train_Score_trf = m8.score(X_train,y_train)

Test_Score_trf = accuracy_score(y_test,y_pred_trf)
print('Training Accuracy :',Train_Score_trf)

print('Testing Accuracy  :',Test_Score_trf)

print(classification_report(y_test,y_pred_trf))
fpr,tpr, _ = roc_curve(y_test,y_pred_trf)

roc_auc_trf = auc(fpr, tpr)



print('Auc for Tuned Random Forest is:',roc_auc_trf)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
data_model.treatments.replace(['insulin','io'],[0,1],inplace = True)
data_model.head().T
a = data_model.drop(['age','diabetesMed','readmitted','treatments'],axis=1)

b = data_model.treatments
a.dtypes
cate_features_index = np.where(a.dtypes != int)[0]

xtrain,xtest,ytrain,ytest = train_test_split(a,b,train_size=.70,random_state=2)

from catboost import CatBoostClassifier, Pool,cv

#let us make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
#show the model test acc, but you have to note that the acc is not the cv acc,

#so recommend to use the cv acc to evaluate your model!

print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))

test_score_catboost = accuracy_score(ytest,model.predict(xtest))

print("the train accuracy is :",model.score(xtrain,ytrain))

train_score_catboost = model.score(xtrain,ytrain)
fpr,tpr, _ = roc_curve(ytest,model.predict(xtest))

roc_auc_cb = auc(fpr, tpr)



print('Auc for Cat Boost is:',roc_auc_cb)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
model.predict(xtest)
Model_Scores=pd.DataFrame({'Models':['Logistic Regression','KNN','Bernauli Naives Bayes','Decision Tree','Random Forest','Tuned Decison Tree','Tuned KNN','Tuned Random Forest','Cat Boost'],

             'Training Accuracy':[Train_Score_lr,Train_Score_knn,Train_Score_bnb,Train_Score_dt,Train_Score_rf,Train_Score_tdt,Train_Score_tknn,Train_Score_trf,train_score_catboost],

             'Testing Accuracy':[Test_Score_lr,Test_Score_knn,Test_Score_bnb,Test_Score_dt,Test_Score_rf,Test_Score_tdt,Test_Score_tknn,Test_Score_trf,test_score_catboost],

                'AUC':[roc_auc_lr,roc_auc_knn,roc_auc_bnb,roc_auc_dt,roc_auc_rf,roc_auc_tdt,roc_auc_tknn,roc_auc_trf,roc_auc_cb]})



Model_Scores.sort_values(by=('Testing Accuracy'),ascending=False)
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier

bslr=AdaBoostClassifier(base_estimator=LogisticRegression())

bslr.fit(X_train,y_train)

y_pred_blr=bslr.predict(X_test)

Train_Score_bslr = bslr.score(X_train,y_train)

Test_Score_bslr = accuracy_score(y_test,y_pred_blr)
print('Training Accuracy :',Train_Score_bslr)

print('Testing Accuracy  :',Test_Score_bslr)

print(classification_report(y_test,y_pred_blr))
fpr,tpr, _ = roc_curve(y_test,y_pred_blr)

roc_auc_bslr = auc(fpr, tpr)



print('Auc for Boosted Logistic Regression is:',roc_auc_bslr)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
bglr=BaggingClassifier(base_estimator=LogisticRegression())

bglr.fit(X_train,y_train)
y_pred_bglr=bglr.predict(X_test)

Train_Score_bglr = bglr.score(X_train,y_train)

Test_Score_bglr = accuracy_score(y_test,y_pred_blr)
print('Training Accuracy :',Train_Score_bglr)

print('Testing Accuracy  :',Test_Score_bglr)

print(classification_report(y_test,y_pred_bglr))
fpr,tpr, _ = roc_curve(y_test,y_pred_bglr)

roc_auc_bglr = auc(fpr, tpr)



print('Auc for Bagged Logistic Regression is:',roc_auc_bglr)

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
Model_Scores=pd.DataFrame({'Models':['Logistic Regression','Boosted Logistic Regression','Cat Boost'],

             'Training Accuracy':[Train_Score_lr,Train_Score_bslr,train_score_catboost],

             'Testing Accuracy':[Test_Score_lr,Test_Score_bslr,test_score_catboost],

                    'AUC':[roc_auc_lr,roc_auc_bslr,roc_auc_cb]})



Model_Scores.sort_values(by='Testing Accuracy',ascending=False)
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

stacked = VotingClassifier(estimators=[('Logistic Regression',m1),('KNN',m2),('Naive Bayes',m3),('Decision Tree',m4),

                                      ('RandomForest',m5),('Tuned Decision Tree',m6),('Tuned KNN',m7),

                                       ('Tuned Random Forest',m8),('Boosted Logistic Regression',bslr)],voting='hard')



for model, name in zip([m1,m2,m3,m4,m5,m6,m7,m8,bslr,stacked],['Logistic Regression','KNN','Naive Bayes','Decision Tree','RandomForest',

                                                               'Tuned Decision Tree','Tuned KNN','Tuned Random Forest',

                                                               'Boosted Logistic Regression','stacked']):

    scores=cross_val_score(model,X,Y,cv=5,scoring='accuracy')

    print('Accuarcy: %0.02f (+/- %0.4f)(%s)'%(scores.mean(),scores.var(),name))
from sklearn.ensemble import GradientBoostingClassifier

gbdt=GradientBoostingClassifier(n_estimators=150,random_state=2)

gbdt.fit(X_train,y_train)
y_pred_gbdt=gbdt.predict(X_test)

Train_Score_gbdt = gbdt.score(X_train,y_train)

Test_Score_gbdt = accuracy_score(y_test,y_pred_gbdt)
print('Training Accuracy :',Train_Score_gbdt)

print('Testing Accuracy  :',Test_Score_gbdt)

print(classification_report(y_test,y_pred_gbdt))
models=[]

models.append(('Logistic_Regression',m1))

models.append(('KNN',m2))

models.append(('Bernoulli_NB',m3))

models.append(('Decison Tree',m4))

models.append(('Random Forest',m5))

models.append(('Tuned Decision Tree',m6))

models.append(('Tuned KNN',m7))

models.append(('Tuned Random Forest',m8))

models.append(('Bagged Logistic Regression',bglr))

models.append(('Boosted Logistic regression',bslr))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

results=[]

names=[]

scoring='accuracy'

for name,model in models:

    kfold=KFold(n_splits=5,random_state=2)

    cv_results=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg="%s: %f (%f)"%(name,np.mean(cv_results),cv_results.var())

    print(msg)

#boxplot alogorithm comparision

fig=plt.figure(figsize=(16,8))

fig.suptitle('Algorithm Comparision')

ax=fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.metrics import log_loss



log_loss(y_test, y_pred_lr, eps=1e-15)