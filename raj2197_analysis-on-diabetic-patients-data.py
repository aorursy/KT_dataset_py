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



import matplotlib.pyplot as plt #Visualization Library

import seaborn as sns #Visualization Library
patient_details = pd.read_excel('../input/Paitent_details.xlsx')

patient_details.head()
patient_details.info()
print(patient_details.race.value_counts())

print('Missing values in race :',(2273/101766)*100)

patient_details.race.value_counts().plot(kind='bar')
patient_details.race.replace('?',np.nan,inplace=True) # Replacing all '?' with null values

patient_details.race.fillna(patient_details.race.mode()[0],inplace=True) # Replacing all null values with mode

patient_details.isnull().any() # Checking if any null values are present in the data.

print('Since data is collected from US assuming that 2% missing values are caucasian race people and imputing it with caucasian')

patient_details.race.value_counts()
print(patient_details.gender.value_counts())

patient_details.gender.value_counts().plot(kind='bar')
print(patient_details.age.value_counts())

patient_details.age.value_counts().plot(kind='bar')
print(patient_details.weight.value_counts())

print('Missing Values in weight:',(98569/101766)*100)
Diagnosis_session = pd.read_excel('../input/Diagnosis_session.xlsx')

Diagnosis_session.head()
Diagnosis_session.info()
admission_details = pd.read_excel('../input/admission_details.xlsx')

admission_details.head()
admission_details.info()
print(admission_details.admission_type_id.value_counts())

admission_details.admission_type_id.value_counts().plot(kind='bar')
print(admission_details.discharge_disposition_id.value_counts())
print(admission_details.discharge_disposition_id.value_counts().plot(kind='bar'))
print(admission_details.payer_code.value_counts())

print('Missing Values in payer code :',(40256/101766)*100)
admission_details.payer_code.value_counts().plot(kind='bar')
print(admission_details.medical_specialty.value_counts())
print('Missing Values in payer code :',(49949/101766)*100)
diabetic_data = pd.read_csv('../input/diabetic_data.csv')

diabetic_data.head()
diabetic_data.info()
treat=diabetic_data.iloc[:,3:26].replace(['Steady','Up','Down','No'],[1,1,1,0])

treat.set_index(diabetic_data.encounter_id,inplace=True)

print(treat.sum(axis=1).value_counts())
print('insulin based treatments ',treat[treat['insulin']==1].sum(axis=1).value_counts())



print('insulin is not used for treating diabeties',treat[treat['insulin']==0].sum(axis=1).value_counts())
i_p=treat[treat['insulin']==1].sum(axis=1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])

i_a=treat[treat['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['NoMed','other','other','other','other','other','other'])

treatments=pd.concat([i_p,i_a])

treatments = pd.DataFrame({'treatments':treatments})
diabetic_data=diabetic_data.join(treatments,how='inner',on='encounter_id')

diabetic_data.head()
#Dropping all drug details because all information been represented in one column which results in redundancy

diabetic_data.drop(['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',

       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',

       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',

       'tolazamide', 'examide', 'citoglipton', 'insulin',

       'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone'],axis=1,inplace=True)
lab_session = pd.read_excel('../input/Lab-session.xlsx')

lab_session.head()
lab_session.info()
data=pd.concat([patient_details,admission_details,Diagnosis_session,lab_session,diabetic_data],axis=1)
data.info()
# Removing all duplicate columns present in dataset after merging.

data_final=data.T.drop_duplicates().T
#From the initial analysis the below three columns have large missing values and diag_1,diag_2,diag_3 are codes which are mostly not useful in our analysis.

data_final.drop(['weight','payer_code','medical_specialty','diag_1','diag_2','diag_3'],axis=1,inplace=True)
data_final.head()
data_final.info()
eff_diab_data=data_final[(data_final['diabetesMed']=='Yes')&(data_final['readmitted']=='NO')&(data_final['treatments'].isin(['insulin','io']))&(~data_final.discharge_disposition_id.isin([11,13,14,19,20]))]

print(eff_diab_data.shape)

eff_diab_data.head()
eff_diab_data.info()
# Total Number of patients who are taking diabetic treatment

data_final[~data_final.discharge_disposition_id.isin([11,13,14,19,20])].treatments.value_counts()
# Diabetic Patients taking treatment and not readmitting into hospital which means treatment is effective.

eff_diab_data.treatments.value_counts()
print('Readmission Rate of Patients given Solo Insulin ',int(abs(1-(14675/29864))*100),'%')

print('Readmission Rate of patients given Insulin combined with other Drugs',int(abs(1-(12145/23100))*100),'%')
eff_diab_data.head()
eff_diab_data.info()
# First step is converting age column to numeric values using label encoding since data is ordinal and values should maintain order.

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

eff_diab_data['age'] = le.fit_transform(eff_diab_data['age'])
# Coverted age categorical data into numerical values

eff_diab_data.age.value_counts().plot(kind='bar')
#Second Step is converting treatment column with numerical values

eff_diab_data.treatments.replace(['insulin','io'],[0,1],inplace=True)
# Conveted treatment column with numerical values 0:Insulin,1: Insulin combination with other drugs

eff_diab_data.treatments.value_counts().plot(kind='bar')
eff_diab_data.head()
eff_diab_data.info()
# Converting numeric colums to interger data type since data is in object data type

eff_diab_data[['encounter_id', 'patient_nbr','admission_type_id', 'discharge_disposition_id', 'admission_source_id','time_in_hospital', 'number_diagnoses', 'num_lab_procedures','num_procedures','num_medications', 'number_outpatient','number_emergency', 'number_inpatient']]=eff_diab_data[['encounter_id', 'patient_nbr','admission_type_id', 'discharge_disposition_id', 'admission_source_id','time_in_hospital', 'number_diagnoses', 'num_lab_procedures','num_procedures','num_medications', 'number_outpatient','number_emergency', 'number_inpatient']].astype('int64')
# Creating new columns using get dummies for nominal data which helps in intrepretability of the model.

data_model=pd.get_dummies(eff_diab_data)

data_model.head()
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
#Introducing some random numbers and checking weather the test is performing correctly or not on given data

data_model['dummyCat'] = np.random.choice([0, 1], size=(len(data_model),), p=[0.5, 0.5])

data_model.dummyCat.value_counts()
#Initialize ChiSquare Class

cT = ChiSquare(data_model)



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

       'readmitted_NO','dummyCat']

for var in testColumns:

    cT.TestIndependence(colX=var,colY="treatments" )
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.svm import SVC
X = data_model.drop(['encounter_id','patient_nbr','age','num_lab_procedures','number_outpatient','number_emergency',

                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',

                      'readmitted_NO','dummyCat','treatments','dummyCat'],axis=1)

y=data_model['treatments']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
y_p=[]

for i in range(y_test.shape[0]):

    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest

y_pred=pd.Series(y_p)

print('BaseLine Accuracy :',accuracy_score(y_test,y_pred))
model_lr = LogisticRegression(solver='liblinear')

model_lr.fit(X_train,y_train)

y_pred_lr = model_lr.predict(X_test)
traning_acc_lr = model_lr.score(X_train,y_train)

testing_acc_lr = accuracy_score(y_test,y_pred_lr)

print('............. LOGISTIC REGRESSION METRICS ...............')

print('Training Accuracy :',traning_acc_lr)

print('Testing Accuracy  :',testing_acc_lr)

print(confusion_matrix(y_test,y_pred_lr))

print(classification_report(y_test,y_pred_lr))
model_knn = KNeighborsClassifier()

model_knn.fit(X_train,y_train)

y_pred_knn = model_knn.predict(X_test)
traning_acc_knn = model_knn.score(X_train,y_train)

testing_acc_knn = accuracy_score(y_test,y_pred_knn)

print('............. K-Nearest Neighbours METRICS ...............')

print('Training Accuracy :',traning_acc_knn)

print('Testing Accuracy  :',testing_acc_knn)

print(confusion_matrix(y_test,y_pred_knn))

print(classification_report(y_test,y_pred_knn))
model_bnb = BernoulliNB()

model_bnb.fit(X_train,y_train)

y_pred_bnb = model_bnb.predict(X_test)
traning_acc_bnb = model_bnb.score(X_train,y_train)

testing_acc_bnb = accuracy_score(y_test,y_pred_bnb)

print('Training Accuracy :',traning_acc_bnb)

print('Testing Accuracy  :',testing_acc_bnb)

print(confusion_matrix(y_test,y_pred_bnb))

print(classification_report(y_test,y_pred_bnb))
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train,y_train)

y_pred_dt = model_dt.predict(X_test)
traning_acc_dt = model_dt.score(X_train,y_train)

testing_acc_dt = accuracy_score(y_test,y_pred_dt)

print('............. Decision Tree METRICS ...............')

print('Training Accuracy :',traning_acc_dt)

print('Testing Accuracy  :',testing_acc_dt)

print(confusion_matrix(y_test,y_pred_dt))

print(classification_report(y_test,y_pred_dt))
model_rf = RandomForestClassifier()

model_rf.fit(X_train,y_train)

y_pred_rf = model_rf.predict(X_test)


traning_acc_rf = model_rf.score(X_train,y_train)

testing_acc_rf = accuracy_score(y_test,y_pred_rf)

print('............. Random Forest METRICS ...............')

print('Training Accuracy :',traning_acc_rf)

print('Testing Accuracy  :',testing_acc_rf)

print(confusion_matrix(y_test,y_pred_rf))

print(classification_report(y_test,y_pred_rf))
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
#Gridsearch CV to find Optimal K value for KNN model

grid = {'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X_train,y_train)

print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))
model_tknn = KNeighborsClassifier(n_neighbors=39)

model_tknn.fit(X_train,y_train)

y_pred_tknn=model_tknn.predict(X_test)
traning_acc_tknn = model_tknn.score(X_train,y_train)

testing_acc_tknn = accuracy_score(y_test,y_pred_tknn)

print('............. Tunned K Nearest Neighbours METRICS ...............')

print('Training Accuracy :',traning_acc_tknn)

print('Testing Accuracy  :',testing_acc_tknn)

print(confusion_matrix(y_test,y_pred_tknn))

print(classification_report(y_test,y_pred_tknn))
# GridSearchCV to find optimal max_depth

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
model_tdt = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=50,min_samples_split=50)

model_tdt.fit(X_train,y_train)

y_pred_tdt=model_tdt.predict(X_test)
training_acc_tdt = model_tdt.score(X_train,y_train)

testing_acc_tdt = accuracy_score(y_test,y_pred_tdt)

print('............. Tunned Decision Tree METRICS ...............')

print('Training Accuracy :',training_acc_tdt)

print('Testing Accuracy  :',testing_acc_tdt)

print(confusion_matrix(y_test,y_pred_tdt))

print(classification_report(y_test,y_pred_tdt))
rfc=RandomForestClassifier(random_state=42)

parameter={'n_estimators':np.arange(1,101)}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=parameter, cv= 3)

CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
model_trf = RandomForestClassifier(n_estimators=59)

model_trf.fit(X_train,y_train) 

y_pred_trf = model_trf.predict(X_test)
training_acc_trf = model_trf.score(X_train,y_train)

testing_acc_trf = accuracy_score(y_test,y_pred_trf)

print('............. Tunned Random Forest METRICS ...............')

print('Training Accuracy :',training_acc_trf)

print('Testing Accuracy  :',testing_acc_trf)

print(confusion_matrix(y_test,y_pred_trf))

print(classification_report(y_test,y_pred_trf))
model_svc = SVC(kernel='linear')

model_svc.fit(X_train,y_train)

y_pred_svc = model_svc.predict(X_test)
training_acc_svc = model_svc.score(X_train,y_train)

testing_acc_svc = accuracy_score(y_test,y_pred_svc)

print('............. Support Vector Classifier METRICS ...............')

print('Training Accuracy :',training_acc_svc)

print('Testing Accuracy  :',testing_acc_svc)

print(confusion_matrix(y_test,y_pred_svc))

print(classification_report(y_test,y_pred_svc))
models_com = pd.DataFrame({'Model':['Logistic Regression','K-Nearest Neighbours','Bernoulli Naive Bayes','Decision Tree','Random Forest','Tunned K-Nearest Neighbours','Tunned Decision Tree','Tunned Random Forest','SVM'],

                           'Training Accuracy':[traning_acc_lr,traning_acc_knn,traning_acc_bnb,traning_acc_dt,traning_acc_rf,traning_acc_tknn,training_acc_tdt,training_acc_trf,training_acc_svc],

                           'Testing Accuracy':[testing_acc_lr,testing_acc_knn,testing_acc_bnb,testing_acc_dt,testing_acc_rf,testing_acc_tknn,testing_acc_tdt,testing_acc_trf,testing_acc_svc]})

models_com.sort_values(by='Testing Accuracy',ascending=False)
plt.figure(figsize=[25,8])

plt.plot(models_com.Model, models_com['Testing Accuracy'], label='Testing Accuracy')

plt.plot(models_com.Model, models_com['Training Accuracy'], label='Training Accuracy')

plt.legend()

plt.title('Model Comparision',fontsize=20)

plt.xlabel('Models',fontsize=30)

plt.ylabel('Accuracy',fontsize=30)

plt.xticks(models_com.Model)

plt.grid()

plt.show()