import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#importing diabetes data

data = pd.read_csv('C:/Users/makam/Desktop/Capstone Project/diabetic_data.csv')

data1 = pd.read_csv('C:/Users/makam/Desktop/Capstone Project/diabetic_data.csv')

#Now we label encode the values for the drug columns.

drugs = ['metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',

       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',

       'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',

       'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone']

drug_d = pd.DataFrame()

for x in drugs:

    del data1[x]

    drug_d[x]=data[x]

    mapping_dict={x:{'No':0,'Down':1,'Steady':1,'Up':1}}

    drug_d.replace(mapping_dict,inplace=True)

    

#One hot encoding other columns in diabetes data

cont = pd.get_dummies(data1['max_glu_serum'],prefix='max_glu_serum',drop_first=False)

#Adding the results to the master dataframe

data1 = pd.concat([data1,cont],axis=1)

del data1['max_glu_serum']



cont = pd.get_dummies(data1['A1Cresult'],prefix='A1Cresult',drop_first=False)

#Adding the results to the master dataframe

data1 = pd.concat([data1,cont],axis=1)

del data1['A1Cresult']



cont = pd.get_dummies(data1['change'],prefix='change',drop_first=False)

#Adding the results to the master dataframe

data1 = pd.concat([data1,cont],axis=1)

del data1['change']





data1['diabetesMed'].replace({'No':0,'Yes':1},inplace=True)



cont = pd.get_dummies(data1['readmitted'],prefix='readmitted',drop_first=False)

#Adding the results to the master dataframe

data1 = pd.concat([data1,cont],axis=1)

del data1['readmitted']





#We will now check which encounters are having combination of insulin & which are having solo insulin treatment.

drug_d['encounter_id']=data['encounter_id']

ids = data['encounter_id']

ids1 = pd.DataFrame(ids)

ids1['key'] = ids1.index

drug_dt=drug_d.T

insulin_data = drug_d['insulin']

drug_dt.drop(['insulin'],inplace=True)

cols = drug_dt.columns.values.tolist()

drug_dt.drop(['encounter_id'],inplace=True)

no_combo=[]

combo=[]

for x in cols:

    if(drug_dt[x].sum()==0):

        no_combo.append(x)

    else:

        combo.append(x)

combo1 = pd.DataFrame(combo)

combo1.rename(columns={0:'key'},inplace=True)

combo2 = pd.merge(combo1,ids1,on='key',how='inner')

del combo2['key']

no_combo1 = pd.DataFrame(no_combo)

no_combo1.rename(columns={0:'key'},inplace=True)

no_combo2 = pd.merge(no_combo1,ids1,on='key',how='inner')

del no_combo2['key']

ins_data=pd.DataFrame(insulin_data,columns=['insulin'])

ins_data['encounter_id']=data['encounter_id']

combo3 = pd.merge(combo2,ins_data,on='encounter_id',how='inner')

no_combo3 = pd.merge(no_combo2,ins_data,on='encounter_id',how='inner')

no_diabetes = no_combo3[no_combo3['insulin']==0]

type1 = no_combo3[no_combo3['insulin']!=0]

type2 = combo3[combo3['insulin']==0]

t1t2 = combo3[combo3['insulin']!=0]





no_diabetes['treatment']=0

type1['treatment']=1

type2['treatment']=0

t1t2['treatment']=1





diabetes = pd.merge(drug_d,data1,on='encounter_id',how='inner')







#Reading pateint data & one hot encoding the features.

patient_data = pd.read_excel('C:/Users/makam/Desktop/Capstone Project/Paitent_details.xlsx')

df1 = pd.merge(patient_data,data,on='encounter_id',how='inner')



cont = pd.get_dummies(patient_data['race'],prefix='race',drop_first=False)

#Adding the results to the master dataframe

patient_data = pd.concat([patient_data,cont],axis=1)

del patient_data['race']



cont = pd.get_dummies(patient_data['gender'],prefix='gender',drop_first=False)

#Adding the results to the master dataframe

patient_data = pd.concat([patient_data,cont],axis=1)

del patient_data['gender']





patient_data['age'].replace({'[70-80)':7,'[60-70)':6,'[50-60)':5,'[80-90)':8,'[40-50)':4,'[30-40)':3,'[90-100)':9,'[20-30)':2,'[10-20)':1,'[0-10)':0},inplace=True)

del patient_data['weight']

final_diabetes = pd.merge(patient_data,diabetes,on='encounter_id',how='inner')

ndc = pd.merge(no_diabetes['encounter_id'],final_diabetes,on='encounter_id',how='inner')

t1 = pd.merge(type1['encounter_id'],final_diabetes,on='encounter_id',how='inner')

t2= pd.merge(type2['encounter_id'],final_diabetes,on='encounter_id',how='inner')

t12 = pd.merge(t1t2['encounter_id'],final_diabetes,on='encounter_id',how='inner')



ndc['treatment']=0

t1['treatment']=1

t2['treatment']=0

t12['treatment']=1



NDC = pd.merge(no_diabetes[['encounter_id','treatment']],df1,on='encounter_id',how='inner')

T1 = pd.merge(type1[['encounter_id','treatment']],df1,on='encounter_id',how='inner')

T2= pd.merge(type2[['encounter_id','treatment']],df1,on='encounter_id',how='inner')

T12 = pd.merge(t1t2[['encounter_id','treatment']],df1,on='encounter_id',how='inner')



diabetes = pd.concat([t1,t2,t12])

org_diabetes = pd.concat([NDC,T1,T2,T12])

#We can now delete all the drug columns as we have captured the data in treatment column.

for x in drugs:

    del diabetes[x]
d = pd.read_excel('C:/Users/makam/Desktop/Capstone Project/admission_details.xlsx')

del d['patient_nbr']

#We will do label encoding for the columns.

d['payer_code'].replace({'?':'Others'},inplace=True)

d['medical_specialty'].replace({'?':'Others'},inplace=True)



cont = pd.get_dummies(d['payer_code'],prefix='payer_code',drop_first=False)

#Adding the results to the master dataframe

d = pd.concat([d,cont],axis=1)

del d['payer_code']



del d['medical_specialty']



cont = pd.get_dummies(d['admission_type_id'],prefix='admission_type_id',drop_first=False)

#Adding the results to the master dataframe

d = pd.concat([d,cont],axis=1)

del d['admission_type_id']



cont = pd.get_dummies(d['discharge_disposition_id'],prefix='discharge_disposition_id',drop_first=False)

#Adding the results to the master dataframe

d = pd.concat([d,cont],axis=1)

del d['discharge_disposition_id']



cont = pd.get_dummies(d['admission_source_id'],prefix='admission_source_id',drop_first=False)

#Adding the results to the master dataframe

d = pd.concat([d,cont],axis=1)

del d['admission_source_id']





diabetes1 = pd.merge(diabetes,d,on='encounter_id',how='inner')
len(diabetes1.columns.values)
d1 = pd.read_excel('C:/Users/makam/Desktop/Capstone Project/Lab-session.xlsx')

diabetes2 = pd.merge(diabetes1,d1,on='encounter_id',how='inner')


dig_d = pd.read_excel('C:/Users/makam/Desktop/Capstone Project/Diagnosis_session.xlsx')

'''

er = pd.concat([dig_d['diag_1'],dig_d['diag_2'],dig_d['diag_3']],axis=0).tolist()

y = 0

for x in er:

    er[y] = str(x)

    y = y+1

er1 = pd.DataFrame(er)

l = LabelEncoder()

l.fit(er1[0])

y = 0

l1 = dig_d['diag_1'].tolist()

l2 = dig_d['diag_2'].tolist()

l3 = dig_d['diag_3'].tolist()

for x in l1:

    l1[y] = str(x)

    y = y+1

y = 0

for x in l2:

    l2[y] = str(x)

    y = y+1

y = 0

for x in l3:

    l3[y] = str(x)

    y = y+1

m1 = l.transform(l1).tolist()

m2 = l.transform(l2).tolist()

m3 = l.transform(l3).tolist()

dig_d['diag_1']=m1

dig_d['diag_2']=m2

dig_d['diag_3']=m3

del dig_d['patient_nbr']

'''

df = pd.merge(diabetes2,dig_d[['encounter_id','number_diagnoses']],on='encounter_id',how='inner')
df.columns.values
df.shape
# Distribution of Readmission 

sns.countplot(df['treatment']).set_title('Distribution of Treatment')
fig = plt.figure(figsize=(15,5))

sns.countplot(x= df['age'], hue = df['treatment']).set_title('Age of Patient VS. Treatment')
fig = plt.figure(figsize=(8,8))

sns.countplot(x = org_diabetes['race'], hue = org_diabetes['treatment'])
fig = plt.figure(figsize=(8,8))

sns.barplot(x = df['treatment'], y = (df['num_medications'])).set_title("Number of medication used VS. Treatment")
fig = plt.figure(figsize=(8,8))

sns.countplot(x = org_diabetes['max_glu_serum'], hue = org_diabetes['treatment']).set_title('Glucose test serum test result VS. Treatment')

                                                                                            

                                                                                            

                                                                                            

                                                                                            

                                                                                                                                                                                                             
fig = plt.figure(figsize=(8,8))

sns.countplot(x= org_diabetes['A1Cresult'], hue = org_diabetes['treatment']).set_title('A1C test result VS. Treatment')
fig = plt.figure(figsize=(8,8))

sns.countplot(x= org_diabetes['treatment'], hue = org_diabetes['readmitted']).set_title('readmission VS. Treatment')
fig = plt.figure(figsize=(15,6),)

ax=sns.kdeplot(df.loc[(df['treatment'] == 0),'num_lab_procedures'] , color='b',shade=True,label='No Insulin')

ax=sns.kdeplot(df.loc[(df['treatment'] == 1),'num_lab_procedures'] , color='r',shade=True, label='Insulin Present')

ax.set(xlabel='Number of lab procedure', ylabel='Frequency')

plt.title('Number of lab procedure VS. Treatment')
fig = plt.figure(figsize=(13,7),)

ax=sns.kdeplot(df.loc[(df['treatment'] == 0),'time_in_hospital'] , color='b',shade=True,label='No Insulin')

ax=sns.kdeplot(df.loc[(df['treatment'] == 1),'time_in_hospital'] , color='r',shade=True, label='Insulin present')

ax.set(xlabel='Time in Hospital', ylabel='Frequency')

plt.title('Time in Hospital VS. Treatment')
fig = plt.figure(figsize=(15,5),)

ax=sns.kdeplot(df.loc[(df['treatment'] == 0),'number_diagnoses'] , color='b',shade=True,label='No Insulin')

ax=sns.kdeplot(df.loc[(df['treatment'] == 1),'number_diagnoses'] , color='r',shade=True, label='Insulin present')

ax.set(xlabel='number_diagnoses', ylabel='Frequency')

plt.title('number_diagnoses VS. Treatment')
fig = plt.figure(figsize=(15,5),)

ax=sns.kdeplot(df.loc[(df['treatment'] == 0),'num_medications'] , color='b',shade=True,label='No Insulin')

ax=sns.kdeplot(df.loc[(df['treatment'] == 1),'num_medications'] , color='r',shade=True, label='Insulin present')

ax.set(xlabel='number_diagnoses', ylabel='Frequency')

plt.title('num_medications VS. Treatment')
fig = plt.figure(figsize=(15,5),)

ax=sns.kdeplot(df.loc[(df['treatment'] == 0),'number_diagnoses'] , color='b',shade=True,label='No Insulin')

ax=sns.kdeplot(df.loc[(df['treatment'] == 1),'number_diagnoses'] , color='r',shade=True, label='Insulin present')

ax.set(xlabel='number_diagnoses', ylabel='Frequency')

plt.title('number_diagnoses VS. Treatment')
df.columns.values
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
cT = ChiSquare(df)

#Feature Selection

testColumns = diabetes2.columns.values.tolist()

testColumns.remove('treatment')
for var in testColumns:

    cT.TestIndependence(colX=var,colY="treatment" )
cT1 = ChiSquare(org_diabetes)

testcolumns1 = org_diabetes.columns.values.tolist()

testcolumns1.remove('treatment')

for var in testcolumns1:

    cT1.TestIndependence(colX=var,colY="treatment" )
list = ['encounter_id', 

       'race_Hispanic', 

       'gender_Unknown/Invalid', 'diabetesMed', 

       'A1Cresult_>7',  'readmitted_>30',

        'payer_code_CP', 

       'payer_code_FR',  'payer_code_OT',

        'payer_code_WC',

        'admission_type_id_4',

        'discharge_disposition_id_2',

       

       'discharge_disposition_id_9', 'discharge_disposition_id_10'

       , 'discharge_disposition_id_12',

        'discharge_disposition_id_16',

       'discharge_disposition_id_17',

       'discharge_disposition_id_19', 'discharge_disposition_id_20',

       'discharge_disposition_id_22', 

       'discharge_disposition_id_24', 

       'discharge_disposition_id_27', 'discharge_disposition_id_28',

        'admission_source_id_6',

       'admission_source_id_8',

        'admission_source_id_10',

       'admission_source_id_11', 'admission_source_id_13',

       'admission_source_id_14','admission_source_id_22',

       'admission_source_id_25']
len(list)
for x in list:

    del df[x]       
len(df.columns.values)
df.shape
df.info()
x = df.drop('treatment',axis=1)

y = df['treatment']

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=100)

from sklearn.ensemble import RandomForestClassifier

model5 = RandomForestClassifier(random_state=100,n_estimators=85,max_features=7)

print(model5)

from sklearn.ensemble import BaggingClassifier

model1 = BaggingClassifier(bootstrap_features=True,n_estimators=100,random_state=100)

print(model1)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=100)

model5.fit(xtrain,ytrain)

y_pred_test= model5.predict(xtest)

from sklearn.metrics import *

print('Testing set accuracy')

print(accuracy_score(ytest, y_pred_test))

print(confusion_matrix(ytest,y_pred_test))

y_pred_train = model5.predict(xtrain)

print('Training set accuracy')

print(accuracy_score(ytrain, y_pred_train))

print(confusion_matrix(ytrain,y_pred_train))
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.model_selection import GridSearchCV
#Logistic Regression



m1=KNeighborsClassifier()

m1.fit(xtrain,ytrain)

y_pred_lr=m1.predict(xtest)

Train_Score_lr = m1.score(xtrain,ytrain)

Test_Score_lr = accuracy_score(ytest,y_pred_lr)





print('Training Accuracy is:',Train_Score_lr)

print('Testing Accuracy is:',Test_Score_lr)

print(classification_report(ytest,y_pred_lr))
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 3



# parameters to build the model on

parameters = {'max_depth': np.arange(1,5,1),

    'min_samples_leaf': np.arange(1,15,1),

    'min_samples_split': np.arange(2,5,1),

    'criterion': ["entropy", "gini"]}



# instantiate the model

dtree = DecisionTreeClassifier(random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(xtrain, ytrain)
tree.best_params_
dtree = DecisionTreeClassifier(random_state = 100,criterion= 'gini',max_depth= 4,min_samples_leaf= 14,min_samples_split= 2)

dtree.fit(xtrain,ytrain)

y_pred_test = dtree.predict(xtest)

print('Testing set accuracy')

print(accuracy_score(ytest, y_pred_test))

print(confusion_matrix(ytest,y_pred_test))

y_pred_train = model5.predict(xtrain)

print('Training set accuracy')

print(accuracy_score(ytrain, y_pred_train))

print(confusion_matrix(ytrain,y_pred_train))

#Gridsearch CV to find Optimal K value for KNN model

grid = {'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=3)

knn_cv.fit(xtrain,ytrain)

 



print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))
d1= KNeighborsClassifier(n_neighbors= 47)

d1.fit(xtrain,ytrain)

y_pred_test = d1.predict(xtest)

print('Testing set accuracy')

print(accuracy_score(ytest, y_pred_test))

print(confusion_matrix(ytest,y_pred_test))

y_pred_train = d1.predict(xtrain)

print('Training set accuracy')

print(accuracy_score(ytrain, y_pred_train))

print(confusion_matrix(ytrain,y_pred_train))