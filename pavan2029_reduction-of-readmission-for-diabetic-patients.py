import pandas as pd
import numpy as np
import os
data =pd.read_csv("../input/diabetic_data.csv")
data.keys()
data.shape
data.info()
data.describe()
for column in data.columns:
    if data[column].dtype == object:
        print(column,data[column][data[column]== '?'].count())    
data.drop(['encounter_id', 'patient_nbr', 'weight','medical_specialty', 'payer_code','admission_source_id' ], axis=1, inplace= True)
data['gender'].value_counts()
data = data[data.gender != 'Unknown/Invalid']
data = data[(data.discharge_disposition_id != 11) & (data.discharge_disposition_id != 19) & (data.discharge_disposition_id != 20) & (data.discharge_disposition_id != 21) & (data.discharge_disposition_id != 7) ]
data = data[(data.diag_1 != '?') | (data.diag_2 != '?') | (data.diag_3 != '?')]
data.drop(['chlorpropamide','acetohexamide', 'tolbutamide', 'rosiglitazone', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'glyburide-metformin', 'glipizide-metformin','glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone'],axis=1, inplace= True)
data.drop(['diag_1','diag_2','diag_3'], axis=1, inplace= True)
data.race.value_counts()
data["Race"]= data["race"].map(lambda x:'NA' if x=='?' else x)
data.drop(['race'], axis=1, inplace= True)
def get_fn(row):
    if row['admission_type_id']==1 or row['admission_type_id']==2 or row['admission_type_id']==7 :
        return "Non ELective"
    elif row['admission_type_id']==3 or row['admission_type_id']==4:
        return "Elective"
    else :
        return "NA"
data['admission_type']= data.apply(get_fn,axis=1)
data.drop(['admission_type_id'], axis=1, inplace= True)
data.readmitted.value_counts()
def fn(x):
    if x =='NO' or x=='>30':
        return 0
    else :
        return 1
data['readmit']= data['readmitted'].map(fn)
data.drop(['readmitted'], axis=1, inplace= True)
data['A1Cresult'].value_counts()
def fun(z):
    if z =='None' or z=='Norm':
        return 1
    else :
        return 0
data['A1C']= data['A1Cresult'].map(fun)
data.drop(['A1Cresult'], axis=1, inplace= True)
def gt_ag(a):
    if a =='[0-10)' or a=='[10-20)' or a=='[20-30)':
        return 'young'
    elif a =='[30-40)' or a=='[40-50)' or a=='[50-60)':
        return 'mid'
    else:
        return'old'
data['Age']= data['age'].map(gt_ag)
data.drop(['age'], axis=1, inplace= True)
data['max_glu_serum'].value_counts()
data['max_glu_serum']=data['max_glu_serum'].replace('None',0)
data['max_glu_serum']=data['max_glu_serum'].replace('Norm',0)
data['max_glu_serum']=data['max_glu_serum'].replace('>200',1)
data['max_glu_serum']=data['max_glu_serum'].replace('>300',1)
def dp_id(a):
    if a ==6 or a==8 or a==9 or a==13 or a==1:
        return 'Discharged Home'
    elif a==18 or a ==25 or a==26 :
        return 'NA'
    else:
        return'Discharged/Transferred'
data['discharge']= data['discharge_disposition_id'].map(dp_id)
data.drop(['discharge_disposition_id'], axis=1, inplace= True)
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='readmit',data=data, palette='hls')
plt.savefig('admit-readmit')
plt.show()
count_0 =len(data[data['readmit']==0])
count_1 = len(data['readmit'])-count_0
prctg_0 = count_0/len(data['readmit'])
prctg_1 = count_1/len(data['readmit'])
print("percentage of readmission", prctg_1*100)
print("percentage of no readmission", prctg_0*100)
pd.crosstab(data.diabetesMed,data.readmit).plot(kind='bar')
plt.title('diabetesMed Vs readmit')
plt.xlabel('diabetesMed')
plt.ylabel('count of readmits')
plt.savefig('diabetesMed vs readmit')
pd.crosstab(data.Age,data.readmit).plot(kind='line')
plt.title('Age Vs readmit')
plt.xlabel('Age')
plt.ylabel('count of readmits')
plt.savefig('Age vs readmit')
pd.crosstab(data.gender,data.readmit).plot(kind='bar')
plt.title('gender Vs readmit')
plt.xlabel('gender')
plt.ylabel('count of readmits')
plt.savefig('gender vs readmit')
dummy_metformin = pd.get_dummies(data['metformin'], prefix='metformin')
data= data.join(dummy_metformin.drop("metformin_No", axis=1))
data.drop(['metformin'], axis=1, inplace= True)
dummy_repaglinide = pd.get_dummies(data['repaglinide'], prefix='repaglinide')
data= data.join(dummy_repaglinide.drop("repaglinide_No", axis=1))
data.drop(['repaglinide'], axis=1, inplace= True)
dummy_insulin = pd.get_dummies(data['insulin'], prefix='insulin')
data= data.join(dummy_insulin.drop("insulin_No", axis=1))
data.drop(['insulin'], axis=1, inplace= True)
dummy_nateglinide = pd.get_dummies(data['nateglinide'], prefix='nateglinide')
data= data.join(dummy_nateglinide.drop("nateglinide_No", axis=1))
data.drop(['nateglinide'], axis=1, inplace= True)
dummy_glimepiride = pd.get_dummies(data['glimepiride'], prefix='glimepiride')
data= data.join(dummy_glimepiride.drop("glimepiride_No", axis=1))
data.drop(['glimepiride'], axis=1, inplace= True)
dummy_glipizide = pd.get_dummies(data['glipizide'], prefix='glipizide')
data= data.join(dummy_glipizide.drop("glipizide_No", axis=1))
data.drop(['glipizide'], axis=1, inplace= True)
dummy_glyburide = pd.get_dummies(data['glyburide'], prefix='glyburide')
data= data.join(dummy_glyburide.drop("glyburide_No", axis=1))
data.drop(['glyburide'], axis=1, inplace= True)
dummy_pioglitazone = pd.get_dummies(data['pioglitazone'], prefix='pioglitazone')
data= data.join(dummy_pioglitazone.drop("pioglitazone_No", axis=1))
data.drop(['pioglitazone'], axis=1, inplace= True)
dummy_acarbose = pd.get_dummies(data['acarbose'], prefix='acarbose')
data= data.join(dummy_acarbose.drop("acarbose_No", axis=1))
data.drop(['acarbose'], axis=1, inplace= True)
dummy_gender = pd.get_dummies(data['gender'], prefix='gender')
data= data.join(dummy_gender.drop("gender_Female", axis=1))
data.drop(['gender'], axis=1, inplace= True)
dummy_admission = pd.get_dummies(data['admission_type'], prefix='admission')
data= data.join(dummy_admission.drop("admission_NA", axis=1))
data.drop(['admission_type'], axis=1, inplace= True)
dummy_change = pd.get_dummies(data['change'], prefix='change')
data= data.join(dummy_change.drop("change_Ch", axis=1))
data.drop(['change'], axis=1, inplace= True)
dummy_Age = pd.get_dummies(data['Age'], prefix='Age')
data= data.join(dummy_Age.drop("Age_mid", axis=1))
data.drop(['Age'], axis=1, inplace= True)
dummy_diabetesMed = pd.get_dummies(data['diabetesMed'], prefix='diabetesMed')
data= data.join(dummy_diabetesMed.drop("diabetesMed_No", axis=1))
data.drop(['diabetesMed'], axis=1, inplace= True)
dummy_race = pd.get_dummies(data['Race'], prefix='Race')
data= data.join(dummy_race.drop("Race_Other", axis=1))
data.drop(['Race'], axis=1, inplace= True)
data.discharge.value_counts()
dummy_discharge = pd.get_dummies(data['discharge'], prefix='discharge')
data= data.join(dummy_discharge.drop("discharge_NA", axis=1))
data.drop(['discharge'], axis=1, inplace= True)
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
X_train, X_test, Y_train, Y_test = train_test_split(data.drop('readmit', axis=1), data['readmit'], test_size=0.2, random_state=12)
X_train = sm.add_constant(X_train)
X_test.shape
X_train.shape
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['glyburide_Up','glyburide_Down','glyburide_Steady'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['acarbose_Down','acarbose_Steady','acarbose_Up'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['nateglinide_Down','nateglinide_Steady','nateglinide_Up'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['number_outpatient'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['Age_young'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['glimepiride_Up','glimepiride_Down'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['pioglitazone_Up','pioglitazone_Steady','pioglitazone_Down'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['glipizide_Steady','glipizide_Up'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['Race_AfricanAmerican','Race_Asian','Race_Caucasian','Race_Hispanic','Race_NA'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['insulin_Steady','insulin_Up'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['change_No'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['admission_Elective','admission_Non ELective'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['repaglinide_Down','repaglinide_Up','repaglinide_Steady'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['metformin_Down'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['max_glu_serum'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['gender_Male'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['num_procedures'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_train.drop(['time_in_hospital'], axis=1, inplace=True)
model= sm.GLM(Y_train, X_train, family=sm.families.Binomial()).fit()
print(model.summary2())
X_test = sm.add_constant(X_test[['num_lab_procedures','num_medications','number_emergency','number_inpatient','number_diagnoses','A1C','metformin_Steady','metformin_Up','insulin_Down','glimepiride_Steady','glipizide_Down','Age_old','diabetesMed_Yes','discharge_Discharged Home','discharge_Discharged/Transferred']])
probabilities = model.predict(X_test)
probabilities.head()
predicted_classes = probabilities.map(lambda x: 1 if x > 0.1 else 0)
predicted_classes.head()
accuracy = sum(predicted_classes == Y_test) / len(Y_test)
accuracy
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
%matplotlib inline
confusion_mat = confusion_matrix(Y_test, predicted_classes)
confusion_df = pd.DataFrame(confusion_mat, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
confusion_df
_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
probs = model.predict(X_test)
auc = roc_auc_score(Y_test, probs)
print('AUC',auc)
fpr, tpr, threshold = roc_curve(Y_test, probs)
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC')
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, predicted_classes)
recall_score(Y_test, predicted_classes)
f1_score(Y_test, predicted_classes)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]
optimal_threshold
new_predictions = np.where(probs>optimal_threshold, 1, 0)
new_confusion_mat = confusion_matrix(Y_test, new_predictions)
new_confusion_df = pd.DataFrame(new_confusion_mat, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
new_confusion_df
_=sns.heatmap(new_confusion_df, cmap='coolwarm', annot=True)
accuracy = sum(new_predictions == Y_test) / len(Y_test)
accuracy
precision_score(Y_test, new_predictions)
recall_score(Y_test, new_predictions)
f1_score(Y_test, new_predictions)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=10,max_depth=25,min_samples_split=3)
rf_model.fit(X_train, Y_train)
predictions = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
precision_score(Y_test, predictions)
confusion_mat = confusion_matrix(Y_test, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Actual 0','Actual 1'],\
                            columns=['Predicted 0','Predicted 1'])
confusion_df
feature_list = X_train.columns
features = rf_model.feature_importances_
most_imp = pd.DataFrame([a for a in zip(feature_list,features)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp)), most_imp.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp)), most_imp.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features')
plt.savefig('Most imp')
plt.show()
