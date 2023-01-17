# Importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import preprocessing,model_selection
#reading data into dfs

data = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data.head()
data.info()

print(data.isnull().sum())
columns = ['city','gender','relevent_experience','enrolled_university','education_level','major_discipline','experience','last_new_job','company_size','company_type']

for col in columns:

  cont = pd.crosstab(data['target'],data[col])

  chi_val = stats.chi2_contingency(cont)

  print('p-value for:',col,chi_val[1])
data = data.drop(['major_discipline','company_size'],axis = 'columns')

data_test = data_test.drop(['major_discipline','company_size'],axis = 'columns')

print(data.isnull().sum())
data['experience'].fillna("1",inplace = True)

data_test['experience'].fillna("1",inplace = True)
data['gender'].fillna('Male',inplace = True)

dummies_gender = pd.get_dummies(data['gender'])



data_test['gender'].fillna('Male',inplace = True)

dummies_gender_test = pd.get_dummies(data_test['gender'])

data['relevent_experience'].fillna('Has relevent experience',inplace = True)

dummies_relexp = pd.get_dummies(data['relevent_experience'])



data_test['relevent_experience'].fillna('Has relevent experience',inplace = True)

dummies_relexp_test = pd.get_dummies(data_test['relevent_experience'])



data['enrolled_university'].fillna('no_enrollment',inplace = True)

dummies_enruniv = pd.get_dummies(data['enrolled_university'])



data_test['enrolled_university'].fillna('no_enrollment',inplace = True)

dummies_enruniv_test = pd.get_dummies(data_test['enrolled_university'])
data['education_level'].fillna('Graduate',inplace = True)

dummies_edlevel = pd.get_dummies(data['education_level'])



data_test['education_level'].fillna('Graduate',inplace = True)

dummies_edlevel_test = pd.get_dummies(data_test['education_level'])
data['last_new_job'].fillna(1,inplace = True)

dummies_lastjob = pd.get_dummies(data['last_new_job'],prefix = 'last_')



data_test['last_new_job'].fillna(1,inplace = True)

dummies_lastjob_test = pd.get_dummies(data_test['last_new_job'],prefix = 'last_')
data['experience'].replace({"1":1,"2":1,"3":0,"4":0,"19":0,"20":0,">20":0,"<1":1,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0,"11":0,"12":0,"13":0,"14":0,"15":0,"16":0,"17":0,"18":0},inplace = True)

data_test['experience'].replace({"1":1,"2":1,"3":0,"4":0,"19":0,"20":0,">20":0,"<1":1,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0,"11":0,"12":0,"13":0,"14":0,"15":0,"16":0,"17":0,"18":0},inplace = True)

data['company_type'].fillna('Pvt Ltd',inplace = True)

dummies_ctype = pd.get_dummies(data['company_type'])



data_test['company_type'].fillna('Pvt Ltd',inplace = True)

dummies_ctype_test = pd.get_dummies(data_test['company_type'])
merged = pd.concat([data,dummies_gender,dummies_relexp,dummies_enruniv,dummies_edlevel,dummies_ctype,dummies_lastjob],axis = 'columns')

data = merged.drop(['city','gender','relevent_experience','enrolled_university','education_level','company_type','last_new_job'],axis = 'columns')



merged_test = pd.concat([data_test,dummies_gender_test,dummies_relexp_test,dummies_enruniv_test,dummies_edlevel_test,dummies_ctype_test,dummies_lastjob_test],axis = 'columns')

data_test = merged_test.drop(['city','gender','relevent_experience','enrolled_university','education_level','company_type','last_new_job'],axis = 'columns')



data.head()
data_test.head()


col_list = data.columns[1:]

col_list_test = data_test.columns[1:]



col = col_list.to_list()

col_test = col_list_test.to_list()

print(col,col_test)


scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data[col] = scaler.fit_transform(data[col])

data = pd.DataFrame(data)

col.insert(0,'enrollee_id')

data.columns = col



data_test[col_test] = scaler.fit_transform(data_test[col_test])

data_test = pd.DataFrame(data_test)

col_test.insert(0,'enrollee_id')

data_test.columns = col_test
data.head()
print("target data:",data['target'].value_counts())

from sklearn.utils import resample

df_majority = data[data.target==0]

df_minority = data[data.target==1]

print(df_majority,df_minority)

# Upsample minority class



 
df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=15934,    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

data = pd.concat([df_majority, df_minority_upsampled])

data.info()
from sklearn.model_selection import train_test_split

x = data.drop(['enrollee_id','target','last__1'], axis = 'columns')



y = data['target']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,random_state = 42,stratify = y)



print(x_train.shape, y_train.shape)

print(x_val.shape, y_val.shape)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

lr = LogisticRegression(C=0.001,random_state = 0)

lr.fit(x_train,y_train)



lr_probs = lr.predict_proba(x_val)

lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_val, lr_probs)



lr_auc

lr_fpr, lr_tpr, _ = roc_curve(y_val, lr_probs)

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 50, min_samples_leaf = 1,random_state = 0)

rf.fit(x_train,y_train)



rf_probs = rf.predict_proba(x_val)

rf_probs = rf_probs[:, 1]

rf_auc = roc_auc_score(y_val, rf_probs)



rf_auc
rf_fpr, rf_tpr, _ = roc_curve(y_val, rf_probs)

plt.plot(rf_fpr, rf_tpr, marker='.', label='Logistic')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
#Decision Tree



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



dt_probs = dt.predict_proba(x_val)

dt_probs = dt_probs[:, 1]

dt_auc = roc_auc_score(y_val, dt_probs)



dt_auc
dt_fpr, dt_tpr, _ = roc_curve(y_val, dt_probs)

plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB(binarize=0.55)

bnb.fit(x_train,y_train)

bnb_probs = bnb.predict_proba(x_val)

bnb_probs = bnb_probs[:, 1]

bnb_auc = roc_auc_score(y_val, bnb_probs)



bnb_auc
bnb_fpr, bnb_tpr, _ = roc_curve(y_val, bnb_probs)

plt.plot(bnb_fpr, bnb_tpr, marker='.', label='Bernoulli')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.svm import SVC

svc = SVC(probability=True,tol = 0.01)

svc.fit(x_train,y_train)

svc_probs = svc.predict_proba(x_val)

svc_probs = svc_probs[:, 1]

svc_auc = roc_auc_score(y_val, svc_probs)



svc_auc
svc_fpr, svc_tpr, _ = roc_curve(y_val, svc_probs)

plt.plot(svc_fpr, svc_tpr, marker='.', label='Support Vector')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.neural_network import MLPClassifier

p = MLPClassifier(random_state=42,

              max_iter=100,tol = 0.0001)

p.fit(x_train,y_train)

p_probs = p.predict_proba(x_val)

p_probs = p_probs[:, 1]

p_auc = roc_auc_score(y_val, p_probs)



p_auc
p_fpr, p_tpr, _ = roc_curve(y_val, p_probs)

plt.plot(p_fpr, p_tpr, marker='.', label='Perceptron')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
auc_scores = [lr_auc,rf_auc,bnb_auc,svc_auc,dt_auc,p_auc]

models =['Logistic Regression','Random Forest','Bernoulli NaiveBayes','Support Vector','Decision Tree','MultiLayerPerceptron']

print("=====================ROC_AUC Scores=========================")

for score,model  in zip(auc_scores,models):

  print(model,": ",score)
data_test.head()
x_test = data_test.drop(['enrollee_id','last__1'], axis = 'columns')
pred_lr = lr.predict(x_test)

unique_elements, counts_elements = np.unique(pred_lr, return_counts=True)

print(unique_elements,counts_elements)
pred_rf = rf.predict(x_test)

unique_elements, counts_elements = np.unique(pred_rf, return_counts=True)

print(unique_elements,counts_elements)
pred_dt = dt.predict(x_test)

unique_elements, counts_elements = np.unique(pred_dt, return_counts=True)

print(unique_elements,counts_elements)
pred_svc = svc.predict(x_test)

unique_elements, counts_elements = np.unique(pred_svc, return_counts=True)

print(unique_elements,counts_elements)
pred_bnb = bnb.predict(x_test)

unique_elements, counts_elements = np.unique(pred_bnb, return_counts=True)

print(unique_elements,counts_elements)
pred_p = p.predict(x_test)

unique_elements, counts_elements = np.unique(pred_p, return_counts=True)

print(unique_elements,counts_elements)
auc_scores = [0.5957,0.5162,0.5950,0.6008,0.5168,0.6133]

models =['Logistic Regression','Random Forest','Bernoulli NaiveBayes','Support Vector','Decision Tree','MultiLayerPerceptron']

print("=====================Final Scores=========================")

for score,model  in zip(auc_scores,models):

  print(model,": ",score)