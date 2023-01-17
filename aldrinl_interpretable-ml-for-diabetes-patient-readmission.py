# data pre processing

import numpy as np

import pandas as pd



# data visuzlization

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



# warnings

import warnings

warnings.filterwarnings('ignore')



# modeling

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score,roc_auc_score



# model explainers

import lime

from lime.lime_tabular import LimeTabularExplainer

import eli5

from eli5.sklearn import PermutationImportance

import shap

from shap import TreeExplainer,KernelExplainer,LinearExplainer

shap.initjs()



#due to problems encounted with the kaggle environment.. I move the installation of skater in the later part of this kernel
df = pd.read_csv('/kaggle/input/diabetic_data.csv')

df.replace('?',np.nan,inplace=True)

df.head()
plt.figure(figsize=(10,8))

sns.heatmap(df.isnull().T, cbar=False);
plt.figure(figsize=(10,8))

missing = pd.DataFrame({'column':df.columns ,'na_percent':df.isnull().sum()/len(df)*100})

missing.sort_values('na_percent',inplace=True)

plt.barh(missing['column'],width=missing['na_percent']);
#dropping columns with high NA percentage

df.drop(['weight','medical_specialty','payer_code'],axis=1,inplace=True)

# dropping columns related to IDs

df.drop(['encounter_id','patient_nbr','admission_type_id',

         'discharge_disposition_id','admission_source_id'],axis=1,inplace=True)

#removing invalid/unknown entries for gender

df=df[df['gender']!='Unknown/Invalid']

# dropping rows with NAs.

df.dropna(inplace=True)
diag_cols = ['diag_1','diag_2','diag_3']

for col in diag_cols:

    df[col] = df[col].str.replace('E','-')

    df[col] = df[col].str.replace('V','-')

    condition = df[col].str.contains('250')

    df.loc[condition,col] = '250'



df[diag_cols] = df[diag_cols].astype(float)



# diagnosis grouping

for col in diag_cols:

    df['temp']=np.nan

    

    condition = df[col]==250

    df.loc[condition,'temp']='Diabetes'

    

    condition = (df[col]>=390) & (df[col]<=458) | (df[col]==785)

    df.loc[condition,'temp']='Circulatory'

    

    condition = (df[col]>=460) & (df[col]<=519) | (df[col]==786)

    df.loc[condition,'temp']='Respiratory'

    

    condition = (df[col]>=520) & (df[col]<=579) | (df[col]==787)

    df.loc[condition,'temp']='Digestive'

    

    condition = (df[col]>=580) & (df[col]<=629) | (df[col]==788)

    df.loc[condition,'temp']='Genitourinary'

    

    condition = (df[col]>=800) & (df[col]<=999)

    df.loc[condition,'temp']='Injury'

    

    condition = (df[col]>=710) & (df[col]<=739)

    df.loc[condition,'temp']='Muscoloskeletal'

    

    condition = (df[col]>=140) & (df[col]<=239)

    df.loc[condition,'temp']='Neoplasms'

    

    condition = df[col]==0

    df.loc[condition,col]='?'

    df['temp']=df['temp'].fillna('Others')

    condition = df['temp']=='0'

    df.loc[condition,'temp']=np.nan

    df[col]=df['temp']

    df.drop('temp',axis=1,inplace=True)



df.dropna(inplace=True)



df['age'] = df['age'].str[1:].str.split('-',expand=True)[0]

df['age'] = df['age'].astype(int)

max_glu_serum_dict = {'None':0,

                      'Norm':100,

                      '>200':200,

                      '>300':300

                     }

df['max_glu_serum'] = df['max_glu_serum'].replace(max_glu_serum_dict)



A1Cresult_dict = {'None':0,

                  'Norm':5,

                  '>7':7,

                  '>8':8

                 }

df['A1Cresult'] = df['A1Cresult'].replace(A1Cresult_dict)



change_dict = {'No':-1,

               'Ch':1

              }

df['change'] = df['change'].replace(change_dict)



diabetesMed_dict = {'No':-1,

                    'Yes':1

                   }

df['diabetesMed'] = df['diabetesMed'].replace(diabetesMed_dict)



d24_feature_dict = {'Up':10,

                    'Down':-10,

                    'Steady':0,

                    'No':-20

                   }

d24_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',

 'glimepiride','acetohexamide','glipizide','glyburide',

 'tolbutamide','pioglitazone','rosiglitazone','acarbose',

 'miglitol','troglitazone','tolazamide','examide',

 'citoglipton','insulin','glyburide-metformin','glipizide-metformin',

 'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

for col in d24_cols:

    df[col] = df[col].replace(d24_feature_dict)



condition = df['readmitted']!='NO'

df['readmitted'] = np.where(condition,1,0)



df.head()
cat_cols = list(df.select_dtypes('object').columns)

class_dict = {}

for col in cat_cols:

    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col])], axis=1)

df.head()
Xs = df.drop('readmitted',axis=1)

y = df['readmitted']

X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.20,random_state=0)

X_train.shape,X_test.shape
%%time

ML_models = {}

model_index = ['LR','RF','NN']

model_sklearn = [LogisticRegression(solver='liblinear',random_state=0),

                 RandomForestClassifier(n_estimators=100,random_state=0),

                 MLPClassifier([100]*5,early_stopping=True,learning_rate='adaptive',random_state=0)]

model_summary = []

for name,model in zip(model_index,model_sklearn):

    ML_models[name] = model.fit(X_train,y_train)

    preds = model.predict(X_test)

    model_summary.append([name,f1_score(y_test,preds,average='weighted'),accuracy_score(y_test,preds),

                          roc_auc_score(y_test,model.predict_proba(X_test)[:,1])])

print(ML_models)
model_summary = pd.DataFrame(model_summary,columns=['Name','F1_score','Accuracy','AUC_ROC'])

model_summary = model_summary.reset_index()

display(model_summary)
g=sns.regplot(data=model_summary, x="index", y="AUC_ROC", fit_reg=False,

               marker="o", color="black", scatter_kws={'s':500})



for i in range(0,model_summary.shape[0]):

     g.text(model_summary.loc[i,'index'], model_summary.loc[i,'AUC_ROC']+0.02, model_summary.loc[i,'Name'], 

            horizontalalignment='center',verticalalignment='top', size='large', color='black')
test_row = pd.DataFrame(X_test.loc[101706,:]).T

test_row
#initialization of a explainer from LIME

explainer = LimeTabularExplainer(X_train.values,

                                 mode='classification',

                                 feature_names=X_train.columns,

                                 class_names=['Readmitted','Not Readmitted'])
exp = explainer.explain_instance(test_row.values[0],

                                 ML_models['LR'].predict_proba,

                                 num_features=X_train.shape[1])

exp.show_in_notebook(show_table=True)
exp = explainer.explain_instance(test_row.values[0],

                                 ML_models['RF'].predict_proba,

                                 num_features=X_train.shape[1])

exp.show_in_notebook(show_table=True)
exp = explainer.explain_instance(test_row.values[0],

                                 ML_models['NN'].predict_proba,

                                 num_features=X_train.shape[1])

exp.show_in_notebook(show_table=True)
eli5.show_weights(ML_models['LR'], feature_names = list(X_test.columns),top=None)
eli5.show_prediction(ML_models['LR'], test_row.values[0],feature_names=list(X_test.columns),top=None)
exp = PermutationImportance(ML_models['LR'],

                            random_state = 0).fit(X_test, y_test)

eli5.show_weights(exp,feature_names=list(X_test.columns),top=None)
eli5.show_weights(ML_models['RF'],feature_names=list(X_test.columns),top=None)
eli5.show_prediction(ML_models['RF'], test_row.values[0],feature_names=list(X_test.columns),top=None)
exp = PermutationImportance(ML_models['RF'],

                            random_state = 0).fit(X_test, y_test)

eli5.show_weights(exp,feature_names=list(X_test.columns),top=None)
eli5.show_weights(ML_models['NN'])
explainer = LinearExplainer(ML_models['LR'], X_train, feature_dependence="independent")

shap_values = explainer.shap_values(test_row.values)

shap.force_plot(explainer.expected_value,

                shap_values,

                test_row.values,

                feature_names=X_test.columns)
shap_values = explainer.shap_values(X_test.head(250).values)

shap.force_plot(explainer.expected_value,

                shap_values,

                X_test.head(250).values,

                feature_names=X_test.columns)
shap_values = explainer.shap_values(X_test.values)

spplot = shap.summary_plot(shap_values, X_test.values, feature_names=X_test.columns)
top4_cols = ['number_inpatient','number_diagnoses','diabetesMed','number_emergency']

for col in top4_cols:

    shap.dependence_plot(col, shap_values, X_test)
explainer = TreeExplainer(ML_models['RF'])

shap_values = explainer.shap_values(test_row.values)

shap.force_plot(explainer.expected_value[1],

                shap_values[1],

                test_row.values,

                feature_names=X_test.columns)
X_train_kmeans = shap.kmeans(X_train, 10)

explainer = KernelExplainer(ML_models['NN'].predict_proba,X_train_kmeans)

shap_values = explainer.shap_values(test_row.values)

shap.force_plot(explainer.expected_value[1],

                shap_values[1],

                test_row.values,

                feature_names=X_test.columns)