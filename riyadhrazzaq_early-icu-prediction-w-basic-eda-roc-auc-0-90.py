import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics as mt
from sklearn.preprocessing import LabelEncoder

plt.style.use('seaborn-muted')
pd.options.display.max_columns = None
df = pd.read_excel("../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
df
df.groupby('PATIENT_VISIT_IDENTIFIER',as_index=False).agg({'WINDOW':list, 'ICU':list}).head()
# number of total patient 
n = df['PATIENT_VISIT_IDENTIFIER'].nunique()

# Grouped by each unique patient
dfg = df.groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False)

# Only Final Window's data
indices = np.arange(4,df.shape[0],5)
sdf = df.iloc[indices,:]
fig, ax = plt.subplots(1,1, figsize=(15,10),facecolor='white')
tmp = (df.groupby('PATIENT_VISIT_IDENTIFIER',as_index=False)['ICU'].sum()['ICU'] > 0).sum()
plt.bar([0,5],[tmp, n-tmp],tick_label=['ICU', 'Non-ICU'])
plt.grid(b=True, axis='y')
plt.show()
print(f'Probability of patient in ICU is at {tmp / n}%')
tmpdf = dfg.agg({'AGE_ABOVE65': lambda x: x[0], 'ICU': lambda x: x[4]})
not_icu_less, in_icu_less, in_icu_over, not_icu_over = tmpdf.groupby('AGE_ABOVE65')['ICU'].value_counts().values

fix, ax = plt.subplots(1,1,figsize=(15,10),facecolor='white')
x = [0,0.5,4,4.5]
height = [not_icu_less, in_icu_less, in_icu_over, not_icu_over]
ax.bar(x, height,color=['green','red','red', 'green'])
ax.set_xticks(x)
ax.set_xticklabels(['Not ICU and AGE<65','In ICU and AGE<65','ICU and AGE>65','Not ICU and AGE>65'],rotation=20)
ax.tick_params()
plt.show()
percentiles = sorted(df['AGE_PERCENTIL'].unique().tolist())
percentiles
tmpgrp = sdf.groupby(by=['AGE_PERCENTIL'],squeeze=True)['ICU'].value_counts()

icu = []
not_icu = []
for p in percentiles:
    icu.append(tmpgrp.loc[(p,1)])
    not_icu.append(tmpgrp.loc[(p,0)])
fig, ax = plt.subplots(1,1,figsize=(15,10))
x_icu = np.arange(0,len(percentiles)*3, 3)
ax.bar(x_icu, icu,label='ICU',alpha=0.5,color='red', width=0.5)
ax.bar(x_icu+1, not_icu,label='NOT ICU', alpha=0.5,align='center',color='green',width=0.5)
ax.legend()
ax.set_xticks(x_icu+.5)
ax.set_xticklabels(percentiles)
ax.grid(b=True,axis='y',color='gray',alpha=0.3)
plt.show()
a,b = sdf.groupby('GENDER')['ICU'].sum()
fig, ax = plt.subplots(1,1,facecolor='white',figsize=(15,10))
ax.bar([0,2],[a,b],label=['0','1'])
ax.set_xticks([0,2])
plt.show()
disease_cols = ['DISEASE GROUPING 1','DISEASE GROUPING 2','DISEASE GROUPING 3',
                'DISEASE GROUPING 4','DISEASE GROUPING 5','DISEASE GROUPING 6',
               'HTN','IMMUNOCOMPROMISED']
in_icu = []
notin_icu = []

for i in disease_cols:
    in_icu.append(sdf.groupby('ICU')[i].sum()[1])
    notin_icu.append(sdf.groupby('ICU')[i].sum()[0])

fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.bar(np.arange(0,len(disease_cols)*4,4), in_icu,label='in ICU', color='red',alpha=0.5)
ax.bar(1+np.arange(0,len(disease_cols)*4,4), notin_icu, label='Not in ICU',color='green',alpha=0.5)
ax.legend()
ax.set_xticks(np.arange(0,len(disease_cols)*4,4))
ax.set_xticklabels(disease_cols, rotation=20)
ax.grid(b=True, axis='y', alpha=0.4)
plt.show()
in_icu = sdf.groupby('ICU')['OTHER'].sum()[1]
notin_icu = sdf.groupby('ICU')['OTHER'].sum()[0]

fig, ax = plt.subplots(1,1,figsize=(15,10))
ax.bar([1,2], [in_icu,notin_icu], color='red',alpha=0.5,width=0.2)
ax.set_xticks([1,2])
ax.set_xticklabels(['In ICU', 'Not in ICU'])
ax.grid(b=True, axis='y', alpha=0.4)
plt.show()
indices = np.arange(0,df.shape[0],5)
window1 = df.iloc[indices,:]
window1.head()
fig,ax = plt.subplots(1,1,figsize=(15,10))

ax.hist([window1['ALBUMIN_MEDIAN'],window1['ALBUMIN_MEAN'],window1['ALBUMIN_MAX']],density=True,
        label=['ALBUMIN_MEDIAN','ALBUMIN_MEAN','ALBUMIN_MAX'])
ax.legend()
plt.show()
test_cols = ['ALBUMIN_MEDIAN', 'ALBUMIN_MEAN', 'ALBUMIN_MIN', 'ALBUMIN_MAX',
       'ALBUMIN_DIFF', 'BE_ARTERIAL_MEDIAN', 'BE_ARTERIAL_MEAN',
       'BE_ARTERIAL_MIN', 'BE_ARTERIAL_MAX', 'BE_ARTERIAL_DIFF',
       'BE_VENOUS_MEDIAN', 'BE_VENOUS_MEAN', 'BE_VENOUS_MIN',
       'BE_VENOUS_MAX', 'BE_VENOUS_DIFF', 'BIC_ARTERIAL_MEDIAN',
       'BIC_ARTERIAL_MEAN', 'BIC_ARTERIAL_MIN', 'BIC_ARTERIAL_MAX',
       'BIC_ARTERIAL_DIFF', 'BIC_VENOUS_MEDIAN', 'BIC_VENOUS_MEAN',
       'BIC_VENOUS_MIN', 'BIC_VENOUS_MAX', 'BIC_VENOUS_DIFF',
       'BILLIRUBIN_MEDIAN', 'BILLIRUBIN_MEAN', 'BILLIRUBIN_MIN',
       'BILLIRUBIN_MAX', 'BILLIRUBIN_DIFF', 'BLAST_MEDIAN', 'BLAST_MEAN',
       'BLAST_MIN', 'BLAST_MAX', 'BLAST_DIFF', 'CALCIUM_MEDIAN',
       'CALCIUM_MEAN', 'CALCIUM_MIN', 'CALCIUM_MAX', 'CALCIUM_DIFF',
       'CREATININ_MEDIAN', 'CREATININ_MEAN', 'CREATININ_MIN',
       'CREATININ_MAX', 'CREATININ_DIFF', 'FFA_MEDIAN', 'FFA_MEAN',
       'FFA_MIN', 'FFA_MAX', 'FFA_DIFF', 'GGT_MEDIAN', 'GGT_MEAN',
       'GGT_MIN', 'GGT_MAX', 'GGT_DIFF', 'GLUCOSE_MEDIAN', 'GLUCOSE_MEAN',
       'GLUCOSE_MIN', 'GLUCOSE_MAX', 'GLUCOSE_DIFF', 'HEMATOCRITE_MEDIAN',
       'HEMATOCRITE_MEAN', 'HEMATOCRITE_MIN', 'HEMATOCRITE_MAX',
       'HEMATOCRITE_DIFF', 'HEMOGLOBIN_MEDIAN', 'HEMOGLOBIN_MEAN',
       'HEMOGLOBIN_MIN', 'HEMOGLOBIN_MAX', 'HEMOGLOBIN_DIFF',
       'INR_MEDIAN', 'INR_MEAN', 'INR_MIN', 'INR_MAX', 'INR_DIFF',
       'LACTATE_MEDIAN', 'LACTATE_MEAN', 'LACTATE_MIN', 'LACTATE_MAX',
       'LACTATE_DIFF', 'LEUKOCYTES_MEDIAN', 'LEUKOCYTES_MEAN',
       'LEUKOCYTES_MIN', 'LEUKOCYTES_MAX', 'LEUKOCYTES_DIFF',
       'LINFOCITOS_MEDIAN', 'LINFOCITOS_MEAN', 'LINFOCITOS_MIN',
       'LINFOCITOS_MAX', 'LINFOCITOS_DIFF', 'NEUTROPHILES_MEDIAN',
       'NEUTROPHILES_MEAN', 'NEUTROPHILES_MIN', 'NEUTROPHILES_MAX',
       'NEUTROPHILES_DIFF', 'P02_ARTERIAL_MEDIAN', 'P02_ARTERIAL_MEAN',
       'P02_ARTERIAL_MIN', 'P02_ARTERIAL_MAX', 'P02_ARTERIAL_DIFF',
       'P02_VENOUS_MEDIAN', 'P02_VENOUS_MEAN', 'P02_VENOUS_MIN',
       'P02_VENOUS_MAX', 'P02_VENOUS_DIFF', 'PC02_ARTERIAL_MEDIAN',
       'PC02_ARTERIAL_MEAN', 'PC02_ARTERIAL_MIN', 'PC02_ARTERIAL_MAX',
       'PC02_ARTERIAL_DIFF', 'PC02_VENOUS_MEDIAN', 'PC02_VENOUS_MEAN',
       'PC02_VENOUS_MIN', 'PC02_VENOUS_MAX', 'PC02_VENOUS_DIFF',
       'PCR_MEDIAN', 'PCR_MEAN', 'PCR_MIN', 'PCR_MAX', 'PCR_DIFF',
       'PH_ARTERIAL_MEDIAN', 'PH_ARTERIAL_MEAN', 'PH_ARTERIAL_MIN',
       'PH_ARTERIAL_MAX', 'PH_ARTERIAL_DIFF', 'PH_VENOUS_MEDIAN',
       'PH_VENOUS_MEAN', 'PH_VENOUS_MIN', 'PH_VENOUS_MAX',
       'PH_VENOUS_DIFF', 'PLATELETS_MEDIAN', 'PLATELETS_MEAN',
       'PLATELETS_MIN', 'PLATELETS_MAX', 'PLATELETS_DIFF',
       'POTASSIUM_MEDIAN', 'POTASSIUM_MEAN', 'POTASSIUM_MIN',
       'POTASSIUM_MAX', 'POTASSIUM_DIFF', 'SAT02_ARTERIAL_MEDIAN',
       'SAT02_ARTERIAL_MEAN', 'SAT02_ARTERIAL_MIN', 'SAT02_ARTERIAL_MAX',
       'SAT02_ARTERIAL_DIFF', 'SAT02_VENOUS_MEDIAN', 'SAT02_VENOUS_MEAN',
       'SAT02_VENOUS_MIN', 'SAT02_VENOUS_MAX', 'SAT02_VENOUS_DIFF',
       'SODIUM_MEDIAN', 'SODIUM_MEAN', 'SODIUM_MIN', 'SODIUM_MAX',
       'SODIUM_DIFF', 'TGO_MEDIAN', 'TGO_MEAN', 'TGO_MIN', 'TGO_MAX',
       'TGO_DIFF', 'TGP_MEDIAN', 'TGP_MEAN', 'TGP_MIN', 'TGP_MAX',
       'TGP_DIFF', 'TTPA_MEDIAN', 'TTPA_MEAN', 'TTPA_MIN', 'TTPA_MAX',
       'TTPA_DIFF', 'UREA_MEDIAN', 'UREA_MEAN', 'UREA_MIN', 'UREA_MAX',
       'UREA_DIFF', 'DIMER_MEDIAN', 'DIMER_MEAN', 'DIMER_MIN',
       'DIMER_MAX', 'DIMER_DIFF', 'BLOODPRESSURE_DIASTOLIC_MEAN',
       'BLOODPRESSURE_SISTOLIC_MEAN', 'HEART_RATE_MEAN',
       'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN',
       'OXYGEN_SATURATION_MEAN', 'BLOODPRESSURE_DIASTOLIC_MEDIAN',
       'BLOODPRESSURE_SISTOLIC_MEDIAN', 'HEART_RATE_MEDIAN',
       'RESPIRATORY_RATE_MEDIAN', 'TEMPERATURE_MEDIAN',
       'OXYGEN_SATURATION_MEDIAN', 'BLOODPRESSURE_DIASTOLIC_MIN',
       'BLOODPRESSURE_SISTOLIC_MIN', 'HEART_RATE_MIN',
       'RESPIRATORY_RATE_MIN', 'TEMPERATURE_MIN', 'OXYGEN_SATURATION_MIN',
       'BLOODPRESSURE_DIASTOLIC_MAX', 'BLOODPRESSURE_SISTOLIC_MAX',
       'HEART_RATE_MAX', 'RESPIRATORY_RATE_MAX', 'TEMPERATURE_MAX',
       'OXYGEN_SATURATION_MAX', 'BLOODPRESSURE_DIASTOLIC_DIFF',
       'BLOODPRESSURE_SISTOLIC_DIFF', 'HEART_RATE_DIFF',
       'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF',
       'OXYGEN_SATURATION_DIFF', 'BLOODPRESSURE_DIASTOLIC_DIFF_REL',
       'BLOODPRESSURE_SISTOLIC_DIFF_REL', 'HEART_RATE_DIFF_REL',
       'RESPIRATORY_RATE_DIFF_REL', 'TEMPERATURE_DIFF_REL',
       'OXYGEN_SATURATION_DIFF_REL']
indices = []
for i in range(df.shape[0]):
    if df.loc[i,'WINDOW'] != '6-12' and df.loc[i,'WINDOW'] != 'ABOVE_12':
        indices.append(i)
# del stat
stat = {}
def build(series,col):
    step = 3
    records = series.values[indices]
    if 'MEDIAN' in k:
        stat[col+"_STATUS"] = []
    else:
        stat[col] = []
        
    for i in range(0,len(indices),step):
        tmp_records = records[i:i+step]
#         print(f"i begins {i} \n", tmp_records)
        first = None
        last = None
        
        for j in range(0,step):
            if pd.notnull(tmp_records[j]):
                first=tmp_records[j]
                break
                
        for j in range(step-1,-1,-1):
            if pd.notnull(tmp_records[j]):
                last=tmp_records[j]
                break
        if pd.isnull(first) and pd.isnull(last):
            flag=0
        elif first > last:
            flag = -1 # decrease
        elif first == last:
            flag = 0 # same
        else:
            flag = 1 # increase
        if 'MEDIAN' in k:
            stat[col+"_STATUS"].append(flag)
        else:
            stat[col] = last
#         print(first, last)
for k in test_cols:
    build(df[k],k)
sdf.head()
exp1 = sdf.copy(deep=True)
exp1.isnull().sum()
exp1.fillna(exp1.mean(), inplace=True)
idx = exp1.dtypes=='object'
cols = exp1.dtypes[idx]
print(len(cols),cols)
le = LabelEncoder()
exp1['AGE_PERCENTIL'] = le.fit_transform(exp1['AGE_PERCENTIL'])
exp1['AGE_PERCENTIL']
input_cols = [c for c in sdf.columns if c!='ICU' and c!='WINDOW']
output_cols = 'ICU'
X_train, X_test, y_train, y_test = tts(exp1.loc[:,input_cols],exp1.loc[:,output_cols])
print(X_train.shape, y_train.shape,'\n', X_test.shape, y_test.shape)
def confusion(ytrue,yhat):
    tp = 0 
    fp = 0 
    tn = 0 
    fn = 0 
    
    for i in range(len(ytrue)):
        if ytrue[i] == yhat[i] and yhat[i]==0: # tn
            tn+=1
        elif ytrue[i] == yhat[i] and yhat[i]==1: # tp
            tp+=1
        elif ytrue[i] != yhat[i] and yhat[i]==0: # fn
            fn+=1
        elif ytrue[i] != yhat[i] and yhat[i]==1: # fp
            fp+=1
        
    return [tn,fp,fn,tp]

def confusion_matrix(ytrue,yhat):
    """
    Ensure: |TN FP|
            |     | 
            |FN TP|
    """
    tn,fp,fn,tp = confusion(ytrue,yhat)
        
    return np.array([[tn,fp],[fn,tp]])
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)
confusion_matrix(y_test.values,y_hat)
mt.roc_auc_score(y_test, y_hat)
lda = LinearDiscriminantAnalysis(solver='lsqr')
lda.fit(X_train, y_train)
yhat = lda.predict(X_test)
mt.roc_auc_score(y_test, yhat)