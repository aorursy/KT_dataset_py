import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import xgboost as xgb

import re

import shap

from skopt import dummy_minimize

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
#Read dataframe

df = pd.read_excel("../input/covid19/dataset.xlsx")

df.head()
#Define target 

df['ward_semi_intensive'] = np.where(df['Patient addmited to regular ward (1=yes, 0=no)']

                          + df['Patient addmited to semi-intensive unit (1=yes, 0=no)']

                          + df['Patient addmited to intensive care unit (1=yes, 0=no)']>=1,1,0)



df = df.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)','Patient addmited to intensive care unit (1=yes, 0=no)','Patient ID'], axis = 1)
#Filter columns with percentage of null >= 99%

df_missing = (df.isna().sum()/len(df)).to_frame()

df_missing =df_missing.rename(columns={0: 'percentage of null'})

print(df_missing[df_missing['percentage of null'] >0.99])



#Drop columns

drop = list(df_missing[df_missing['percentage of null'] >0.99].index)

df = df.drop(drop, axis = 1)
#Target variable

sns.countplot(x=df['ward_semi_intensive'], alpha=0.7, data=df)
#Plot numerics features 

numerics = ['float32', 'float64']

df2 = df[df['ward_semi_intensive']==0].select_dtypes(include=numerics)

df3 = df[df['ward_semi_intensive']==1].select_dtypes(include=numerics)

fig, ax = plt.subplots(16,3,figsize=(22, 100))

for i, col in enumerate(df2):

    plt.subplot(16,3,i+1)

    plt.xlabel(col, fontsize=10)

    sns.kdeplot(df2[col].values, bw=0.5,label='Não')

    sns.kdeplot(df3[col].values, bw=0.5,label='Sim')

plt.show() 



#Drop null colunms

drop = ['Myeloblasts']

df = df.drop(drop, axis = 1)
numerics = ['float32', 'float64']

df2 = df[df['ward_semi_intensive']==0].select_dtypes(include=numerics)

df2['ward_semi_intensive'] = 0

df3 = df[df['ward_semi_intensive']==1].select_dtypes(include=numerics)

df3['ward_semi_intensive'] = 1

x = pd.concat([df2,df3])

x = x.groupby('ward_semi_intensive').count()

x['ward_semi_intensive'] = x.index



fig, axes = plt.subplots(round(len(x.columns) / 4), 4, figsize=(20, 50))



for i, ax in enumerate(fig.axes):

    if i < len(x.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=35)

        sns.countplot(x=x.columns[i], data=x,ax=ax,hue='ward_semi_intensive')

fig.tight_layout()

categorics = ['object']

c1 = df[df['ward_semi_intensive']==0].select_dtypes(include=categorics)

c1['ward_semi_intensive'] = 'Não'

c2 = df[df['ward_semi_intensive']==1].select_dtypes(include=categorics)

c2['ward_semi_intensive'] = 'Sim'



c3 = pd.concat([c1,c2])

fig, axes = plt.subplots(round(len(c3.columns) / 4), 4, figsize=(20, 45))



for i, ax in enumerate(fig.axes):

    if i < len(c3.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=35)

        sns.countplot(x=c3.columns[i], alpha=0.7, data=c3, ax=ax,hue="ward_semi_intensive")



fig.tight_layout()



#Drop columns 

drop = ['Adenovirus','Bordetella pertussis', 'Metapneumovirus','Chlamydophila pneumoniae','Inf A H1N1 2009','Urine - Urobilinogen','Urine - Crystals','Urine - Aspect']

df= df.drop(drop,axis=1)
def detect(x):

    df[x] = np.where(df[x] == "detected",1,0)

    return df[x]   



def positive(x):

    df[x] = np.where(df[x] == "positive",1,0)

    return df[x] 



def absent(x):

    df[x] = np.where(df[x] == "absent",1,0)

    return df[x] 



def present(x):

    df[x] = np.where(df[x] == "present",1,0)

    return df[x] 



#Age quantile

df['age_quantile'] = df['Patient age quantile']

df = df.drop('Patient age quantile',axis=1)



#Detected Respiratory Syncytial Virus'

df['Respiratory Syncytial Virus'] = detect('Respiratory Syncytial Virus')



#Detected rhinovirus/Enterovirus

df['Rhinovirus/Enterovirus'] = detect('Rhinovirus/Enterovirus')



#Detected Influenza A or Influenza B

df['Influenza A'] = detect('Influenza A')

df['Influenza B'] = detect('Influenza B')

df['Influenza A, rapid test'] = positive('Influenza A, rapid test')

df['Influenza B, rapid test'] = positive('Influenza B, rapid test')

df['Influenza_A_or_B'] = np.where((df['Influenza A'] + df['Influenza B'] + df['Influenza A, rapid test'] + df['Influenza B, rapid test']) >= 1,1,0)

df = df.drop(['Influenza A','Influenza B','Influenza A, rapid test','Influenza B, rapid test'],axis=1)



#Positive strepto

df['Strepto A'] = positive('Strepto A')



#Detected any Parainfluenza 

df['Parainfluenza 1'] = detect('Parainfluenza 1')

df['Parainfluenza 2'] = detect('Parainfluenza 2')

df['Parainfluenza 3'] = detect('Parainfluenza 3')

df['Parainfluenza 4'] = detect('Parainfluenza 4')

df['Parainflu_detected'] = np.where((df['Parainfluenza 1']+ df['Parainfluenza 2'] + df['Parainfluenza 3']

                                   + df['Parainfluenza 4']) >= 1,1,0)

df = df.drop(['Parainfluenza 1','Parainfluenza 2','Parainfluenza 3','Parainfluenza 4'],axis=1)



#Detected Alpha coronavirus (Coronavirus229E or CoronavirusNL63)

df['Coronavirus229E'] = detect('Coronavirus229E')

df['CoronavirusNL63'] = detect('CoronavirusNL63')

df['Alpha_coronavirus'] = np.where((df['Coronavirus229E'] + df['CoronavirusNL63'] >= 1),1,0)

df = df.drop(['Coronavirus229E','CoronavirusNL63'],axis=1)



#Detected Beta coronavirus (Coronavirus HKU1 or CoronavirusOC43)

df['Coronavirus HKU1'] = detect('Coronavirus HKU1')

df['CoronavirusOC43'] = detect('CoronavirusOC43')

df['Beta_coronavirus'] = np.where((df['CoronavirusOC43'] + df['Coronavirus HKU1'] >= 1),1,0)

df = df.drop(['CoronavirusOC43','Coronavirus HKU1'],axis=1)



#Positive crovid-19

df['SARS-Cov-2 exam result'] = positive('SARS-Cov-2 exam result')



#Remove text of Urine-Ph columns

df['Urine - pH'] = np.where(df['Urine - pH']=='Não Realizado',np.nan,df['Urine - pH'])

df['Urine - pH'] = df['Urine - pH'].astype(float)



#Absent Urine - Bile pigments

df['Urine_Bile_pigments_absent'] = absent('Urine - Bile pigments')

df = df.drop('Urine - Bile pigments',axis=1)



#Absent 'Urine - Ketone Bodies'

df['Urine_Ketone_Bodies_absent'] = absent('Urine - Ketone Bodies')

df = df.drop('Urine - Ketone Bodies',axis=1)



#Absent  'Urine - Hyaline cylinders'

df['Urine_Hyaline_cylinders_absent'] = absent('Urine - Hyaline cylinders')

df = df.drop('Urine - Hyaline cylinders',axis=1)



#Absent 'Urine - Yeasts'

df['Urine_Yeasts_absent'] = absent('Urine - Yeasts')

df = df.drop('Urine - Yeasts',axis=1)



#Absent 'Urine - Protein'

df['Urine_Protein_absent'] = absent('Urine - Protein')

df = df.drop('Urine - Protein',axis=1)



#Absent 'Urine - Esterase'

df['Urine_Esterase_absent'] = absent('Urine - Esterase')

df = df.drop('Urine - Esterase',axis=1)



#Absent 'Urine -  Granular cylinders'

df['Urine_Granular_cylinders_absent'] = absent('Urine - Granular cylinders')

df = df.drop('Urine - Granular cylinders',axis=1)



#Urine - Leukocytes

df['Urine - Leukocytes'].replace('<', '', regex = True,inplace=True)

df['Urine - Leukocytes'] = df['Urine - Leukocytes'].astype(float)



#Urine - Hemoglobin present and ausent

df['Urine_Hemoglobin_present'] = present('Urine - Hemoglobin')

df['Urine_Hemoglobin_present'] = absent('Urine - Hemoglobin')

df = df.drop('Urine - Hemoglobin',axis=1)



#Urine yellow or light yellow

df['Urine_yellow']= np.where(df['Urine - Color']=='yellow',1,0)

df['Urine_light_yellow']= np.where(df['Urine - Color']=='light_yellow',1,0)

df['Urine_color_yellow'] = np.where(df['Urine_yellow'] + df['Urine_light_yellow'] >=1,1,0)

df = df.drop(['Urine - Color','Urine_yellow','Urine_light_yellow'],axis=1)



#code to fix columns name

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]



#Input median in null values 

numerics = ['float32', 'float64']

df2 = df.select_dtypes(include=numerics).apply(lambda x: x.fillna(x.median()),axis=0)



drop_columns = []

for i in df2.columns:

    drop_columns.append(i)

df = df.drop(drop_columns, axis = 1) 



df = pd.concat([df, df2], axis=1, sort=False)

df.head(5)

#Check and print the correlation between features and target variable1 (top 10 positive correlations)

#Weak positive correlation

df.corr(method='spearman')['ward_semi_intensive'].sort_values(ascending=False).head(11)
#Split feature x and target y 

x = df.drop('ward_semi_intensive',axis=1)

y = df['ward_semi_intensive']



#30% test e 70% train

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
rf = RandomForestClassifier(max_depth=5, random_state=42,n_estimators=100,class_weight = 'balanced')

rf.fit(x_train,y_train)

p2 = rf.predict(x_test)

print('accuracy:' , accuracy_score(p2,y_test))

print('f1_score:' , f1_score(p2,y_test,average='weighted'))
lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear',penalty="l1")

lr.fit(x_train,y_train)

p3 = lr.predict(x_test)

print('accuracy:' , accuracy_score(p3,y_test))

print('f1_score:' , f1_score(p3,y_test,average='weighted'))
svm = SVC(gamma='auto',random_state=42)

svm.fit(x_train, y_train)

p5 = svm.predict(x_test)

print('f1_score:' , f1_score(p5,y_test,average='weighted'))

print('accuracy:' , accuracy_score(p5,y_test))

#Using Xgboost to predict admission to general ward, semi-intensive unit or intensive care 

xgboost = xgb.XGBClassifier(learning_rate = 0.2

                            ,max_depth = 5

                            ,colsample_bytree = 0.9

                            ,n_estimators = 100

                            ,random_state=42

                            ,class_weight='balanced'

                           )



xgboost.fit(x_train,y_train)

p = xgboost.predict(x_test)

print('f1_score:' , f1_score(p,y_test,average='weighted'))

print('accuracy:' , accuracy_score(p,y_test))
#Filter feauture with importance > 0.015

feature = []

aux= []

for feature in zip(x_train, xgboost.feature_importances_):

    if feature[1] > 0.015:

        aux.append(feature[0])

        print(feature)
x_train = x_train[aux] 

x_test  = x_test[aux]

def train_model(params):

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    colsample_bytree = params[3]

    n_estimators = params[4]

    

    

    xgboost = xgb.XGBClassifier(learning_rate=learning_rate

                                , num_leaves=num_leaves

                                , min_child_samples=min_child_samples

                                , colsample_bytree=colsample_bytree

                                , n_estimators =  n_estimators

                                , random_state=42

                                ,class_weight='balanced')

    

    xgboost.fit(x_train,y_train)

    

    p = xgboost.predict(x_test)

    

    return -f1_score(p,y_test,average='weighted')



space = [(1e-3, 1e-1, 'log-uniform'), #learning rate

         (2, 50), # num_leaves

         (1, 100), # min_child_samples

         (0.1, 1.0),# colsample bytree

         (100,300)] #n_estimator



result = dummy_minimize(train_model, space, random_state=0, n_calls=100)



#Print Hiper-parameters

print(result.x)
#New Xgboost 

xgboost = xgb.XGBClassifier(learning_rate = 0.002237781857878498

                            ,num_leaves= 29

                            ,min_child_samples = 52

                            ,colsample_bytree = 0.5168351799911199

                            ,n_estimators = 118

                            ,random_state=42

                            ,max_depth = 5

                            ,class_weight='balanced'

                           )



xgboost.fit(x_train,y_train)

p3 = xgboost.predict(x_test)

print('f1_score:' , f1_score(p3,y_test,average='weighted'))

print('accuracy:' , accuracy_score(p3,y_test))

print(classification_report(p3,y_test))

explainer = shap.TreeExplainer(xgboost)

shap_values = explainer.shap_values(x_train)

shap.summary_plot(shap_values, x_train)
stacking = VotingClassifier(estimators=[

    ('rf', rf), ('lr',lr),('xgboost',xgboost),('svm',svm)], voting='hard')

stacking.fit(x_train, y_train)

p4 = stacking.predict(x_test)

print('f1_score:' , f1_score(p4,y_test,average='weighted'))

print('accuracy:' , accuracy_score(p4,y_test))

print(classification_report(p4,y_test))
print('f1_score:' , f1_score(p4,y_test,average='weighted'))

print('accuracy:' , accuracy_score(p4,y_test))