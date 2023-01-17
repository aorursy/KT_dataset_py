# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install seaborn

!pip install bubbly

!pip install plotly

!pip install IPython

!pip install eli5

!pip install shap

!pip install pdpbox
import pandas as pd

from IPython.display import Image

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.font_manager as fm

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

from bubbly.bubbly import bubbleplot

import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

import eli5

from eli5.sklearn import PermutationImportance

import shap

from pdpbox import pdp, info_plots

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix, classification_report

from sklearn import tree

import numpy as np

from sklearn.tree import export_graphviz

import graphviz
dt = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

dt.head()
# data shape

dt.shape
# Check data type

dt.info()
# Change columns's name generally

dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
# Divided features by scale

category_ft_dt = dt[['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','num_major_vessels','thalassemia','target']]

numeric_ft_dt = dt[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','target']]

target = dt[['target']]
print('category features','\n',category_ft_dt.head())

print('numeric features','\n',numeric_ft_dt.head())
# categories value change string

eda_heart1=dt.copy()

eda_heart1['sex'][eda_heart1['sex']==0] ='female'

eda_heart1['sex'][eda_heart1['sex']==1] ='male'

eda_heart1['chest_pain_type'][eda_heart1['chest_pain_type']==0] = 'typical angina'

eda_heart1['chest_pain_type'][eda_heart1['chest_pain_type']==1] = 'atypical angina'

eda_heart1['chest_pain_type'][eda_heart1['chest_pain_type']==2] = 'non-anginal pain'

eda_heart1['chest_pain_type'][eda_heart1['chest_pain_type']==3] = 'asymptomatic'

eda_heart1['fasting_blood_sugar'][eda_heart1['fasting_blood_sugar']==0] = 'lower than 120mg/ml'

eda_heart1['fasting_blood_sugar'][eda_heart1['fasting_blood_sugar']==1] = 'greater than 120mg/ml'

eda_heart1['rest_ecg'][eda_heart1['rest_ecg']==0] = 'normal'

eda_heart1['rest_ecg'][eda_heart1['rest_ecg']==1] = 'ST-T wave abnormality'

eda_heart1['rest_ecg'][eda_heart1['rest_ecg']==2] = 'left ventricular hypertrophy'

eda_heart1['exercise_induced_angina'][eda_heart1['exercise_induced_angina']==0] = 'no'

eda_heart1['exercise_induced_angina'][eda_heart1['exercise_induced_angina']==1] = 'yes'

eda_heart1['st_slope'][eda_heart1['st_slope']==0] = 'upsloping'

eda_heart1['st_slope'][eda_heart1['st_slope']==1] = 'flat'

eda_heart1['st_slope'][eda_heart1['st_slope']==2] = 'downsloping'

eda_heart1['thalassemia'][eda_heart1['thalassemia']==0] = 'Unknown'

eda_heart1['thalassemia'][eda_heart1['thalassemia']==1] = 'normal'

eda_heart1['thalassemia'][eda_heart1['thalassemia']==2] = 'fixed defect'

eda_heart1['thalassemia'][eda_heart1['thalassemia']==3] = 'reversable defect'

eda_heart1['target'][eda_heart1['target']==0] = '< 50% diameter narrowing'

eda_heart1['target'][eda_heart1['target']==1] = '> 50% diameter narrowing'
eda_heart1.head()
plt.rcParams['axes.axisbelow'] = True

plt.figure(figsize=(30,30))

for i in range(len(category_ft_dt.columns)):

    plt.subplot(3,3,i+1)

    sns.countplot(eda_heart1[category_ft_dt.columns[i]])

    plt.xlabel(category_ft_dt.columns[i],size=20)

    plt.ylabel('count',size=20, rotation=0,labelpad=30)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    plt.grid(axis='y')
print('category data ratio','\n')

for i in category_ft_dt.columns:

    print(i, '\n')

    for j in eda_heart1[i].value_counts().index:

        print(j,' : ',round(eda_heart1[i].value_counts()[j]/eda_heart1.shape[0]*100,2),'%')

    print()
xlabeling=['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','num_major_vessels','thalassemia']

for i in range(len(xlabeling)):

    f,ax = plt.subplots(figsize=(10,8))

    sns.countplot(x=xlabeling[i],hue="target",data=eda_heart1)

    bars = ax.patches

    half = int(len(ax.patches)/2)

    plt.xlabel(xlabeling[i],size=20)

    plt.ylabel('count',size=20, rotation=0,labelpad=30)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    plt.legend(loc=1,prop={'size':20})

    plt.grid(axis='y')



    for first,second in zip(bars[:half],bars[half:]):

        height1 =  first.get_height()

        height2 = second.get_height()

        total_height= height1+height2

        ax.text(first.get_x()+first.get_width()/2, height1+1,'{0:.0%}'.format(height1/total_height), ha ='center')

        ax.text(second.get_x()+second.get_width()/2, height2+1,'{0:.0%}'.format(height2/total_height), ha ='center')
labeling = ['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']

graph_title=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']

y_labeling = ['age','mm/hg','mg/dl','heart_rate','inclination']

plt.figure(figsize=(40,20))

for i in  range(len(labeling)):

    plt.subplot(2,3,i+1)

    sns.boxplot('target',labeling[i],data=numeric_ft_dt)

    plt.title(graph_title[i],size=30)

    plt.xlabel('target',size=20)

    plt.ylabel(y_labeling[i],size=20, rotation=0,labelpad=53)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)
numeric_ft_dt.drop(['target'],axis=1).describe()
eda_heart2=numeric_ft_dt.copy()

eda_heart2['interval_age']=pd.cut(eda_heart2.age,bins=[28,38,48,58,68,78])

eda_heart2['interval_resting_blood_pressure']=pd.cut(eda_heart2.resting_blood_pressure,bins=range(min(eda_heart2.resting_blood_pressure),max(eda_heart2.resting_blood_pressure)+20,20))

eda_heart2['interval_cholesterol']=pd.cut(eda_heart2.cholesterol,bins=range(min(eda_heart2.cholesterol),max(eda_heart2.cholesterol)+100,100))

eda_heart2['interval_max_heart_rate_achieved']=pd.cut(eda_heart2.max_heart_rate_achieved,bins=range(min(eda_heart2.max_heart_rate_achieved),max(eda_heart2.max_heart_rate_achieved)+20,20))

eda_heart2['interval_st_depression']=pd.cut(eda_heart2.st_depression,bins=[-0.1,1,2,3,4,5,6,7])
eda_heart2['target_name']=eda_heart1['target']

labeling = ['interval_age','interval_resting_blood_pressure','interval_cholesterol','interval_max_heart_rate_achieved','interval_st_depression']

for i in range(len(labeling)):

    f,ax = plt.subplots(figsize=(10,8))

    sns.countplot(x=labeling[i],hue="target_name",data=eda_heart2)

    bars = ax.patches

    half = int(len(ax.patches)/2)

    plt.title(graph_title[i],size=30)

    plt.xlabel(labeling[i],size=20)

    plt.ylabel('count',size=20, rotation=0,labelpad=30)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    plt.legend(loc=1,prop={'size':20})

    plt.grid(axis='y')



    for first,second in zip(bars[:half],bars[half:]):

        height1 =  first.get_height()

        height2 = second.get_height()

        total_height= height1+height2

        ax.text(first.get_x()+first.get_width()/2, height1+1,'{0:.0%}'.format(height1/total_height), ha ='center')

        ax.text(second.get_x()+second.get_width()/2, height2+1,'{0:.0%}'.format(height2/total_height), ha ='center')
# nan% check

for i in labeling:

    for j in range(len(eda_heart2[i].unique())):

        print(eda_heart2[i].unique()[j],round((len(eda_heart2[eda_heart2[i]==eda_heart2[i].unique()[j]]))/303,4))
numeric_ft_dt['target']==numeric_ft_dt.target.astype('object')

dum=pd.get_dummies(numeric_ft_dt.target,prefix='target')

corr_numeric_dt=pd.concat([numeric_ft_dt,dum],axis=1).drop('target',axis=1)

corr_numeric_dt.head()
plt.figure(figsize=(20,10))

sns.heatmap(numeric_ft_dt.corr(),annot =True, cmap = 'Wistia')

plt.title('Heatmap for the Dataset',fontsize=20)

plt.show()
sns.pairplot(numeric_ft_dt, hue='target')

plt.show()
association_dt=eda_heart1[['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','thalassemia','target']]

association_dt.astype('category')
sex_crosstab=pd.crosstab(association_dt['sex'],association_dt.target,margins=True)

chest_pain_type_crosstab=pd.crosstab(association_dt['chest_pain_type'],association_dt.target,margins=True)

fasting_blood_sugar_crosstab=pd.crosstab(association_dt['fasting_blood_sugar'],association_dt.target,margins=True)

rest_ecg_crosstab=pd.crosstab(association_dt['rest_ecg'],association_dt.target,margins=True)

exercise_induced_angina_crosstab=pd.crosstab(association_dt['exercise_induced_angina'],association_dt.target,margins=True)

st_slope_crosstab=pd.crosstab(association_dt['st_slope'],association_dt.target,margins=True)

thalassemia_crosstab=pd.crosstab(association_dt['thalassemia'],association_dt.target,margins=True)
from scipy.stats import chisquare, chi2_contingency, fisher_exact

stat, p, dof, expect=chi2_contingency(sex_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

sex_crosstab
stat, p, dof, expect=chi2_contingency(chest_pain_type_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

chest_pain_type_crosstab
stat, p, dof, expect=chi2_contingency(fasting_blood_sugar_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

fasting_blood_sugar_crosstab
stat, p, dof, expect=chi2_contingency(rest_ecg_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)



rest_ecg_crosstab
stat, p, dof, expect=chi2_contingency(exercise_induced_angina_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

exercise_induced_angina_crosstab
stat, p, dof, expect=chi2_contingency(st_slope_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

st_slope_crosstab
stat, p, dof, expect=chi2_contingency(thalassemia_crosstab.iloc[:-1,:-1])

print('test statistic : ',round(stat,4),'\n',

     'p_value : ',round(p,5),'\n',

     'freedom : ',dof,'\n',

     'Expected frequency : ',expect)

thalassemia_crosstab
train_data=eda_heart1.copy()
for i in ['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','thalassemia']:

    aa=pd.get_dummies(train_data[i])

    train_data=pd.concat([train_data,aa],axis=1)

    train_data=train_data.drop([i],axis=1)
train_data=pd.concat([train_data.drop(['target'],axis=1),dt['target']],axis=1)

train_data['target']=train_data['target'].astype('category')
X = train_data.drop(['target'],axis=1)

y = train_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y, random_state=123)
clf = RandomForestClassifier(max_depth=5, random_state=0)

clf.fit(X_train,y_train)
export_graphviz(clf.estimators_[0],feature_names=X_train.columns,filled=True, out_file='tree.dot')



# 생성된 .dot 파일을 .png로 변환

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'])



# jupyter notebook에서 .png 직접 출력

from IPython.display import Image

Image(filename = 'decistion-tree.png')
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('training accuracy : ', round(clf.score(X_train,y_train),2))

print('test accuracy : ',clf.score(X_test,y_test),'\n')

print(classification_report(y_test,y_pred))

sns.heatmap(cm,annot =True, annot_kws = {'size':15},cmap='PuBu')

plt.title('confusion_matrix')

plt.xticks([0.5,1.5],('< 50% dn predict','> 50% dn predict'))

plt.yticks([0.5,1.5],('< 50% diameter narrowing','> 50% diameter narrowing'),rotation=0)
y_pred_quant = clf.predict_proba(X_test)[:,1]



fpr,tpr,thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()

ax.plot(fpr,tpr)

ax.plot([0,1],[0,1],transform=ax.transAxes, ls="-",c='.3')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])



plt.rcParams['figure.figsize'] = (6,5)

plt.title('ROC curve for diabetes classifier',fontweight = 30)

plt.xlabel('False Positive Rate (1- Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid()

plt.show()

auc = auc(fpr,tpr)

print('AUC Score :',round(auc,2))
# model accuracy

clf.score(X_test,y_test)
# Feature importance

perm = PermutationImportance(clf,random_state=0).fit(X_test,y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist(),top=26)
train_data2=dt.copy()

for i in ['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','thalassemia']:

    aa=pd.get_dummies(train_data2[i],prefix=i)

    train_data2=pd.concat([train_data2,aa],axis=1)

    train_data2=train_data2.drop([i],axis=1)

X = train_data2.drop(['target'],axis=1)

y = train_data2['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y, random_state=123)

clf2 = RandomForestClassifier(max_depth=5, random_state=0)

clf2.fit(X_train,y_train)
clf2.score(X_test,y_test)
base_features = train_data2.columns.values.tolist()

base_features.remove('target')

aa=['st_depression','num_major_vessels','age','resting_blood_pressure','cholesterol','max_heart_rate_achieved']

for i in range(len(aa)):

    feat_name = aa[i]

    pdp_dist = pdp.pdp_isolate(model=clf, dataset=X_test,

                               model_features = base_features,

                               feature = feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1],X_test, plot_type='bar')
shap.summary_plot(shap_values[1], X_test)
def patient_analysis(model, patient):

  explainer = shap.TreeExplainer(model)

  shap_values = explainer.shap_values(patient)

  shap.initjs()

  return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
patients = X_test.iloc[1,:].astype(float)

patient_analysis(clf, patients)
patients = X_test.iloc[5,:].astype(float)

patient_analysis(clf, patients)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier()

neigh.fit(X_train,y_train)

neigh.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)

gnb.score(X_test,y_test)
from sklearn.svm import LinearSVC

lsvc = LinearSVC()

lsvc.fit(X_train,y_train)

lsvc.score(X_test,y_test)