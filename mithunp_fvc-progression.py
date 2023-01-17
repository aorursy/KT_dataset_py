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

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Lets load & have look at the train and test data

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_df.head()
#LETS SORT THE PATIENT-ID FOLLOWED BY WEEKS

sort_train=train_df.sort_values(["Patient","Weeks"])

sort_test=test_df.sort_values(["Patient","Weeks"])
print('Shape of Training data: ', train_df.shape)

print('Shape of Test data: ', test_df.shape)
train_df.info()
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")
train_patient_ids = set(train_df['Patient'].unique())

test_patient_ids = set(test_df['Patient'].unique())



train_patient_ids.intersection(test_patient_ids)
train_df.describe()
columns = train_df.keys()

columns = list(columns)

print(columns)
import os

from os import listdir

import pandas as pd

import numpy as np

import glob

import tqdm

from typing import Dict

import matplotlib.pyplot as plt

%matplotlib inline



#color

from colorama import Fore, Back, Style



import seaborn as sns

sns.set(style="whitegrid")



#pydicom

import pydicom
files = folders = 0



path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)



print(Fore.YELLOW +f'{files:,}',Style.RESET_ALL,"files/images, " + Fore.BLUE + f'{folders:,}',Style.RESET_ALL ,'folders/patients')
patient_df = train_df.groupby([train_df.Patient,train_df.Age,train_df.Sex, train_df.SmokingStatus])['Patient'].count()

patient_df.index = patient_df.index.set_names(['PatientId','Age','Sex','SmokingStatus'])

patient_df = patient_df.reset_index()

patient_df.rename(columns = {'Patient': 'freq'},inplace = True)

patient_df.rename(columns = {'PatientId': 'Patient'},inplace = True)

patient_df.shape
patient_df.head()
plt.hist(patient_df["freq"],bins=5,color='green')

plt.show()
plt.hist(patient_df["Age"],bins=20,color='blue')

plt.show()
plt.hist(patient_df["SmokingStatus"],color="orange")

plt.show()
plt.hist(patient_df["Sex"],color="red")

plt.show()
import plotly.express as px

import plotly.graph_objs as go



fig = px.histogram(patient_df, x='SmokingStatus',color = 'Sex')

fig.update_traces(marker_line_color='black',marker_line_width=2, opacity=0.85)

fig.update_layout(title = 'Distribution of SmokingStatus for unique patients')

fig.show()
fig = px.histogram(patient_df, x='Age',color = 'Sex')

fig.update_layout(title = 'Distribution of Age w.r.t Sex for unique patients')

fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)

fig.show()
fig = px.histogram(patient_df, x='Age',color = 'SmokingStatus')

fig.update_layout(title = 'Distribution of Age w.r.t SmokingStatus for unique patients')

fig.update_traces(marker_line_color='black',marker_line_width=1.5, opacity=0.85)

fig.show()
from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



train_df['Weeks'].value_counts().iplot(kind='barh',

                                      xTitle='Counts(Weeks)', 

                                      linecolor='black', 

                                      opacity=0.8,

                                      color='violet',

                                      theme='pearl',

                                      bargap=0.2,

                                      gridcolor='black',

                                      title='Distribution of the Weeks in the training set')
train_df['FVC'].iplot(kind='hist',

                      xTitle='Lung Capacity(ml)', 

                      linecolor='black', 

                      opacity=0.8,

                      color='orange',

                      bargap=0.5,

                      gridcolor='white',

                      title='Distribution of the FVC in the training set')
fig = px.scatter(train_df, x="FVC", y="Percent", color='Age')

fig.show()
fig = px.scatter(train_df, x="FVC", y="Age", color='Sex')

fig.show()
fig = px.scatter(train_df, x="FVC", y="Weeks", color='SmokingStatus')

fig.show()
patient1 = train_df[train_df.Patient == 'ID00007637202177411956430']

patient2 = train_df[train_df.Patient == 'ID00012637202177665765362']

patient3 = train_df[train_df.Patient == 'ID00082637202201836229724']

patient4 = train_df[train_df.Patient == 'ID00011637202177653955184']



patient1['text'] ='ID: ' + (patient1['Patient']).astype(str) + '<br>FVC ' + patient1['FVC'].astype(str) + '<br>Percent ' + patient1['Percent'].astype(str) + '<br>Week ' + patient1['Weeks'].astype(str)

patient2['text'] ='ID: ' + (patient2['Patient']).astype(str) + '<br>FVC ' + patient2['FVC'].astype(str)+ '<br>Percent ' + patient2['Percent'].astype(str)  + '<br>Week ' + patient2['Weeks'].astype(str)

patient3['text'] ='ID: ' + (patient3['Patient']).astype(str) + '<br>FVC ' + patient3['FVC'].astype(str) + '<br>Percent ' + patient3['Percent'].astype(str) + '<br>Week ' + patient3['Weeks'].astype(str)



fig = go.Figure()

fig.add_trace(go.Scatter(x=patient1['Weeks'], y=patient1['FVC'],hovertext = patient1['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2),

                    name='Ex-smoker'))

fig.add_trace(go.Scatter(x=patient2['Weeks'], y=patient2['FVC'],hovertext = patient2['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2),

                    name='Never smoked'))

fig.add_trace(go.Scatter(x=patient3['Weeks'], y=patient3['FVC'],hovertext = patient3['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2), name='Currently smokes'))



fig.update(layout_title_text='FVC vs Weeks for different patients')

fig.update_layout( width=1000,height=700)

fig.show()
patient1['text'] ='ID: ' + (patient1['Patient']).astype(str) + '<br>Percent ' + patient1['Percent'].astype(str) + '<br>FVC ' + patient1['FVC'].astype(str) + '<br>Week ' + patient1['Weeks'].astype(str)

patient2['text'] ='ID: ' + (patient2['Patient']).astype(str) + '<br>Percent ' + patient2['Percent'].astype(str) + '<br>FVC ' + patient2['FVC'].astype(str) + '<br>Week ' + patient2['Weeks'].astype(str)

patient3['text'] ='ID: ' + (patient3['Patient']).astype(str) + '<br>Percent ' + patient3['Percent'].astype(str) + '<br>FVC ' + patient3['FVC'].astype(str) + '<br>Week ' + patient3['Weeks'].astype(str)





fig = go.Figure()

fig.add_trace(go.Scatter(x=patient1['Weeks'], y=patient1['Percent'],hovertext = patient1['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2),

                    name='Ex-smoker'))

fig.add_trace(go.Scatter(x=patient2['Weeks'], y=patient2['Percent'],hovertext = patient2['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2),

                    name='Never smoked'))

fig.add_trace(go.Scatter(x=patient3['Weeks'], y=patient3['Percent'],hovertext = patient3['text'],

                    mode='lines+markers',marker=dict(size = 12,line_width = 2), name='Currently smokes'))



fig.update(layout_title_text='Percent vs Weeks for 3 different patients')

fig.update_layout( width=700,height=500)

fig.show()
fig=px.line(train_df.loc[650:800,:],x="Weeks",y="FVC",color="Sex",line_group="Patient",hover_name="Patient")

fig.show()
fig=px.line(train_df.loc[650:800,:],x="Weeks",y="Percent",color="Sex",line_group="Patient",hover_name="Patient")

fig.show()
fig=px.line(train_df.loc[650:800,:],x="Weeks",y="FVC",color="SmokingStatus",line_group="Patient",hover_name="Patient")

fig.show()
fig=px.line(train_df.loc[650:800,:],x="Weeks",y="Percent",color="SmokingStatus",line_group="Patient",hover_name="Patient")

fig.show()
output=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

## This file has 730 rows i.e.5*146 where 146 the number of weeks -12 to 133

#And 3 col

output.shape
test_df.head()
test_patients=test_df["Patient"].values.tolist()

test_patients
patient_dict={patient :{} for patient in test_patients}

patient_dict
for patient in patient_dict.keys():

    for i in range(len(test_df)):

        if(test_df.loc[i,"Patient"]==patient):

            patient_dict[patient][test_df.loc[i,"Weeks"]]=test_df.loc[i,"FVC"]

    for i in range(len(train_df)):

        if(train_df.loc[i,"Patient"]==patient):

            patient_dict[patient][train_df.loc[i,"Weeks"]]=train_df.loc[i,"FVC"]



print(patient_dict)
from scipy.interpolate import interp1d



for patient in patient_dict.keys():

    x=list(patient_dict[patient].keys())

    print(x)

    y=list(patient_dict[patient].values())

    print(y)

    plt.scatter(x,y)

    f=interp1d(x,y,fill_value='extrapolate')

    x_test=np.arange(-12,134)

    y_test=f(x_test)

plt.show()
for i in range(len(output)):

    patient=output.loc[i,"Patient_Week"][:25]

    x=list(patient_dict[patient].keys())

    y=list(patient_dict[patient].values())

    f=interp1d(x,y,fill_value='extrapolate')

    temp=max(min(f(int(output.loc[i,"Patient_Week"][26:])),1.1*y[0]),0.85*y[0])

    output.loc[i,"FVC"]=temp
output.to_csv("submission.csv",index=False)
length=len("ID00419637202311204720264")

for i in range(len(output)):

    patient=output.loc[i,"Patient_Week"][:25]

    x=list(patient_dict[patient].keys())

    y=list(patient_dict[patient].values())

    temp=y[0]

    output.loc[i,"FVC"]=temp
output.to_csv("submission.csv",index=False)
output.head()
length=len("ID00419637202311204720264")

for i in range(len(output)):

    patient=output.loc[i,"Patient_Week"][:25]

    x=list(patient_dict[patient].keys())

    y=list(patient_dict[patient].values())

    temp=(y[-1]+y[-2]*0.9+y[-3]*0.81+y[-4]*0.72+y[-5]*0.64+y[-6]*0.56)/(1+0.9+0.81+0.72+0.64+0.56)

    output.loc[i,"FVC"]=temp
output.to_csv("submission.csv",index=False)
output.head()
test_df_0=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_1=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_2=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_3=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_4=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_5=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_6=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_7=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_8=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_9=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

test_df_10=pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])



i=0

for patient in test_df.Patient:

    j=0

    for k in range(len(train_df)):

        if(train_df.loc[k,"Patient"]==patient):

            eval("test_df_" + str(j)).loc[i,:]=train_df.loc[k,:]

            j+=1

    i+=1



test_df_5.shape
test_df_4.head(50)
from tqdm.notebook import tqdm



train = pd.concat([train_df,test_df])



output = pd.DataFrame()



train_uniq = train.groupby('Patient') # Combines all col data by object name and return mean values respectively



tk0 = tqdm(train_uniq, total = len(train_uniq))



for _, usr_df in tk0:

    usr_output = pd.DataFrame()

    for week, tmp in usr_df.groupby("Weeks"):

        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'}

        

        tmp = tmp.rename(columns = rename_cols)

        

        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent'] 

        

        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')

        

        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']

        

        # Concat the empty DF with edited DF

        usr_output = pd.concat([usr_output, _usr_output])

    output = pd.concat([output, usr_output])

        

train = output[output['Week_passed']!=0].reset_index(drop=True)
train.shape
train.head()
test = test_df.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'})



# Adding Sample Submission

submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")



# In submisison file, format: ID_'week', using lambda to split the ID

submission['Patient'] = submission['Patient_Week'].apply(lambda x:x.split('_')[0])



# In submisison file, format: ID_'week', using lambda to split the Week

submission['predict_Week'] = submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)



test = submission.drop(columns = ["FVC", "Confidence"]).merge(test, on = 'Patient')



test['Week_passed'] = test['predict_Week'] - test['base_Week']



test.set_index('Patient_Week', inplace=True)
test.tail()
test.shape
def run_single_model(clf, train_df, test_df, folds, features, target, fold_num=0):

    trn_idx = folds[folds.fold!=fold_num].index

    val_idx = folds[folds.fold==fold_num].index

    

    y_tr = target.iloc[trn_idx].values

    X_tr = train_df.iloc[trn_idx][features].values

    y_val = target.iloc[val_idx].values

    X_val = train_df.iloc[val_idx][features].values

    

    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))

    clf.fit(X_tr, y_tr)

    

    oof[val_idx] = clf.predict(X_val)

    predictions += clf.predict(test_df[features])

    return oof, predictions
def run_kfold_model(clf, train, test, folds, features, target, n_fold=9):

    

    # n_fold from 5 to 7

    

    oof = np.zeros(len(train))

    predictions = np.zeros(len(test))

    feature_importance_df = pd.DataFrame()



    for fold_ in range(n_fold):



        _oof, _predictions = run_single_model(clf,train, test, folds, features, target, fold_num = fold_)



        oof += _oof

        predictions += _predictions/n_fold

    

    return oof, predictions
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import mean_squared_error

import category_encoders as ce



from sklearn.linear_model import Ridge, ElasticNet



TARGET='FVC'

N_FOLD=9

folds = train[['Patient', TARGET]].copy()

folds = train[['Patient', TARGET]].copy()

Fold = GroupKFold(n_splits=N_FOLD)

groups = folds['Patient'].values

for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):

    folds.loc[val_index, 'fold'] = int(n)

folds['fold'] = folds['fold'].astype(int)
target = train[TARGET]

test[TARGET] = np.nan # Displays all Null values

# features

cat_features = ['Sex', 'SmokingStatus'] # Categorical Features

num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)] # Numerical Features



features = num_features + cat_features

drop_features = [TARGET, 'predict_Week', 'Percent', 'base_Week']

features = [c for c in features if c not in drop_features]



if cat_features:

    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')

    ce_oe.fit(train)

    train = ce_oe.transform(train)

    test = ce_oe.transform(test)
import math

from functools import partial

import scipy as sp



for alpha1 in [0.3]:

    for l1s in [0.8]:

        

        print(" For alpha:",alpha1,"& l1_ratio:",l1s)

        clf = ElasticNet(alpha=alpha1, l1_ratio = l1s)

        oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)

        train['FVC_pred'] = oof

        test['FVC_pred'] = predictions

        # baseline score

        train['Confidence'] = 100

        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

        train['diff'] = abs(train['FVC'] - train['FVC_pred'])

        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

        score = train['score'].mean()

        print(score)



        def loss_func(weight, row):

            confidence = weight

            sigma_clipped = max(confidence, 70)

            diff = abs(row['FVC'] - row['FVC_pred'])

            delta = min(diff, 1000)

            score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

            return -score



        results = []

        tk0 = tqdm(train.iterrows(), total=len(train))

        for _, row in tk0:

            loss_partial = partial(loss_func, row=row)

            weight = [100]

            result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')

            x = result['x']

            results.append(x[0])



        # optimized score

        train['Confidence'] = results

        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

        train['diff'] = abs(train['FVC'] - train['FVC_pred'])

        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

        score = train['score'].mean()

        print(score)
TARGET = 'Confidence'



target = train[TARGET]

test[TARGET] = np.nan

ID="Patient_Week"

# features

cat_features = ['Sex', 'SmokingStatus']

num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]

features = num_features + cat_features

drop_features = [ID, TARGET, 'predict_Week', 'base_Week', 'FVC', 'FVC_pred']

features = [c for c in features if c not in drop_features]



oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)

train['Confidence'] = oof

train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

train['diff'] = abs(train['FVC'] - train['FVC_pred'])

train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

score = train['score'].mean()

print(score)
test['Confidence'] = predictions

test = test.reset_index()
test.tail(5)
sub = submission[['Patient_Week']].merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], on='Patient_Week')

sub = sub.rename(columns={'FVC_pred': 'FVC'})



for i in range(len(test_df)):

    sub.loc[sub['Patient_Week']==test_df.Patient[i]+'_'+str(test_df.Weeks[i]), 'FVC'] = test_df.FVC[i]

    sub.loc[sub['Patient_Week']==test_df.Patient[i]+'_'+str(test_df.Weeks[i]), 'Confidence'] = 0.1

    

sub[sub.Confidence<1]
for i in range(len(sub)):

    sub.loc[i,"Confidence"]=150



sub.to_csv('submission.csv', index=False, float_format='%.1f')
sub.head()