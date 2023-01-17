import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

plt.show()
file_path='../input/osic-pulmonary-fibrosis-progression'

raw_data =pd.read_csv(file_path+'/train.csv')

test_data =pd.read_csv(file_path+'/test.csv')

sub =pd.read_csv(file_path+'/sample_submission.csv')
print('The shape of the training dataset is ', raw_data.shape)

print('The shape of the training dataset is ', test_data.shape)

print('The Totl number of patients visited :',len(raw_data.Patient.unique()))
raw_data.info()
# reading the data from the first week of evry patient

df=raw_data.groupby(['Patient']).first()
df.head()
Smoke=df.groupby(['SmokingStatus']).count()['Sex'].to_frame()

Smoke
sns.barplot(x=Smoke.Sex.keys(),y=Smoke.Sex.values)
df.groupby(['Sex']).count()['SmokingStatus'].to_frame()
plt.figure(figsize=(10, 5))

sns.countplot(data=df, x='SmokingStatus', hue='Sex')
# Age dostribution plot 

mu=df.Age.std()

mean=df.Age.mean()

plt.figure(figsize=(10,6))

plt.title('Age distirbution [mu {:.2f} and mean {:.2f}]'.format(mu,mean),fontsize=15,color='black')

sns.distplot(df['Age'],kde=True)
# smoking staus versus Age distribution

smoker_dist=df.loc[df.SmokingStatus=='Currently smokes']['Age']

exsmoker_dist=df.loc[df.SmokingStatus=='Ex-smoker']['Age']

nonsmoker_dist=df.loc[df.SmokingStatus=='Never smoked']['Age']



plt.figure(figsize=(10,6))

sns.kdeplot(smoker_dist,shade=True,label='currenty smokes')

sns.kdeplot(exsmoker_dist,shade=True,label='Ex-smoker')

sns.kdeplot(nonsmoker_dist,shade=True,label='Never smoked')
# Gender and Age distribution

Male_dist=df.loc[df.Sex=='Male']['Age']

Female_dist=df.loc[df.Sex=='Female']['Age']



plt.figure(figsize=(10,6))

sns.kdeplot(Male_dist,shade=True,label='Male')

sns.kdeplot(Female_dist,shade=True,label='Female')
plt.figure(figsize=(10,6))

sns.swarmplot(x=df["Sex"],y=df['Age'],hue=df['SmokingStatus'])
patient_ids=raw_data.Patient.unique()
patient_week=[]

patient_fvc=[]

patient_percentage=[]

for ids in patient_ids:

    week=raw_data.loc[raw_data['Patient']==ids]['Weeks'].values

    fvc=raw_data.loc[raw_data['Patient']==ids]['FVC'].values

    percent=raw_data.loc[raw_data['Patient']==ids]['Percent'].values

    patient_week.append(week)

    patient_fvc.append(fvc)

    patient_percentage.append(percent)
plt.figure(figsize=(10,10))

plt.title("Each patient's FVC decay over the weeks")

plt.xlabel('Weeks')

plt.ylabel('FVC deacy ')

for i in range(len(patient_ids)):

    sns.lineplot(x=patient_week[i],y=patient_fvc[i],label ='P'+str(i+1),lw=1,legend=False)
plt.figure(figsize=(10,10))

plt.title("Each patient's Percentage over the weeks")

plt.xlabel('Weeks')

plt.ylabel('Percentage')

for i in range(len(patient_ids)):

    sns.lineplot(x=patient_week[i],y=patient_percentage[i],label ='P'+str(i+1),lw=1,legend=False)
plt.figure(figsize=(10,10))

plt.title("Each patient's Percentage Vs FVC")

plt.xlabel('FVC')

plt.ylabel('Percentage')

for i in range(len(patient_ids)):

    sns.lineplot(x=patient_fvc[i],y=patient_percentage[i],label ='P'+str(i+1),lw=1,legend=False)
df_base = raw_data.drop_duplicates(subset='Patient', keep='first')

df_base = df_base[['Patient', 'Weeks', 'FVC', 

                   'Percent', 'Age']].rename(columns={'Weeks': 'base_week',

                                                      'Percent': 'base_percent',

                                                      'Age': 'base_age',

                                                      'FVC': 'base_FVC'})
data_train =raw_data.merge(df_base,how='left',on=['Patient'])

data_train =data_train.loc[data_train.Weeks!=data_train.base_week]# removing the first week from the weeks

data_train['week_count']=data_train.Weeks-data_train.base_week # to check the weeks count from base week



data_train= pd.get_dummies(data_train,columns=['Sex','SmokingStatus']) # to get the dummy columns for Sex and smokingststaus
data_train.head()
data_train_inp_file =data_train.drop(columns=['Patient','FVC','Percent','Weeks','Age'],axis=1)
target =data_train['FVC']
def log_likely_hood(y_true,y_pred,y_pred_std):

    

    sigma_clipped = np.maximum(y_pred_std,70)

    

    delta = np.minimum(abs(y_true-y_pred),1000)

    

    metric = -(np.sqrt(2*delta)/sigma_clipped)-np.log(np.sqrt(2*sigma_clipped))

    

    return np.mean(metric)
from sklearn.linear_model import ARDRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data_train_inp_file,target,test_size=0.2)
X_train.shape,y_test.shape
from sklearn.preprocessing import StandardScaler



scalar= StandardScaler()



X_train_scaled = scalar.fit_transform(X_train)



X_test_scaled = scalar.fit_transform(X_test)
ard= ARDRegression()

ard.fit(X_train_scaled,y_train)
y_pred,y_pred_std=ard.predict(X_test_scaled,return_std=True)
log_likely_hood(y_test,y_pred,y_pred_std)# prediction on training set
sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

sub.head()
sub_mod = sub.drop(columns=['FVC','Confidence'],axis=1)

sub_mod.head()
df_test=test_data.rename(columns={'Weeks': 'base_week',

                                'Percent': 'base_percent',

                                'Age': 'base_age',

                                'FVC': 'base_FVC'})



df_test=pd.get_dummies(df_test,columns=['Sex','SmokingStatus'])
df_test['Sex_Female']=0

df_test['SmokingStatus_Currently smokes']=0
df_test_mod2=sub_mod.merge(df_test,how='left',on=['Patient'])

sub2= df_test_mod2.copy()
sub2.head()
data_test_inp_file = sub2.copy()

data_test_inp_file['week_count']= data_test_inp_file.Weeks-data_test_inp_file.base_week

data_test_inp_file.drop(['Patient','Weeks'],axis=1,inplace=True)
features=data_train_inp_file.columns

data_test_inp_file_id=data_test_inp_file['Patient_Week']

data_test_file =data_test_inp_file.drop(['Patient_Week'],axis=1)

data_test_file = data_test_file[features]
data_test_file.head()
y_pred_test,y_pred_test_std=ard.predict(data_test_file.values,return_std=True)

submission=pd.DataFrame({'Patient_Week':data_test_inp_file_id,'FVC':y_pred_test,'Confidence':y_pred_test_std})

submission.to_csv('submission.csv', index=False)