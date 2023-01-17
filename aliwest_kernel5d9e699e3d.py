#Import The packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#reading the data set

data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.tail()
#Here we can see that there's missing values in salaries

data.describe(include='all')
#See how many missing values in the data set

data.isnull().sum()
#Droping the salary column because it's not so important givin that we want to know only the status of palced or not 

#Salaries are giving after being placed

#People who are not placed don't have a salary

data=data.drop(['salary'],axis=1)
#Check how is the data distributed

sns.distplot(data['hsc_p'])
data['hsc_p'].mean()
sns.distplot(data['ssc_p'])
sns.distplot(data['degree_p'])
#The data after cleaning

data.describe(include='all')
#Maping the binary features like gender and work experince

data1=data.copy()

data1['status'] =data1['status'].map({'Not Placed':0 ,'Placed':1})

data1['workex'] =data1['workex'].map({'No':0 ,'Yes':1})
cols = ['status','sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',

       'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation',

       'mba_p']
#Data preproccesd

datapp = data1[cols]
datapp['gender'] = datapp['gender'].map({'F':0,'M':1})
data1.tail()
import statsmodels.api as sm
datapp
#Define dependent and independent feautres

target = datapp['status']

inputs = datapp.drop(['status','hsc_s','sl_no','ssc_b','hsc_b','degree_t','specialisation','workex'],axis=1)
inputs
y = target

x1 = inputs
#Create a regression with the significant features .

#We can see that etest_p which is a test done by college for placeing students which is a litttle insignificant we will keep it

#because it's used by the college to decide 

x=sm.add_constant(x1)

reg_log1 = sm.Logit(y,x)

result_log=reg_log1.fit()

result_log.summary()
#There's 139 Male

datapp['gender'].sum()
#The odds of being male and get placed is 4 times bigger than being female

Gender_Odds=np.exp(1.3941)

Gender_Odds
#Checking probablity for all rows

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})

result_log.predict()
#Comparing the probabilty with the actual values

np.array(datapp['status'])
#the confusion table predicted 49 will not be placed which is true for the model and 18 will be placed which is not ture

#Also it predicted that 10 will be placed which is not true and 138 will be placed which is true 

result_log.pred_table()
#The model was 88% accurate

(51+138)/215
#And 12 inaccurate 

26/215
#Testing the model with new data

data_test = pd.read_csv('../input/testset/datasets_596958_1073629_Placement_Data_Full_Class_test.csv')
data_test['gender']=data_test['gender'].map({'F':0,'M':1})

data_test['status'] =data_test['status'].map({'Not Placed':0 ,'Placed':1})

data_test['workex'] =data_test['workex'].map({'No':0 ,'Yes':1})
data_test
data_test_inputs = data_test.drop(['status','hsc_s','sl_no','ssc_b','hsc_b','degree_t','specialisation','workex','salary'],axis=1)
data_test_inputs
test_actual = data_test['status']

data_test_inputs = sm.add_constant(data_test_inputs)
data_test_inputs
#Function to create a table for confusion and accuracy

def confusion_matrix(data,actual,model):

    pred_values = model.predict(data)

    bins = np.array([0,0.5,1])

    cm=np.histogram2d(actual,pred_values,bins=bins)[0]

    accuracy = (cm[0,0]+cm[1,1])/cm.sum()

    return cm ,accuracy
#The Model has 90% accuarcy with the new data 

#Note I only used data with 30 row so it need further testings

cm = confusion_matrix(data_test_inputs,test_actual,result_log)

cm