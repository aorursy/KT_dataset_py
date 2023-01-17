# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

import warnings

warnings.filterwarnings('ignore')

import os

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')

test = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')

submission_format = pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')
train.head()
train.describe()
test.head()
test.describe()
submission_format.head()
education_df = train[['EducationField','Department','Attrition']]

ed_field_list = []

dept_list = []

attrition_rate_list = []

HR_count = len(education_df[education_df['Department']=='Human Resources']) 

RD_count = len(education_df[education_df['Department']=='Research & Development'])

Sales_count = len(education_df[education_df['Department']=='Sales'])

education_dict = {}

for ed_field,dept_df in education_df.groupby('EducationField'):

    dept_df = dept_df.groupby('Department').sum()

    dept_dict = dept_df['Attrition'].to_dict()

    if 'Human Resources' in dept_dict:

        dept_dict['Human Resources'] = dept_dict['Human Resources']*100/HR_count

    if 'Sales' in dept_dict:

        dept_dict['Sales'] = dept_dict['Sales']*100/Sales_count

    if 'Research & Development' in dept_dict:

        dept_dict['Research & Development'] = dept_dict['Research & Development']*100/Sales_count

    education_dict[ed_field] = dept_dict

education_df = pd.DataFrame(education_dict)

count = 1 

plt.rcParams['figure.figsize'] = (10,35)

for i in list(education_df.columns):

    plt.subplot(6,1,count)

    count+=1

    plt.title('Employees with '+i+' Education Field')

    plt.bar(list(education_df.index),list(education_df[i]),color = ['orange','red','green'])

    plt.xlabel('Department')

    plt.ylabel('Department Attrition %')
def RatingMap(value):

    rating_dict = {1:'Low',2:'Medium',3:'High',4:'Very High'}

    return rating_dict[value]

employee_satisfaction_df = train[['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction','Attrition']]

employee_retained = employee_satisfaction_df[employee_satisfaction_df['Attrition']==0]

employee_left = employee_satisfaction_df[employee_satisfaction_df['Attrition']==1]

for df in [employee_retained,employee_left]:

    for col in ['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction']:

        df[col] = df[col].apply(lambda x:RatingMap(x))



#Visualizing for Employees Retained\Left

count = 1

for employee_df in [employee_retained,employee_left]:

    plt.rcParams['figure.figsize'] = (20,10)

    for col in ['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction']:

        retained_dict = employee_df.groupby(col)['Attrition'].count().to_dict()

        df = pd.DataFrame({col:list(retained_dict.keys()),'Employee Count':list(retained_dict.values())})

        plt.subplot(2,3,count)

        sns.barplot(data = df, x=col,y='Employee Count')

        count+=1

        if(count==3):

            plt.title('Employees who remained in the company')

        if(count==6):

            plt.title('Employees who left the company')
department_df = train[['Department','EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']]

for i in ['EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']:

    department_df[i] = department_df[i].apply(lambda x:RatingMap(x))

count = 1

plt.rcParams['figure.figsize'] = (30,20)

for i in ['EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']:

    department_df_list = []

    for department,rating_df in department_df.groupby('Department'):

        department_dict = {}

        df = rating_df.groupby(i)[i].count()

        df = df*100/df.sum()

        rating_dict = df.to_dict()

        department_dict[department] = rating_dict

        feature_department_df = pd.DataFrame(department_dict)

        department_df_list.append(feature_department_df)

    feature_department_df = pd.concat(department_df_list,axis=1)

    x = list(feature_department_df.index)

    for j in ['Human Resources','Research & Development','Sales']:

        plt.subplot(3,3,count)

        plt.bar(x,feature_department_df[j],color=['#32CD32','red','orange','green'])

        plt.xlabel('Satisfaction/Involvement Level')

        plt.ylabel('Percentage')

        count+=1

        plt.title(j+' Department '+i+' Level',loc='center')
def performance(val):

    performance_dict = {1:'Low',2:'Medium',3:'High',4:'Very High'}

    return performance_dict[val]

performance_df = train[['PerformanceRating','Attrition']]

performance_df['PerformanceRating'] = performance_df['PerformanceRating'].apply(lambda x:performance(x))

employees_left = performance_df[performance_df['Attrition']==1]

employees_retained = performance_df[performance_df['Attrition']==0]

#For Employees Left

plt.subplot(2,1,1)

plt.title('Performance Rating of Employees who left the organization', fontsize=14, ha='center')

employees_left = employees_left.groupby('PerformanceRating').count()

plt.pie(employees_left,explode = (0,0.1),autopct='%1.1f%%',labels=list(employees_left.index)) 

#For Employess Retained

plt.subplot(2,1,2)

plt.title('Performance Rating of Employees retained in the organization', fontsize=14, ha='center')

employees_retained = employees_retained.groupby('PerformanceRating').count()

plt.pie(employees_retained,explode = (0,0.1),autopct='%1.1f%%',labels=list(employees_left.index))

employee_year_df = train[['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','Attrition']]

employee_retained_df = employee_year_df[employee_year_df['Attrition']==0]

employee_left_df = employee_year_df[employee_year_df['Attrition']==1]

plt.rcParams['figure.figsize'] = (10,3)



Avg_years_company_retained = employee_retained_df['YearsAtCompany'].mean()

Avg_years_company_left = employee_left_df['YearsAtCompany'].mean()

plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years at Company')



Avg_years_company_retained = employee_retained_df['YearsInCurrentRole'].mean()

Avg_years_company_left = employee_left_df['YearsInCurrentRole'].mean()

plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years in Current Role')



Avg_years_company_retained = employee_retained_df['YearsSinceLastPromotion'].mean()

Avg_years_company_left = employee_left_df['YearsSinceLastPromotion'].mean()

plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years since Last Promotion')



plt.xlabel('Average Years')

plt.legend()
def behaviour(val):

    behaviour_dict = {1:'Good',2:'Bad',3:'Not Rated'}

    return behaviour_dict[val]

def communication(val):

    communication_dict = {1:'Bad',2:'Average',3:'Good',4:'Better',5:'Best'}

    return communication_dict[val]

df = train[['JobRole','Behaviour','CommunicationSkill']]

df['Behaviour'] = train['Behaviour'].apply(lambda x: behaviour(x))

df['CommunicationSkill'] = train['CommunicationSkill'].apply(lambda x: communication(x))

jobrole_count = len(set(df['JobRole']))

count = 1

plt.rcParams['figure.figsize'] = (30,30) 

for jobrole,jobrole_df in df.groupby('JobRole'):

    total_count = len(jobrole_df)

    behaviour_dict = (jobrole_df['Behaviour'].value_counts()*100/total_count).to_dict()

    comm_dict = (jobrole_df['CommunicationSkill'].value_counts()*100/total_count).to_dict()

    plt.subplot(jobrole_count,2,count)

    plt.title(jobrole+' Behaviour')

    plt.barh(list(behaviour_dict.keys()),list(behaviour_dict.values()),color='#90EE90')

    count+=1

    plt.subplot(jobrole_count,2,count)

    plt.title(jobrole+' Communication Skill')

    plt.barh(sorted(comm_dict.keys()),list(comm_dict.values()),color=['orange','red','green','lime','#90EE90'])

    count+=1
df = train[['JobRole','Attrition']]

JobRole_attrition_df = df.groupby('JobRole').sum()

total_Attrition = JobRole_attrition_df['Attrition'].sum()

JobRole_attrition_df['JobRole'] = list(JobRole_attrition_df.index) 

JobRole_attrition_df.index = range(len(JobRole_attrition_df))

JobRole_attrition_df['Attrition %'] = JobRole_attrition_df['Attrition']*100/total_Attrition

plt.rcParams['figure.figsize'] = (10,3)

sns.barplot(data = JobRole_attrition_df,x='JobRole',y='Attrition %')

plt.xticks(rotation=90)

df = train[['CommunicationSkill','Attrition']]

df['CommunicationSkill'] = df['CommunicationSkill'].apply(lambda x:communication(x))

attrition_df = df[['CommunicationSkill','Attrition']].groupby('CommunicationSkill').sum()

comm_skill_count_df = df['CommunicationSkill'].value_counts()

attrition_df['Attrition %'] = attrition_df['Attrition']*100/comm_skill_count_df

attrition_df['CommunicationSkill'] = list(attrition_df.index)

attrition_df.index = range(len(attrition_df))

sns.barplot(data=attrition_df,x='CommunicationSkill',y='Attrition %')

plt.title('Communication Skill vs Attrition %')
age_gen_df = train[['Age','Gender','Attrition']]

sns.distplot(age_gen_df['Age'])

plt.ylabel('Employee Relative frequency')
age_groups = [[10,20],[21,30],[31,40],[41,50],[51,60]]

result = []

total_attrition = age_gen_df['Attrition'].sum()

for group in age_groups:

    df = age_gen_df[(age_gen_df['Age']>=group[0]) & (age_gen_df['Age']<=group[1])]

    age_df = df[['Age','Attrition']].groupby('Age').sum()

    for gen in ['Male','Female']:

        gender_df = df[df['Gender']==gen]

        result.append([str(group[0])+'-'+str(group[1]),gender_df['Attrition'].sum()*100/total_attrition,gen])

age_group_df = pd.DataFrame(data = result,columns=['Age-group','Attrition %','Gender'])

sns.barplot(data = age_group_df,x='Age-group',y='Attrition %',hue='Gender')

plt.title('Gender-wise Age group vs Attrition %')
marriage_df = train[['MaritalStatus','BusinessTravel']]

df_list = []

for travel_info,status_df in marriage_df.groupby('BusinessTravel'):

    new_status = status_df['MaritalStatus'].value_counts()

    new_status_df = pd.DataFrame({'MaritalStatus':list(new_status.index),'Employee Count':list(new_status)})

    new_status_df.index = range(len(new_status_df))

    new_status_df['BusinessTravel'] = [travel_info]*len(new_status_df)

    df_list.append(new_status_df)

marriage_df = pd.concat(df_list)

sns.barplot(data=marriage_df,x='BusinessTravel',y='Employee Count',hue='MaritalStatus')

plt.title('Business Travel, Marital Status of Employees')
travel_df = train[['BusinessTravel','Attrition']]

total_attrition = travel_df['Attrition'].sum()

travel_attrition_df = travel_df[['BusinessTravel','Attrition']].groupby('BusinessTravel').sum()*100/total_attrition

plt.barh(list(travel_attrition_df.index),list(travel_attrition_df['Attrition']),color=['Green','Orange','Red'])

plt.xlabel('Attrition %')

plt.title('Traveling Frequency vs Attrition %')
marriage_df = train[['MaritalStatus','Attrition']]

total_attrition = marriage_df['Attrition'].sum()

marriage_attrition_df = marriage_df[['MaritalStatus','Attrition']].groupby('MaritalStatus').sum()*100/total_attrition

plt.barh(list(marriage_attrition_df.index),list(marriage_attrition_df['Attrition']),color=['Green','Orange','Blue'])

plt.xlabel('Attrition %')

plt.title('Marital Status of Employees vs Attrition %')
new_train = train.copy()

bool_series = new_train['EmployeeNumber'].duplicated()

new_train = new_train[~bool_series]
len(new_train[new_train['Attrition']==0])*100/len(new_train)
len(new_train[new_train['Attrition']==1])*100/len(new_train)
new_train.isnull().sum()
test.isnull().sum()
for col in new_train.columns:

    if(isinstance(train[col][0],str)):

        new_train[col] = LabelEncoder().fit_transform(new_train[col])
new_train = new_train.drop(['Id','EmployeeNumber'],axis = 1)
corr_df = new_train.drop('Behaviour',axis=1).corr()

sns.heatmap(corr_df,annot=True)

plt.rcParams['figure.figsize'] = (30,30)
X = new_train.drop('Attrition',axis=1)

X['MonthlyIncome'] = np.cbrt(X['MonthlyIncome'])

X['TotalWorkingYears'] = np.cbrt(X['TotalWorkingYears'])

X['YearsAtCompany'] = np.cbrt(X['YearsAtCompany'])

X['YearsSinceLastPromotion'] = np.cbrt(X['YearsSinceLastPromotion'])

X['DistanceFromHome'] = np.cbrt(X['DistanceFromHome'])

Y = new_train['Attrition']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=1)
#rf = RandomForestClassifier()

#param = {'n_estimators':[50,100,200],'max_features':range(1,X.shape[1]+1),'random_state':[0]}

#rf_gridsearch = GridSearchCV(rf,param_grid = param,n_jobs=-1,scoring='roc_auc')

#rf_gridsearch.fit(X_test,Y_test)

#rf_gridsearch.best_params_

#best_params = {'max_features': 24, 'n_estimators': 100, 'random_state': 0}
#dtc = DecisionTreeClassifier()

#param = {'max_features':range(1,X.shape[1]+1),'random_state':[0]}

#dtc_gridsearch = GridSearchCV(dtc,param_grid = param,n_jobs=-1,scoring='roc_auc')

#dtc_gridsearch.fit(X_test,Y_test)

#dtc_gridsearch.best_params_

#best_params = {'max_features': 24, 'random_state': 0}
#gbc = GradientBoostingClassifier()

#param = {'n_estimators':[50,100,200],'random_state':[0],'max_features':range(1,X.shape[1]+1),

#        'learning_rate':[0.01,0.1,1]}

#gbc_gridsearch = GridSearchCV(gbc,param_grid = param,n_jobs=-1,scoring='roc_auc')

#gbc_gridsearch.fit(X_test,Y_test)

#gbc_gridsearch.best_params_

#best_param = {'learning_rate': 1, 'max_features': 5, 'n_estimators': 100, 'random_state': 0}
#svc = SVC(probability=True)

#param = {'kernel':['rbf'],'gamma':[0.001,0.01,0.1,1,10],'C':[0.001,0.01,0.1,1,10]}

#svc_gridsearch = GridSearchCV(svc,param_grid = param,n_jobs=-1,scoring='roc_auc')

#svc_gridsearch.fit(X_test,Y_test)

#svc_gridsearch.best_params_

#best_params = {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
#log_reg = LogisticRegression()

#param = {'max_iter':[100,1000,10000],'C':[0.01,0.1,1,10]}

#log_reg_gridsearch = GridSearchCV(log_reg,param_grid = param,n_jobs=-1,scoring='roc_auc')

#log_reg_gridsearch.fit(X_test,Y_test)

#log_reg_gridsearch.best_params_

#best_params = {'C': 1, 'max_iter': 1000}
#mlp = MLPClassifier()

#param = {'random_state':[0],'activation':['logistic'],'max_iter':range(100,1100,100),

#         'solver':['lbfgs', 'sgd', 'adam'],'hidden_layer_sizes':[(100,),(1000,),(10000,)]}

#mlp_gridsearch = GridSearchCV(mlp,param_grid=param,n_jobs=-1,scoring='roc_auc')

#mlp_gridsearch.fit(X_test,Y_test)

#mlp_gridsearch.best_params_

#best_params = {'activation': 'logistic','hidden_layer_sizes': (10000,), 'max_iter': 300, 'random_state': 0, 'solver': 'adam'}
rf = RandomForestClassifier(n_estimators=100, random_state = 0,max_features = 24)

rf.fit(X_train,Y_train)

print('For Random Forest Classifier')

score = roc_auc_score(Y_train, rf.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
gbc = GradientBoostingClassifier(n_estimators=100, random_state = 0,learning_rate = 1,max_features=5)

gbc.fit(X_train,Y_train)

print('For Gradient Boost Classifier')

score = roc_auc_score(Y_train, gbc.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, gbc.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
dtc = DecisionTreeClassifier(random_state = 0,max_features=24)

dtc.fit(X_train,Y_train)

print('For Decision Tree Classifier')

score = roc_auc_score(Y_train, dtc.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, dtc.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
svc = SVC(probability=True,kernel='rbf',C=0.1,gamma=0.001)

svc.fit(X_train,Y_train)

print('For Support Vector Classifier')

score = roc_auc_score(Y_train, svc.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, svc.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
log_reg = LogisticRegression(C = 1,max_iter=1000) 

log_reg.fit(X_train,Y_train)

print('For Logistic Regression')

score = roc_auc_score(Y_train, log_reg.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, log_reg.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
mlp = MLPClassifier(random_state=0,activation='logistic',max_iter=300,hidden_layer_sizes=(10000,))

mlp.fit(X_train,Y_train)

print('For Mulit-Layer Perceptron')

score = roc_auc_score(Y_train, mlp.predict_proba(X_train)[:,1])

print('Train roc_auc_score:',score)

score = roc_auc_score(Y_test, mlp.predict_proba(X_test)[:,1])

print("Test roc_auc_score:",score)
#Fitting Models

models = [rf,gbc,dtc,svc,log_reg,mlp]

for model in models:

    model.fit(X,Y)
new_test = test.copy()

for col in new_test.columns:

    if(isinstance(test[col][0],str)):

        new_test[col] = LabelEncoder().fit_transform(new_test[col])

new_test = new_test.drop(['Id','EmployeeNumber'],axis = 1)

X_test = new_test

X_test['MonthlyIncome'] = np.cbrt(X_test['MonthlyIncome'])

X_test['TotalWorkingYears'] = np.cbrt(X_test['TotalWorkingYears'])

X_test['YearsAtCompany'] = np.cbrt(X_test['YearsAtCompany'])

X_test['YearsSinceLastPromotion'] = np.cbrt(X_test['YearsSinceLastPromotion'])

X_test['DistanceFromHome'] = np.cbrt(X_test['DistanceFromHome'])
models = [rf,gbc,dtc,svc,log_reg,mlp]

modelname = ['Random Forest','GradientBoost','DecisionTree','SupportVector','Logistic_reg','MLPClassifier']

for model,name in zip(models,modelname):

    test_prob = model.predict_proba(X_test)[:,1]

    result = pd.DataFrame({'Id':list(test['Id']),'Attrition':list(test_prob)})

    result.to_csv('/kaggle/working/'+str(name)+'.csv',index=False)