

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



from plotly.offline import iplot, init_notebook_mode



import cufflinks as cf

import plotly.graph_objs as go

# import chart_studio.plotly as py



init_notebook_mode(connected=True)

cf.go_offline(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='ggplot')
df = pd.read_csv('/kaggle/input/employee-attrition/employee_attrition_train.csv', na_values= np.nan)
df.columns.str.upper().values
df.columns = ['AGE', 'ATTRITION', 'BUSINESS_TRAVEL', 'DAILY_RATE', 'DEPARTMENT',

 'DISTANCE_FROM_HOME', 'EDUCATION', 'EDUCATION_FIELD', 'EMPLOYEE_COUNT',

 'EMPLOYEE_NUMBER', 'ENVIRONMENT_SATISFACTION', 'GENDER',

 'HOURLY_RATE', 'JOB_INVOLVEMENT', 'JOB_LEVEL', 'JOB_ROLE',

 'JOB_SATISFACTION', 'MARITAL_STATUS', 'MONTHLY_INCOME', 'MONTHLY_RATE',

 'NUM_COMPANIES_WORKED', 'OVER_18', 'OVER_TIME', 'PERCENT_SALARY_HIKE',

 'PERFORMANCE_RATING', 'RELATIONSHIP_SATISFACTION', 'STANDARD_HOURS',

 'STOCK_OPTION_LEVEL', 'TOTAL_WORKING_YEARS', 'TRAINING_TIMES_LAST_YEAR',

 'WORK_LIFE_BALANCE', 'YEARS_AT_COMPANY', 'YEARS_IN_CURRENT_ROLE',

 'YEARS_SINCE_LAST_PROMOTION', 'YEARS_WITH_CURR_MANAGER']
df.loc[:,df.dtypes=="object"].columns
df[['EMPLOYEE_NUMBER','ENVIRONMENT_SATISFACTION','JOB_INVOLVEMENT','JOB_LEVEL','JOB_SATISFACTION','RELATIONSHIP_SATISFACTION','WORK_LIFE_BALANCE','STOCK_OPTION_LEVEL']] = df[['EMPLOYEE_NUMBER','ENVIRONMENT_SATISFACTION','JOB_INVOLVEMENT','JOB_LEVEL','JOB_SATISFACTION','RELATIONSHIP_SATISFACTION','WORK_LIFE_BALANCE','STOCK_OPTION_LEVEL']].astype(str)
df.info()
print(df.iloc[:2,:10].head())

print(df.iloc[:2,10:20].head())

print(df.iloc[:2,20:].head())
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt
def customized_heatmap(corr_df):

#     corr_mat = corr_df.iloc[1:,:-1].copy()

    corr_mat = corr_df.copy()

    

    #Create masks

    mask = np.triu(np.ones_like(corr_mat), k=1)

    

    # Plot

    plt.figure(figsize=(20,14))

    plt.title("Heatmap Corrleation")

    ax = sns.heatmap(corr_mat, vmin=-1, vmax=1, cbar=False,

                     cmap='coolwarm', mask=mask, annot=True)

    

    # format the text in the plot to make it easier to read

    for text in ax.texts:

        t = float(text.get_text())

        if -0.15 < t < 0.15:

            text.set_text('')        

        else:

            text.set_text(round(t, 2))

        text.set_fontsize('x-large')

    plt.xticks( size='x-large')

    plt.yticks(rotation=0, size='x-large')

    plt.savefig("Correlation Heatmap")

    plt.show()

    

    

    
!pip install dython

from dython.nominal import associations
df_4_corr = df.copy()



df_4_corr.drop(['EMPLOYEE_NUMBER',"EMPLOYEE_COUNT"], inplace=True, axis=1)
assoc = associations(df_4_corr,plot=False,bias_correction=False)

corr_df = assoc['corr']

customized_heatmap(corr_df)

missing_df = df.copy()
missing_df.loc[:,missing_df.isnull().sum()>0].info()
bins = [i for i in range(0,49,9)]

labels =[str(val-9)+"-"+str(val) for val in bins[1:]]



working_years_bins = pd.cut(missing_df['TOTAL_WORKING_YEARS'], bins=bins, labels=labels, right=False)
missing_df['WORKING_YEARS_BINS'] = working_years_bins
grp_by_age_exp = missing_df[['WORKING_YEARS_BINS','JOB_ROLE','AGE']].groupby(['WORKING_YEARS_BINS']).mean().round()
print(grp_by_age_exp.index.values)

print("*"*20)

print(grp_by_age_exp.AGE.values)
missing_df.WORKING_YEARS_BINS.unique().tolist()
import math



# 

def age_missing_values(cols):

    age = cols[0]

    years_experience= cols[1]

    if math.isnan(age): # if age is missing

        if years_experience == '1-9':

            return 32.0

        elif years_experience == '9-18':

            return 38.0

        elif years_experience == '18-27':

            return 45.0

        elif years_experience == '27-36':

            return 52.0

        else:

            return 56.0

    else:# if age is not missing

        

        return age

        

        

        
age =missing_df[['AGE','WORKING_YEARS_BINS']].apply(age_missing_values, axis=1)

missing_df['AGE']= age
missing_df.columns
# missing_df.distancefromhome.value_counts(dropna=False)
missing_df[['DISTANCE_FROM_HOME','STANDARD_HOURS','OVER_TIME','GENDER']].head()
missing_df.STANDARD_HOURS.value_counts()
missing_df.drop('STANDARD_HOURS', axis=1, inplace=True)
missing_df[['DISTANCE_FROM_HOME','OVER_TIME','GENDER']].groupby(['GENDER','OVER_TIME']).mean().round(2).unstack(0)
def fill_distance_frm_home(cols):

    distance = cols[0]

    gender = cols[1]

    overtime=cols[2]

    

    if math.isnan(distance):

        if gender=="Male" and overtime=="Yes":

            return 10.82

        elif gender=="Male" and overtime=="No":

            return 9.32

        elif gender=="Female" and overtime=="No":

            return 10.33

        else:

            return 10.10

    else:

        return distance

    

    
distance_frm_home = missing_df[['DISTANCE_FROM_HOME','OVER_TIME','GENDER']].apply(fill_distance_frm_home,axis=1)
missing_df['DISTANCE_FROM_HOME'] = distance_frm_home
missing_df.loc[:,missing_df.columns.str.contains('MONTH')].columns
missing_df.loc[:,missing_df.columns.str.contains('YEAR')].columns
missing_df.loc[:,missing_df.columns.str.contains('RATE')].columns
missing_df.DAILY_RATE.describe()
missing_df.DAILY_RATE.hist()
missing_df.DAILY_RATE.fillna(value=missing_df.DAILY_RATE.mean(), inplace=True)
missing_df.loc[missing_df.BUSINESS_TRAVEL.isna()][['BUSINESS_TRAVEL','JOB_ROLE','DEPARTMENT' ]]
missing_df[(missing_df.JOB_ROLE=="Research Scientist")]['BUSINESS_TRAVEL'].value_counts()
missing_df[(missing_df.JOB_ROLE=="Manufacturing Director")]['BUSINESS_TRAVEL'].value_counts()
missing_df[(missing_df.JOB_ROLE=="Sales Executive")]['BUSINESS_TRAVEL'].value_counts()
missing_df[(missing_df.JOB_ROLE=="Sales Representative")]['BUSINESS_TRAVEL'].value_counts()
missing_df[(missing_df.JOB_ROLE=="Laboratory Technician")]['BUSINESS_TRAVEL'].value_counts()
missing_df.BUSINESS_TRAVEL.fillna(value="Travel_Rarely", inplace=True)
missing_df.MARITAL_STATUS.value_counts()
missing_df.loc[missing_df.MARITAL_STATUS.isna()][['GENDER','AGE','DEPARTMENT','JOB_ROLE' ]]
missing_df[(missing_df.GENDER=="Female") & (missing_df.DEPARTMENT=="Research & Development")]['MARITAL_STATUS'].value_counts()
missing_df[(missing_df.GENDER=="Male") & (missing_df.DEPARTMENT=="Research & Development")]['MARITAL_STATUS'].value_counts()
missing_df[(missing_df.GENDER=="Male") & (missing_df.DEPARTMENT=="Sales")]['MARITAL_STATUS'].value_counts()
missing_df.dropna(inplace=True, axis=0)
missing_df.loc[:,missing_df.isna().sum()>0].columns
eda_df = missing_df.copy()
eda_df.drop('WORKING_YEARS_BINS',axis=1, inplace=True)
eda_df.head()
eda_df.drop('EMPLOYEE_COUNT', axis=1, inplace=True)
int_columns = eda_df.loc[:,eda_df.dtypes!='object'].columns



# sns.pairplot(eda_df[int_columns])



fig,axes = plt.subplots(len(int_columns), figsize=(8,35))

for i,col in enumerate(int_columns):

    axes[i].hist(eda_df[eda_df.ATTRITION == "No"][col].values, alpha=0.5, color="maroon", bins=15 )

    axes[i].hist(eda_df[eda_df.ATTRITION == "Yes"][col].values, alpha=0.5, bins=15)

    axes[i].set_title(col)

    axes[i].set_yticks(())#cause we are not actually looking for numbers

axes[0].set_xlabel("Feature Columns")

axes[0].set_ylabel("Frequency")

axes[0].legend(["No", "Yes"], loc="best")

# plt.savefig("Distribution Plot")

fig.tight_layout()
 # lets drop employee_id



eda_df.drop('EMPLOYEE_NUMBER', inplace=True, axis=1)
print(eda_df.OVER_18.unique())

#lets drop this column since it has only one values

eda_df.drop('OVER_18', inplace=True, axis=1)
str_columns = eda_df.loc[:,eda_df.dtypes=='object'].columns

eda_df[str_columns].head()

edu_levels= {    

    1: 'Highschool', 

    2 :'College  ' ,

    3: 'Bachelor', 

    4: 'Master' ,

    5 :'PHD'

}
eda_df.EDUCATION = eda_df.EDUCATION.replace(edu_levels)
# !conda install -c plotly plotly-orca
labels = eda_df.EDUCATION.value_counts().index.values

values = eda_df.EDUCATION.value_counts().values



fig = go.Figure()

fig.add_trace(go.Pie(labels=labels, values=values))

fig.update_layout(title="Qualifications Of Employees", legend_title="Degrees", template="plotly_dark")

# fig.write_image("/kaggle/working/Employee_Qualifications.png")
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()


employee_by_education_norm =eda_df[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME','EDUCATION']].copy()

employee_by_education_norm[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME']] = scaler.fit_transform(employee_by_education_norm[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME']])




employee_grp_by_education = eda_df[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME','EDUCATION']].groupby(['EDUCATION']).mean()

employee_grp_by_education_norm = employee_by_education_norm.groupby(['EDUCATION']).mean().sort_values('MONTHLY_INCOME',ascending=True)

x_value= employee_grp_by_education_norm.index.values



fig = go.Figure()

fig.add_trace(go.Bar(x=x_value, y = employee_grp_by_education_norm.AGE, name="Average Age", hovertext =employee_grp_by_education.AGE.values))

fig.add_trace(go.Bar(x=x_value, y = employee_grp_by_education_norm.TOTAL_WORKING_YEARS, name="Average Experience", hovertext =employee_grp_by_education.TOTAL_WORKING_YEARS.values))

fig.add_trace(go.Scatter(x=x_value,

                         y = employee_grp_by_education_norm.MONTHLY_INCOME,

                         mode="lines+markers", name="Monthly Income",

                         hovertext =employee_grp_by_education.MONTHLY_INCOME.values,

                         marker=dict(size =200*employee_grp_by_education_norm.MONTHLY_INCOME.values)))



fig.update_layout(title="Employees Education, Experience and Salary",

                  xaxis_title="Education Qualifications",

                  yaxis_title="Normalized Range",

                  template="plotly_white", 

                  xaxis_showgrid=False,

                  yaxis_showgrid=False)

# fig.write_image("/kaggle/working/Employee_Qualifications_n_Salary.png")
eda_df.MONTHLY_INCOME.describe()
sns.boxplot(eda_df.MONTHLY_INCOME)
# Lets check for IQR 

# IQR  = third_quartile - first_quartile



iqr = eda_df.MONTHLY_INCOME.describe()[-2] - eda_df.MONTHLY_INCOME.describe()[4]





# Upper and Lower boundry of box plot 

upper_bound = 1.5*iqr + eda_df.MONTHLY_INCOME.describe()[-2]

lower_bound = eda_df.MONTHLY_INCOME.describe()[4] - 1.5*iqr 





print(iqr, upper_bound,lower_bound)
employee_after_filtered_income_norm =eda_df[eda_df.MONTHLY_INCOME <=17000][['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME','EDUCATION']].copy()

employee_after_filtered_income_norm[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME']] = scaler.fit_transform(employee_after_filtered_income_norm[['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME']])
employee_filetered_income_grp_by_education = eda_df[eda_df.MONTHLY_INCOME <=17000][['AGE','TOTAL_WORKING_YEARS','MONTHLY_INCOME','EDUCATION']].groupby(['EDUCATION']).mean()

employee_filetered_income_grp_by_education_norm = employee_after_filtered_income_norm.groupby(['EDUCATION']).mean().sort_values('MONTHLY_INCOME',ascending=True)
x_value= employee_filetered_income_grp_by_education_norm.index.values



fig = go.Figure()

fig.add_trace(go.Bar(x=x_value,

                     y = employee_filetered_income_grp_by_education_norm.AGE,

                     name="Average Age", 

                     hovertext =employee_filetered_income_grp_by_education.AGE.values))

fig.add_trace(go.Bar(x=x_value,

                     y = employee_filetered_income_grp_by_education_norm.TOTAL_WORKING_YEARS, 

                     name="Average Experience", 

                     hovertext =employee_filetered_income_grp_by_education.TOTAL_WORKING_YEARS.values))

fig.add_trace(go.Scatter(x=x_value,

                         y = employee_filetered_income_grp_by_education_norm.MONTHLY_INCOME,

                         mode="lines+markers", 

                         name="Monthly Income",

                         hovertext =employee_filetered_income_grp_by_education.MONTHLY_INCOME.values,

                         marker=dict(size =200*employee_filetered_income_grp_by_education_norm.MONTHLY_INCOME.values)))



fig.update_layout(title="Employees Education, Experience and Salary",

                  xaxis_title="Qualifications",

                  yaxis_title="Normalized Range",

                  template="plotly_white", 

                  xaxis_showgrid=False,

                  yaxis_showgrid=False)

# fig.write_image("/kaggle/working/Employee_Qualifications_n_Salary.png")
eda_df[eda_df.EDUCATION =="Highschool"]['AGE'].describe()
highschool_education = eda_df[eda_df.EDUCATION=="Highschool"].sort_values('AGE', ascending=True).copy()

highschool_education_norm = highschool_education.copy()

highschool_education_norm.loc[:,highschool_education_norm.dtypes!= 'object'] = scaler.fit_transform(highschool_education_norm.loc[:,highschool_education_norm.dtypes!= 'object'])

# highschool_education.loc[:,highschool_education.dtypes!= 'object'].columns
highschool_grp_by_age = highschool_education.groupby(['AGE']).mean()

highschool_grp_by_age_norm = highschool_education_norm.groupby(['AGE']).mean()
x_value= highschool_grp_by_age.index.values



fig = go.Figure()



fig.add_trace(go.Scatter(x=x_value, y = highschool_grp_by_age_norm.MONTHLY_INCOME,

                         name="Average Monthly Income",mode="lines+markers",

                         hovertext=highschool_grp_by_age.MONTHLY_INCOME.values ))



fig.add_trace(go.Bar(x=x_value, y = highschool_grp_by_age_norm.YEARS_AT_COMPANY, 

                     name="Average Years On Company ",

                     hovertext= highschool_grp_by_age.YEARS_AT_COMPANY.values))

# fig.update_layout(hovermode="x")

fig.update_layout(hovermode="x unified")



eda_df.ENVIRONMENT_SATISFACTION.value_counts()
# eda_df[(eda_df.ATTRITION=="Yes") & (eda_df.JOB_SATISFACTION=="1")]

# Attrition And Environment Satisfaction 

attrition_n_env_satisfaction = eda_df[['ATTRITION','ENVIRONMENT_SATISFACTION','JOB_SATISFACTION']].groupby(['ATTRITION','ENVIRONMENT_SATISFACTION']).count().unstack(0)

attrition_n_env_satisfaction.columns= ['Attrition_No', "Attrition_Yes"]

attrition_n_env_satisfaction.index= ['level_1','level_2','level_3','level_4']





# Attrition And  JOB_INVOLVEMENT

attrition_n_job_involvement = eda_df[['ATTRITION','JOB_INVOLVEMENT','JOB_SATISFACTION']].groupby(['ATTRITION','JOB_INVOLVEMENT']).count().unstack(0)

attrition_n_job_involvement.columns= ['Attrition_No', "Attrition_Yes"]

attrition_n_job_involvement.index= ['level_1','level_2','level_3','level_4']





# Attrition And  JOB_SATISFACTION

attrition_n_job_satisfaction = eda_df[['ATTRITION','JOB_INVOLVEMENT','JOB_SATISFACTION']].groupby(['ATTRITION','JOB_SATISFACTION']).count().unstack(0)

attrition_n_job_satisfaction.columns= ['Attrition_No', "Attrition_Yes"]

attrition_n_job_satisfaction.index= ['level_1','level_2','level_3','level_4']





# Attrition And  RELATIONSHIP_SATISFACTION

attrition_n_rel_satisfaction = eda_df[['ATTRITION','JOB_INVOLVEMENT','RELATIONSHIP_SATISFACTION']].groupby(['ATTRITION','RELATIONSHIP_SATISFACTION']).count().unstack(0)

attrition_n_rel_satisfaction.columns= ['Attrition_No', "Attrition_Yes"]

attrition_n_rel_satisfaction.index= ['level_1','level_2','level_3','level_4']





# Attrition And  WORK_LIFE_BALANCE

attrition_n_work_life_bal = eda_df[['ATTRITION','JOB_INVOLVEMENT','WORK_LIFE_BALANCE']].groupby(['ATTRITION','WORK_LIFE_BALANCE']).count().unstack(0)

attrition_n_work_life_bal.columns= ['Attrition_No', "Attrition_Yes"]

attrition_n_work_life_bal.index= ['level_1','level_2','level_3','level_4']



# #Now Lets plot them by satisfaction level and attrition No



# fig = go.Figure()



# fig.add_trace(go.Scatter(x= attrition_n_work_life_bal.index.values, y= attrition_n_work_life_bal.Attrition_Yes, name="Work Life Balance"))





# fig.add_trace(go.Scatter(x= attrition_n_rel_satisfaction.index.values, y= attrition_n_rel_satisfaction.Attrition_Yes, name="Relatoinship Satisfaction"))

# fig.add_trace(go.Scatter(x= attrition_n_env_satisfaction.index.values, y= attrition_n_env_satisfaction.Attrition_Yes, name="Environment Satisfaction"))

# fig.add_trace(go.Scatter(x= attrition_n_job_involvement.index.values, y= attrition_n_job_involvement.Attrition_Yes, name="Job Involivement "))

# fig.add_trace(go.Scatter(x= attrition_n_job_satisfaction.index.values, y= attrition_n_job_satisfaction.Attrition_Yes, name="Job Satisfaction"))



# fig.update_layout(title="Satisfaction Lvl Vs Attrition = 'Yes'", 

#                  xaxis_title="Satisfaction Level, more the better",

#                 yaxis_title="Frequency",

#                   template="plotly_white", 

#                   xaxis_showgrid=False,

#                   yaxis_showgrid=False,

#                   legend_title="Features")
#Now Lets plot them by satisfaction level and attrition No



fig = go.Figure()



fig.add_trace(go.Bar(x= attrition_n_work_life_bal.index.values, y= attrition_n_work_life_bal.Attrition_Yes, name="Work Life Balance"))





fig.add_trace(go.Bar(x= attrition_n_rel_satisfaction.index.values, y= attrition_n_rel_satisfaction.Attrition_Yes, name="Relatoinship Satisfaction"))

fig.add_trace(go.Bar(x= attrition_n_env_satisfaction.index.values, y= attrition_n_env_satisfaction.Attrition_Yes, name="Environment Satisfaction"))

fig.add_trace(go.Bar(x= attrition_n_job_involvement.index.values, y= attrition_n_job_involvement.Attrition_Yes, name="Job Involivement "))

fig.add_trace(go.Bar(x= attrition_n_job_satisfaction.index.values, y= attrition_n_job_satisfaction.Attrition_Yes, name="Job Satisfaction"))



fig.update_layout(title="Satisfaction Lvl Vs Attrition = 'Yes'", 

                 xaxis_title="Satisfaction Level, more the better",

                yaxis_title="Frequency",

                  template="plotly_white", 

                  xaxis_showgrid=False,

                  yaxis_showgrid=False,

                  legend_title="Features")



# fig.write_image("/kaggle/working/Satisfaction_Lvls_n_Attrition_Yes.png")
#Now Lets plot them by satisfaction level and attrition No



fig = go.Figure()



fig.add_trace(go.Bar(x= attrition_n_work_life_bal.index.values, y= attrition_n_work_life_bal.Attrition_No, name="Work Life Balance"))





fig.add_trace(go.Bar(x= attrition_n_rel_satisfaction.index.values, y= attrition_n_rel_satisfaction.Attrition_No, name="Relatoinship Satisfaction"))

fig.add_trace(go.Bar(x= attrition_n_env_satisfaction.index.values, y= attrition_n_env_satisfaction.Attrition_No, name="Environment Satisfaction"))

fig.add_trace(go.Bar(x= attrition_n_job_involvement.index.values, y= attrition_n_job_involvement.Attrition_No, name="Job Involivement "))

fig.add_trace(go.Bar(x= attrition_n_job_satisfaction.index.values, y= attrition_n_job_satisfaction.Attrition_No, name="Job Satisfaction"))



fig.update_layout(title="Satisfaction Lvl Vs Attrition = 'No'", 

                 xaxis_title="Satisfaction Level, more the better",

                yaxis_title="Frequency",

                  template="plotly_white", 

                  xaxis_showgrid=False,

                  yaxis_showgrid=False,

                  legend_title="Features")

# fig.write_image("/kaggle/working/Satisfaction_Lvls_n_Attrition_No.png")
# eda_df[['ENVIRONMENT_SATISFACTION','ATTRITION', 'DEPARTMENT', 'GENDER','JOB_INVOLVEMENT','JOB_LEVEL','JOB_SATISFACTION','MARITAL_STATUS',

#        'OVER_TIME','PERFORMANCE_RATING','RELATIONSHIP_SATISFACTION', 'WORK_LIFE_BALANCE','YEARS_SINCE_LAST_PROMOTION']]
feature_df = eda_df.copy()
feature_df.drop('GENDER', inplace=True, axis=1)
int_columns = feature_df.loc[:,feature_df.dtypes=="int"].columns

obj_columns = feature_df.loc[:,feature_df.dtypes!="int"].columns
dummified_feature_df = pd.get_dummies(feature_df.loc[:,feature_df.columns != "ATTRITION"],drop_first=True)

target =feature_df['ATTRITION']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dummified_feature_df[int_columns] = scaler.fit_transform(dummified_feature_df[int_columns])
from sklearn.model_selection import train_test_split, KFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(dummified_feature_df, target, test_size=0.3, random_state=101)
X_hold, X_test_final, y_hold, y_test_final = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

num_folds = 10

scoring = "accuracy"

models=[]

rand_seed= 101



models.append(("Knn",KNeighborsClassifier(n_neighbors=5,p=2,leaf_size=10,) ))

models.append(("Svm",SVC(random_state=rand_seed )))

models.append(("Rf", RandomForestClassifier(n_estimators=50,max_depth=5,random_state=rand_seed )))







results=[]

names=[]

metrics=[]
for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=rand_seed, shuffle=True)

    cv_score = cross_val_score(model,X_train,y_train, cv=kfold, scoring=scoring)

    

    names.append(name)

    results.append(cv_score)

    metrics.append(cv_score.mean())

    

    print("{name}: {score}".format(name=name,score= cv_score.mean()))
from sklearn.model_selection import RandomizedSearchCV



from sklearn.model_selection import GridSearchCV
# # Number of trees in random forest

# n_estimators = [i for i in range(200,2001, 200)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [i for i in range(10, 111,10)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}

# print(random_grid)
# COmmented to save version faster





# rf = RandomForestClassifier()





# # Random search of parameters, using 3 fold cross validation, 

# # search across 80 different combinations, and use all available cores

# rf_random_cv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 80, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random_cv.fit(X_train,y_train)



# print(rf_random_cv.best_params_)
param_grid_rf = {

    'bootstrap':[False],

    'max_depth': [10,20,30,20,50],

    'max_features': ['auto'],

    'min_samples_leaf': [ 1,3,4, 8],

    'min_samples_split': [2,4,6,8],

    'n_estimators': [200,400,600,1200,1000]

}
# # Instantiate the grid search model

# grid_search_cv_rf = GridSearchCV(RandomForestClassifier(), param_grid = param_grid_rf, 

#                           cv = 10, n_jobs = -1, verbose = 0)



# grid_search_cv_rf.fit(X_train, y_train)





#Commented to run version 
# print(grid_search_cv_rf.best_params_)
final_rf= RandomForestClassifier(bootstrap=False,                                

                                   max_depth=50,

                                   min_samples_leaf=8,

                                   min_samples_split=8,

                                   n_estimators=200)
import time
rf_start = time.time()



final_rf.fit(X_train,y_train)

rf_end = time.time()

eval_time_rf = rf_end -rf_start



# print("Accuracy For Random Forest on Test Set: {}.".format(pipe.score(X_test_final,y_test_final)*100) )

print("Accuracy For Random Forest on Hold out Set: {}.".format(final_rf.score(X_hold,y_hold)*100) )

print("Total time taken by RF to fit the model: {:.2f} sec".format(eval_time_rf))

from sklearn.decomposition import PCA
pca = PCA(2)
pca_transformed= pca.fit_transform(dummified_feature_df)
pca_transformed_df = pd.DataFrame(pca.components_, columns=dummified_feature_df.columns)
plt.figure(figsize=(12,5))

sns.heatmap(pca_transformed_df,cmap="viridis", yticklabels=["Comp 1", "Comp 2"] ,lw=1,linecolor="black" );

plt.yticks(rotation=360);

np.abs(pca_transformed_df.iloc[0,:]).sort_values().iplot(kind="bar")
np.abs(pca_transformed_df.iloc[1,:]).sort_values().iplot(kind="bar")
np.abs(pca_transformed_df.iloc[0,:].sort_values())