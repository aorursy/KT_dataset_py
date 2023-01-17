import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

employee_df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
employee_df.head()
employee_df['PercentSalaryHike'].describe()
employee_df['PercentSalaryHike'].unique()
#import for interactive plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
employee_df.info()
sns.set()
# Let's see if we have any missing data, luckily we don't!
sns.heatmap(employee_df.isnull(), yticklabels=False, cbar=False, cmap='Blues')
for column in employee_df.select_dtypes(include=['object']):
    print(column)
    print(employee_df[column].unique())
    print('-----------------------------------')
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x:1 if x == "Yes" else 0 )
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x:1 if x =="Yes" else 0 )
#drop the column, use inplce True for delete the information from the memeory
employee_df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1, inplace=True)
employee_df.head()
# Visulazing the distibution of the data for every feature
employee_df.hist(bins=30, figsize=(20,20), color='b', alpha=0.6)
attrition = employee_df[employee_df['Attrition'] == 1]
no_attrition = employee_df[employee_df['Attrition']==0]
fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type":"xy"},{"type":"domain"}]],
                    subplot_titles= ("Count of Attrition", "Distribution of Attrition"))

fig.add_trace(go.Bar(x = employee_df['Attrition'].value_counts(),
                     y = ['Employee who stayed', 'Employee who left'],
                     orientation = 'h',
                     opacity=0.8),
                     row=1,col=1)

fig.add_trace(go.Pie(values=employee_df['Attrition'].value_counts(),
                    opacity=0.8),
                    row=1, col=2)
fig.update_layout(height=400, showlegend=False)

fig.show()
def categorical_colum_investigaton(col_name):
    """First Plot: Pie chart for categorical column to see percentage of each value
       Secons Plot: Count plot for categorical column to see the number of count for each of the type
       Third Plot is Number of Count for separeted for Attribition"""

    f,ax = plt.subplots(1,3, figsize=(18,6))
    employee_df[col_name].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True, cmap='Set3')
    employee_df[col_name].value_counts().plot.bar(cmap='Set3',ax=ax[1])
    ax[1].set_title(f'Number of Employee by {col_name}')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(f'{col_name}')
    sns.countplot(col_name, hue='Attrition',data=employee_df, ax=ax[2], palette='Set3')
    ax[2].set_title(f'Attrition by {col_name}')
    ax[2].set_xlabel(f'{col_name}')
    ax[2].set_ylabel('Count')

categorical_colum_investigaton('Gender')
categorical_colum_investigaton('BusinessTravel')
categorical_colum_investigaton('EducationField')
categorical_colum_investigaton('MaritalStatus')
categorical_colum_investigaton('Department')
categorical_colum_investigaton('JobRole')
correlations = employee_df.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, annot=True)
# Job level is strongly correlated with total working hours
# Monthly income is strongly correlated with Job level
# Monthly income is strongly correlated with total working hours
# Age is stongly correlated with monthly income

def categorical_numerical_comperation(numerical_col, caterical_col1, caterical_col2):
    

    f,ax = plt.subplots(1,2, figsize=(18,6))
    
    g1= sns.swarmplot( caterical_col1, numerical_col,hue='Attrition', data=employee_df, dodge=True, ax=ax[0], palette='Set2')
    ax[0].set_title(f'{numerical_col} vs {caterical_col1} separeted by Attrition')
    g1.set_xticklabels(g1.get_xticklabels(), rotation=45) 

    
    g2 = sns.swarmplot( caterical_col2, numerical_col,hue='Attrition', data=employee_df, dodge=True, ax=ax[1], palette='Set2')
    ax[1].set_title(f'{numerical_col} vs {caterical_col1} separeted by Attrition')
    g2.set_xticklabels(g2.get_xticklabels(), rotation=45) 
categorical_numerical_comperation('Age','Gender','MaritalStatus')
categorical_numerical_comperation('Age','JobRole','EducationField')
categorical_numerical_comperation('MonthlyIncome','Gender','MaritalStatus')
categorical_colum_investigaton('Total_Satisfaction')
categorical_colum_investigaton('EnvironmentSatisfaction')
categorical_colum_investigaton('JobInvolvement')
categorical_colum_investigaton('NumCompaniesWorked')
categorical_colum_investigaton('YearsWithCurrManager')
categorical_colum_investigaton('OverTime')
categorical_colum_investigaton('StockOptionLevel')
categorical_colum_investigaton('Education')
categorical_colum_investigaton('TrainingTimesLastYear')
def numerical_colum_investigaton(col_name):
    f,ax = plt.subplots(1,2, figsize=(18,6))
    sns.kdeplot(attrition[col_name], label='Employee who left',ax=ax[0], shade=True, color='palegreen')
    sns.kdeplot(no_attrition[col_name], label='Employee who stayes', ax=ax[0], shade=True, color='salmon')
    
    sns.boxplot(y=col_name, x='Attrition',data=employee_df, palette='Set2', ax=ax[1])
    
numerical_colum_investigaton("Age")
numerical_colum_investigaton('DistanceFromHome')
numerical_colum_investigaton('MonthlyIncome')
numerical_colum_investigaton('HourlyRate')
numerical_colum_investigaton('JobInvolvement')
numerical_colum_investigaton('PercentSalaryHike')
numerical_colum_investigaton('Age')
numerical_colum_investigaton('DailyRate')
numerical_colum_investigaton('TotalWorkingYears')
numerical_colum_investigaton('YearsAtCompany')

employee_df['Total_Satisfaction'] = (employee_df['EnvironmentSatisfaction'] + 
                                     employee_df['JobInvolvement'] + 
                                     employee_df['JobSatisfaction'] + 
                                     employee_df['RelationshipSatisfaction'] +
                                     employee_df['WorkLifeBalance']) /5 

# drop used column and 
employee_df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)
employee_df['Total_Satisfaction_bool'] = employee_df['Total_Satisfaction'].apply(lambda x:1 if x>=2.2 else 0 ) 
employee_df.drop('Total_Satisfaction', axis=1, inplace=True)

#Age Column:Employee more likey the drop the job younger than 35
employee_df['Age_bool'] = employee_df['Age'].apply(lambda x:1 if x<35 else 0)
employee_df.drop('Age', axis=1, inplace=True)

#Daily Rate:Employee more likey the drop the job if dailtRate less than 750
employee_df['DailyRate_bool'] = employee_df['DailyRate'].apply(lambda x:1 if x<750 else 0)
employee_df.drop('DailyRate', axis=1, inplace=True)

#Departman: Employee more likey the drop the job if the employee working at Sales Departmaen
employee_df['Department_bool'] = employee_df['Department'].apply(lambda x:1 if x=='Sales' else 0)
employee_df.drop('Department', axis=1, inplace=True)

# DistanceFromHome: Employee more likey the drop the job if the employeeworking far more than 10
employee_df['DistanceFromHome_bool'] = employee_df['DistanceFromHome'].apply(lambda x:1 if x>10 else 0)
employee_df.drop('DistanceFromHome', axis=1, inplace=True)

# HourlyRate: Employee  more likey the drop the job if the employee working hourly rate less than 65
employee_df['HourlyRate_bool'] = employee_df['HourlyRate'].apply(lambda x:1 if x<65 else 0)
employee_df.drop('HourlyRate', axis=1, inplace=True)

#JobRole: Employee more likey the drop the job if the employee working as Sales Executive 
employee_df['JobRole_bool'] = employee_df['JobRole'].apply(lambda x:1 if x=='Sales Executive' else 0)
employee_df.drop('JobRole', axis=1, inplace=True)

#MontlyIncome:Employee more likey the drop the job if the employee working as Sales Executive 
employee_df['MonthlyIncome_bool'] = employee_df['MonthlyIncome'].apply(lambda x:1 if x<3500 else 0)
employee_df.drop('MonthlyIncome', axis=1, inplace=True)

#NumCompaniesWorked: Employee more likey the drop the job if the employee working as Sales Executive 
employee_df['NumCompaniesWorked_bool'] = employee_df['NumCompaniesWorked'].apply(lambda x:1 if x>4 else 0)
employee_df.drop('NumCompaniesWorked', axis=1, inplace=True)

#TotalWorkingYears: Employee  more likey the drop the job if the employee working as Sales Executive 
employee_df['TotalWorkingYears_bool'] = employee_df['TotalWorkingYears'].apply(lambda x:1 if x<8 else 0)
employee_df.drop('TotalWorkingYears', axis=1, inplace=True)

#YearsAtCompany: Employee more likey the drop the job if the employee working as Sales Executive 
employee_df['YearsAtCompany_bool'] = employee_df['YearsAtCompany'].apply(lambda x:1 if x<3 else 0)
employee_df.drop('YearsAtCompany', axis=1, inplace=True)

#JobRole Column  more likey the drop the job if the employee working as Sales Executive 
employee_df['YearsInCurrentRole_bool'] = employee_df['YearsInCurrentRole'].apply(lambda x:1 if x<3 else 0)
employee_df.drop('YearsInCurrentRole', axis=1, inplace=True)

#JobRole Column  more likey the drop the job if the employee working as Sales Executive 
employee_df['YearsSinceLastPromotion_bool'] = employee_df['YearsSinceLastPromotion'].apply(lambda x:1 if x<1 else 0)
employee_df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

#JobRole Column  more likey the drop the job if the employee working as Sales Executive 
employee_df['YearsWithCurrManager_bool'] = employee_df['YearsWithCurrManager'].apply(lambda x:1 if x<1 else 0)
employee_df.drop('YearsWithCurrManager', axis=1, inplace=True)

employee_df.drop(['MonthlyRate','PercentSalaryHike'], axis=1, inplace=True)



convert_category = ['BusinessTravel','Education','EducationField','MaritalStatus','StockOptionLevel','OverTime','Gender','TrainingTimesLastYear']
for col in convert_category:
        employee_df[col] = employee_df[col].astype('category')

employee_df.info()

#separate the categorical and numerical column
X_categorical = employee_df.select_dtypes(include=['category'])
X_numerical = employee_df.select_dtypes(include=['int64'])

#create teh target column
y = employee_df['Attrition']
X_numerical.drop('Attrition', axis=1, inplace=True)

#handle the categorical variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

X_categorical = onehotencoder.fit_transform(X_categorical).toarray()
X_categorical = pd.DataFrame(X_categorical)
X_categorical
#concat the categorical and numerical values

X_all = pd.concat([X_categorical, X_numerical], axis=1)
X_all.head()
#Scaler the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

print(f"Train data shape: {X_train.shape}, Test Data Shape {X_test.shape}")
#import libraires
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression()

model.fit(X_train, y_train)
model
y_pred = model.predict(X_test)
print(f"Accuracy of Logistic Regression: %{100* accuracy_score(y_pred, y_test)}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#predict probabilities
y_probability = model.predict_proba(X_test)
#keep probabilities positive outcomes only
y_probability = y_probability[:,1]
## calculate scores
logistic_roc_score = roc_auc_score(y_test, y_probability)
print("Logistic ROC AUC:%.3f" %(logistic_roc_score) )

# generate a no skill prediction (majority class)
noskill_prob = [0 for _ in range(len(y_test))]
noskill_roc_auc = roc_auc_score(y_test,noskill_prob )
print("No Skill ROC AUC:%.3f" %(noskill_roc_auc))

#calculate the roc curve
lo_fpr, lo_tpr, _ =roc_curve(y_test, y_probability)
no_fpr, no_tpr, _ =roc_curve(y_test, noskill_prob)

plt.plot(lo_fpr, lo_tpr, linestyle='dashed', label ='Logistic')
plt.plot(no_fpr, no_tpr, linestyle='dotted', label= 'No Skill')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
from sklearn.ensemble import RandomForestClassifier

model_Ran = RandomForestClassifier()

model_Ran.fit(X_train, y_train)

y_pred_Ran = model_Ran.predict(X_test)

print(f"Accuracy of Randon Forest Model: {accuracy_score(y_test, y_pred_Ran)}")

cm_Ran = confusion_matrix(y_test, y_pred_Ran)
sns.heatmap(cm_Ran, annot=True)

print(classification_report(y_test, y_pred_Ran))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction (majority class)
noskill_prob = [0 for _ in range(len(y_test))]
noskill_roc_auc = roc_auc_score(y_test,noskill_prob )
print("No Skill ROC AUC:%.3f" %(noskill_roc_auc))

#predict probabilities
y_probability = model.predict_proba(X_test)
#keep probabilities positive outcomes only
y_probability = y_probability[:,1]
## calculate scores
logistic_roc_score = roc_auc_score(y_test, y_probability)
print("Logistic ROC AUC:%.3f" %(logistic_roc_score) )



#Probabilities for random forest
y_probability_Ran = model_Ran.predict_proba(X_test)
#keep probabilities positive outcomes only
y_probability_Ran = y_probability_Ran[:,1]
## calculate scores
RandomF_roc_score = roc_auc_score(y_test, y_probability_Ran)
print("Random Forest ROC AUC:%.3f" %(RandomF_roc_score) )


#calculate the roc curve
lo_fpr, lo_tpr, _ =roc_curve(y_test, y_probability)
no_fpr, no_tpr, _ =roc_curve(y_test, noskill_prob)
ra_fpr, ra_tpr, _ =roc_curve(y_test, y_probability_Ran)


plt.plot(lo_fpr, lo_tpr, linestyle='dashed', color= 'r', label ='Logistic Regression Model')
plt.plot(ra_fpr, ra_tpr, linestyle='dashed', color= 'b', label= 'Random Forest Model')
plt.plot(no_fpr, no_tpr, linestyle='dotted', label= 'No Skill')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV




kfold = KFold(n_splits=10, random_state=22,shuffle=True)
xyz= []
accuracy = []
std = []
y_pred_list = []


classifiers = ['Linear Svm',
              'Radical Svm',
              'Logistic Regression',
              'KNN',
              'Decision Tree',
              'Naive Bayes',
              'Random Forest',
              'XGBoost']

models = [svm.SVC(kernel='linear'),
         svm.SVC(kernel='rbf'),
         LogisticRegression(solver='liblinear'),
         KNeighborsClassifier(),
          DecisionTreeClassifier(),
          GaussianNB(),
          RandomForestClassifier(n_estimators=100),
          XGBClassifier()
        ]

for i in models:
    model = i 
    cv_result =cross_val_score(model, 
                               X_train, 
                               y_train, 
                               cv=kfold,
                              scoring='accuracy')
    y_pred = cross_val_predict(model, X, y, cv=10)
    y_pred_list.append(y_pred)
    cv_result =cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
    
    
    
    
new_model_data_frame =pd.DataFrame({'Cv Mean': xyz,
                                   'Std': std},
                                   index=classifiers)

new_model_data_frame
plt.subplots(figsize=(12,16))
plt.xticks(rotation=45)
sns.boxplot(new_model_data_frame.index, accuracy)
from sklearn.model_selection import GridSearchCV
C=[0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper = {'kernel':kernel, 'C':C, 'gamma':gamma}
gd =GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)


gd.fit(X_train,y_train)


print(gd.best_score_)
print(gd.best_estimator_)