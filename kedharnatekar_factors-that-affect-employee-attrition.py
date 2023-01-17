import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mp
# general_data

emp_gen = pd.read_csv('../input/hr-analytics-case-study/general_data.csv')

# Employee Survey data

emp_sur = pd.read_csv('../input/hr-analytics-case-study/employee_survey_data.csv')

# Manager Survey Data

emp_man = pd.read_csv('../input/hr-analytics-case-study/manager_survey_data.csv')

# Merging datasets for general data and Employee Survey

emp1 = pd.merge(emp_gen,emp_sur,on=['EmployeeID'],how='inner')

# Merging the resultant dataset with Manager survey data

emp = pd.merge(emp1,emp_man,on=['EmployeeID'],how='inner')

emp.columns
emp.head(10)
emp.shape
def plot_bar(x):

    pd.Series(emp[x].value_counts()).plot(kind= 'bar')

    plt.title('Plot for '+x+' employee counts')

    plt.xlabel(x)

    plt.ylabel('Employee_counts')
plot_bar('BusinessTravel')
plot_bar('Department')
plot_bar('EducationField')
plot_bar('Gender')
plot_bar('JobRole')
plot_bar('MaritalStatus')
plot_bar('JobLevel')
plot_bar('Education')
plot_bar('StockOptionLevel')
plot_bar('EnvironmentSatisfaction')
plot_bar('JobSatisfaction')
plot_bar('WorkLifeBalance')
plot_bar('JobInvolvement')
plot_bar('PerformanceRating')
emp['Age'].value_counts()
emp['DistanceFromHome'].value_counts()
emp['MonthlyIncome'].value_counts()
emp['NumCompaniesWorked'].value_counts()
emp['PercentSalaryHike'].value_counts()
emp['TotalWorkingYears'].value_counts()
emp['TrainingTimesLastYear'].value_counts()
emp['YearsAtCompany'].value_counts()
emp['YearsSinceLastPromotion'].value_counts()
emp['YearsWithCurrManager'].value_counts()
emp['Over18'].value_counts()
emp['StandardHours'].value_counts()
emp['EmployeeCount'].value_counts()
len(emp['EmployeeID'].unique())
emp.isna().any()
pd.set_option('mode.chained_assignment',None)



emp['NumCompaniesWorked'][emp['NumCompaniesWorked'].isna() == True] = round(emp['NumCompaniesWorked'].mean())

emp['TotalWorkingYears'][emp['TotalWorkingYears'].isna() == True] = round(emp['TotalWorkingYears'].mean())

emp['EnvironmentSatisfaction'][emp['EnvironmentSatisfaction'].isna() == True] = round(emp['EnvironmentSatisfaction'].mean())

emp['JobSatisfaction'][emp['JobSatisfaction'].isna() == True] = round(emp['JobSatisfaction'].mean())

emp['WorkLifeBalance'][emp['WorkLifeBalance'].isna() == True] = round(emp['WorkLifeBalance'].mean())
def box_plot(x):

    f1, p1 = plt.subplots()

    p1.set_title(x)

    p1.boxplot(emp[x])
box_plot('Age')
box_plot('DistanceFromHome')
box_plot('MonthlyIncome')
box_plot('NumCompaniesWorked')
box_plot('PercentSalaryHike')
box_plot('TotalWorkingYears')
box_plot('TrainingTimesLastYear')
box_plot('YearsAtCompany')
box_plot('YearsSinceLastPromotion')
box_plot('YearsWithCurrManager')
emp_ohe = pd.get_dummies(emp, columns=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","JobLevel","Education","StockOptionLevel","EnvironmentSatisfaction","JobSatisfaction","WorkLifeBalance","JobInvolvement","PerformanceRating"], prefix=["BusinessTravel:","Department:","EducationField:","Gender:","JobRole:","MaritalStatus:","JobLevel:","Education:","StockOptionLevel:","EnvironmentSatisfaction:","JobSatisfaction:","WorkLifeBalance:","JobInvolvement:","PerformanceRating:"] ,drop_first = True)
emp_ohe.columns
from sklearn.preprocessing import LabelEncoder

label_encoder_y = LabelEncoder()

emp_ohe['Attrition'] = label_encoder_y.fit_transform(emp_ohe['Attrition'])
emp_ohe.drop(['Over18'],axis =1,inplace=True)

emp_ohe.drop(['EmployeeID'],axis =1,inplace=True)

emp_ohe.drop(['StandardHours'],axis =1,inplace=True)

emp_ohe.drop(['EmployeeCount'],axis =1,inplace=True)
import statsmodels.formula.api as sm



def vif_cal(input_data, dependent_col):

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  

        vif=round(1/(1-rsq),2)

        if vif > 5:

            print (xvar_names[i], " VIF = " , vif )

vif_cal(input_data=emp_ohe, dependent_col="Attrition")
emp_ohe.drop(['EducationField:_Life Sciences'],axis =1,inplace=True)

vif_cal(input_data=emp_ohe, dependent_col="Attrition")
emp_ohe.drop(['Department:_Sales'],axis =1,inplace=True)

vif_cal(input_data=emp_ohe, dependent_col="Attrition")
y = emp_ohe['Attrition']

X = emp_ohe[

["Age"]+ ["DistanceFromHome"]+ ["MonthlyIncome"]+["NumCompaniesWorked"]+ ["PercentSalaryHike"]+ 

["TotalWorkingYears"]+["TrainingTimesLastYear"]+ ["YearsAtCompany"]+ ["YearsSinceLastPromotion"]+

["YearsWithCurrManager"]+ ["BusinessTravel:_Travel_Frequently"]+["BusinessTravel:_Travel_Rarely"]+ 

["Department:_Research & Development"]+["EducationField:_Marketing"]+ ["EducationField:_Medical"]+

["EducationField:_Other"]+ ["EducationField:_Technical Degree"]+["Gender:_Male"]+ 

["JobRole:_Human Resources"]+["JobRole:_Laboratory Technician"]+ ["JobRole:_Manager"]+

["JobRole:_Manufacturing Director"]+ ["JobRole:_Research Director"]+["JobRole:_Research Scientist"]+ 

["JobRole:_Sales Executive"]+["JobRole:_Sales Representative"]+ ["MaritalStatus:_Married"]+

["MaritalStatus:_Single"]+ ["JobLevel:_2"]+ ["JobLevel:_3"]+ ["JobLevel:_4"]+["JobLevel:_5"]+ 

["Education:_2"]+ ["Education:_3"]+ ["Education:_4"]+["Education:_5"]+ ["StockOptionLevel:_1"]+ 

["StockOptionLevel:_2"]+["StockOptionLevel:_3"]+ ["EnvironmentSatisfaction:_2.0"]+["EnvironmentSatisfaction:_3.0"]+ 

["EnvironmentSatisfaction:_4.0"]+["JobSatisfaction:_2.0"]+ ["JobSatisfaction:_3.0"]+ ["JobSatisfaction:_4.0"]+["WorkLifeBalance:_2.0"]+ 

["WorkLifeBalance:_3.0"]+ ["WorkLifeBalance:_4.0"]+["JobInvolvement:_2"]+ ["JobInvolvement:_3"]+ ["JobInvolvement:_4"]+["PerformanceRating:_4"]]
import statsmodels.api as sm

m1=sm.Logit(y,X)

m1.fit()

m1.fit().summary()
X = emp_ohe[

["Age"]+ ["NumCompaniesWorked"]+ ["PercentSalaryHike"]+ 

["TotalWorkingYears"]+["TrainingTimesLastYear"]+ ["YearsAtCompany"]+ ["YearsSinceLastPromotion"]+

["YearsWithCurrManager"]+ ["BusinessTravel:_Travel_Frequently"]+["BusinessTravel:_Travel_Rarely"]+ 

["EducationField:_Marketing"]+ ["EducationField:_Other"]+ ["EducationField:_Technical Degree"]+

["JobRole:_Laboratory Technician"]+ ["JobRole:_Research Director"]+["JobRole:_Research Scientist"]+ 

["JobRole:_Sales Executive"]+ ["MaritalStatus:_Married"]+["MaritalStatus:_Single"]+ ["StockOptionLevel:_1"]+ 

["EnvironmentSatisfaction:_2.0"]+["EnvironmentSatisfaction:_3.0"]+ ["EnvironmentSatisfaction:_4.0"]+["JobSatisfaction:_2.0"]+ 

["JobSatisfaction:_3.0"]+ ["JobSatisfaction:_4.0"]+["WorkLifeBalance:_2.0"]+ ["WorkLifeBalance:_3.0"]+ ["WorkLifeBalance:_4.0"]]



# Model fit with new set of features

m1=sm.Logit(y,X)

m1.fit()

m1.fit().summary()

X = emp_ohe[

["Age"]+ ["NumCompaniesWorked"]+ ["PercentSalaryHike"]+ 

["TotalWorkingYears"]+["TrainingTimesLastYear"]+ ["YearsSinceLastPromotion"]+

["YearsWithCurrManager"]+ ["BusinessTravel:_Travel_Frequently"]+["BusinessTravel:_Travel_Rarely"]+ 

["EducationField:_Other"]+ ["JobRole:_Laboratory Technician"]+ 

["JobRole:_Research Director"]+["JobRole:_Research Scientist"]+ 

["JobRole:_Sales Executive"]+ ["MaritalStatus:_Single"]+ 

["EnvironmentSatisfaction:_2.0"]+["EnvironmentSatisfaction:_3.0"]+ ["EnvironmentSatisfaction:_4.0"]+["JobSatisfaction:_2.0"]+ ["JobSatisfaction:_3.0"]+ ["JobSatisfaction:_4.0"]+["WorkLifeBalance:_2.0"]+ ["WorkLifeBalance:_3.0"]+ ["WorkLifeBalance:_4.0"]]



# Model fit with new set of features

m1=sm.Logit(y,X)

m1.fit()

m1.fit().summary()
X = emp_ohe[

["Age"]+ ["NumCompaniesWorked"]+ ["PercentSalaryHike"]+ 

["TotalWorkingYears"]+["TrainingTimesLastYear"]+ ["YearsSinceLastPromotion"]+

["YearsWithCurrManager"]+ ["BusinessTravel:_Travel_Frequently"]+["BusinessTravel:_Travel_Rarely"]+ 

["JobRole:_Laboratory Technician"]+ ["JobRole:_Research Director"]+["JobRole:_Research Scientist"]+ 

["JobRole:_Sales Executive"]+ ["MaritalStatus:_Single"]+ ["EnvironmentSatisfaction:_2.0"]+["EnvironmentSatisfaction:_3.0"]+ ["EnvironmentSatisfaction:_4.0"]+["JobSatisfaction:_2.0"]+ ["JobSatisfaction:_3.0"]+ ["JobSatisfaction:_4.0"]+

["WorkLifeBalance:_2.0"]+ ["WorkLifeBalance:_3.0"]+ ["WorkLifeBalance:_4.0"]]



# Model fit with new set of features

m1=sm.Logit(y,X)

m1.fit()

m1.fit().summary()
X = emp_ohe[["MaritalStatus:_Single"]+["JobSatisfaction:_4.0"]+ ["BusinessTravel:_Travel_Frequently"]+ ["YearsSinceLastPromotion"]+ ["EnvironmentSatisfaction:_4.0"]+["YearsWithCurrManager"]+["WorkLifeBalance:_3.0"]]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
def get_accuracy(x_val,y_val,i):

    logistic= LogisticRegression(solver='lbfgs' , max_iter=i)

    logistic.fit(x_val,y_val)

    predict1=logistic.predict(x_val)

    cm = confusion_matrix(y_val,predict1)

    total=sum(sum(cm))

    accuracy=(cm[0,0]+cm[1,1])/total

    print(accuracy)
get_accuracy(X_train,y_train,1000)
get_accuracy(X_test,y_test,1000)
pd.crosstab(emp.BusinessTravel,emp.Attrition).plot(kind='bar')

plt.title('Plotting BusinessTravel vs Attrition')

plt.xlabel('BusinessTravel')

plt.ylabel('Count of  Attrition')
def percent_plot(x):

    temp = emp[['Attrition',x]]

    temp['Attrition'] = temp['Attrition'].map({'Yes':1 , 'No':0})

    grouped = temp.groupby(x).sum()

    grouped['Total'] = temp.groupby(x).count()

    row_count = emp.shape[0]

    grouped['Percentage_Attrition'] = grouped['Attrition']*100/grouped['Total']

    pd.Series(grouped['Percentage_Attrition'],index = grouped.index).plot()

    title = 'Plotting '+x+' vs % of Attrition'

    plt.title(title)

    plt.xlabel(x)    

    plt.ylabel('Percentage of  Attrition')
percent_plot('BusinessTravel')
percent_plot('MaritalStatus')
percent_plot('EnvironmentSatisfaction')
percent_plot('JobSatisfaction')
percent_plot('YearsSinceLastPromotion')
percent_plot('YearsWithCurrManager')
emp['YearsWithCurrManager'].value_counts()
percent_plot('WorkLifeBalance')