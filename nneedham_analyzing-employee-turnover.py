import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['figure.figsize']=(30,15)
plt.style.use('fivethirtyeight')

df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.shape
df.columns
df.head()
df.describe()
df = df.drop(['EmployeeCount','EmployeeNumber', 'Over18','StandardHours'], axis = 1)
df_num = df[['Age',
             'DailyRate',
             'DistanceFromHome',
             'HourlyRate',
             'MonthlyIncome',
             'MonthlyRate',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager']]
df_num.hist()
sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_num.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()
df_cat = df[['Attrition',
             'BusinessTravel',
             'Department',
             'Education',
             'EducationField',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobRole',
             'JobSatisfaction',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'RelationshipSatisfaction',
             ]]
sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_cat.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()
for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.show()
df_cat['AttritYes'] = df_cat['Attrition'].apply(lambda x: 1 if x =='Yes' else 0)
df_cat['AttritNo'] = df_cat['Attrition'].apply(lambda x: 1 if x =='No' else 0)

p_columns = ['BusinessTravel',
             'Department',
             'Education',
             'EducationField',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobRole',
             'JobSatisfaction',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'RelationshipSatisfaction']

for i in p_columns:
    m = df_cat.pivot_table(columns=i, values = ['AttritYes','AttritNo'], aggfunc=np.sum)
    m.loc['PercentAttrit'] = 0
    for a in m:
        m.loc['PercentAttrit'][a] = ((m[a][1])/(m[a][0]+m[a][1]))*100
    print(m)
    print("")
df_model = df[['Attrition',
              'BusinessTravel',
              'Department',
              'Education',
              'EducationField',
              'EnvironmentSatisfaction',
              'JobInvolvement',
              'JobLevel',
              'JobRole',
              'JobSatisfaction',
               'MaritalStatus',
              'OverTime',
              'PerformanceRating',
               'RelationshipSatisfaction',
              ]]
df_dum = pd.get_dummies(df_model)
from sklearn.model_selection import train_test_split
X =df_dum.drop(['Attrition_Yes', 'Attrition_No'], axis=1) 
y = df_dum.Attrition_Yes.values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_score = lda.score(X_test, y_test)
print('Linear Discriminant accuracy: ', lda_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
pred_lda = lda.predict(X_test)
lda_score = lda.score(X_test, y_test)
print('Linear Discriminant accuracy: ', lda_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
s_vm = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
s_vm.fit(X_train, y_train)
s_vm_score = s_vm.score(X_test, y_test)
print('Support Vector Machine accuracy: ', s_vm_score)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10, random_state=0)
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)
print('Random Forest accuracy: ', rfc_score)
from sklearn.metrics import classification_report
pred_s_vm = s_vm.predict(X_test)
print(classification_report(y_test, pred_s_vm))
