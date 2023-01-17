import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
sns.set()
data=pd.read_csv('empatt.csv')
data.head()
data.describe(include='all')
sns.distplot(data['Age'])
sns.distplot(data['DailyRate'])
sns.distplot(data['DistanceFromHome'])
sns.distplot(data['Education'])
sns.distplot(data['RelationshipSatisfaction'])
sns.distplot(data['StockOptionLevel'])
sns.distplot(data['TotalWorkingYears'])
sns.distplot(data['TrainingTimesLastYear'])
sns.distplot(data['WorkLifeBalance'])
sns.distplot(data['YearsAtCompany'])
sns.distplot(data['YearsInCurrentRole'])
sns.distplot(data['YearsSinceLastPromotion'])
sns.distplot(data['YearsWithCurrManager'])
q = data['TotalWorkingYears'].quantile(0.98)
data1=data[data['TotalWorkingYears']<q]
sns.distplot(data1['TotalWorkingYears'])
r = data1['YearsAtCompany'].quantile(0.98)
data2=data1[data1['YearsAtCompany']<r]
sns.distplot(data2['YearsAtCompany'])
s = data2['YearsSinceLastPromotion'].quantile(0.98)
data3=data2[data2['YearsSinceLastPromotion']<s]
sns.distplot(data3['YearsSinceLastPromotion'])
data3.head()
data3.keys()
attrition = {'Yes': 0, 'No': 1}
data3.Attrition = [attrition[item] for item in data3.Attrition]
businesstravel = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
data3.BusinessTravel = [businesstravel[item] for item in data3.BusinessTravel]
data3.Department.unique()
department = {'Sales':0,'Research & Development':1,'Human Resources':2}

data3.Department = [department[item] for item in data3.Department]
data3.EducationField.unique()
educationfield = {'Life Sciences':0,'Other':1,'Medical':2,'Marketing':3,'Technical Degree':4,'Human Resources':5}

data3.EducationField = [educationfield[item] for item in data3.EducationField]
gender = {'Female':0,'Male':1}

data3.Gender = [gender[item] for item in data3.Gender]
data3.JobRole.unique()
jobrole = {'Sales Executive' :0, 'Research Scientist':1, 'Laboratory Technician':2,
       'Manufacturing Director':3, 'Healthcare Representative':4,
           'Sales Representative':5, 'Research Director':6, 'Manager':7,'Human Resources':8}

data3.JobRole = [jobrole[item] for item in data3.JobRole]
maritalstatus = {'Single':0,'Married':1,'Divorced':3}

data3.MaritalStatus = [maritalstatus[item] for item in data3.MaritalStatus]
overtime = {'Yes':0,'No':1}

data3.OverTime = [overtime[item] for item in data3.OverTime]
data3.head()
drop_elements = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']

data4 = data3.drop(drop_elements,axis=1)
data4.head()
data5 = data4.drop('Attrition',axis=1)
bestfeatures = SelectKBest(score_func=chi2,k=30)
fit = bestfeatures.fit(data5.iloc[:,0:30],data4.iloc[:,1])
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data5.iloc[:,0:30].columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns=['Specs','Score']
print(featureScores.nlargest(30,'Score'))
plt.rcParams['figure.figsize']=(12,8)
model = ExtraTreesClassifier()
model.fit(data5.iloc[:,0:30],data4.iloc[:,1])
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_,index=data5.iloc[:,0:30].columns)
feat_imp.nlargest(30).plot(kind='barh')
plt.show()
corrmat = data4.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(24,30))
g=sns.heatmap(data4[top_corr_features].corr(),annot=True)
train, test = sklearn.model_selection.train_test_split(data4,test_size=0.25)
print("Training size:",len(train))
print("Test size:",len(test))
full_data = [train,test]
features = list(data5.columns)
x_train=train[features]
y_train=train["Attrition"]

x_test=test[features]
y_test=test["Attrition"] 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}")
        print("_______________________________________________")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=1200, 
#                                      bootstrap=False,
#                                      class_weight={0:stay, 1:leave}
                                    )
rand_forest.fit(x_train, y_train)

print_score(rand_forest, x_train, y_train, x_test, y_test, train=True)
print_score(rand_forest, x_train, y_train, x_test, y_test, train=False)
