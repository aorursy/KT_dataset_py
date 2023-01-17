import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
df=pd.read_csv(r"C:\Users\GAURAv.GUPTA\telecom\HR_general_data.csv")
df.head()
df.info()
df.isnull().sum()
df.describe()
cat=df.select_dtypes(exclude=['int32','int64','float64'])
num=df.select_dtypes(include=['int32','int64','float64'])
cat.head()
#Analysis of catgorical data.
sns.countplot(x="Gender", hue="Attrition", data=cat)
fig= plt.figure(figsize=(25,5))
sns.countplot(x="JobRole", hue="Attrition", data=cat)
fig= plt.figure(figsize=(25,5))
sns.countplot(x="Department", hue="Attrition", data=cat)
fig= plt.figure(figsize=(25,5))
sns.countplot(x="MaritalStatus", hue="Attrition", data=cat)

fig= plt.figure(figsize=(25,5))
sns.countplot(x="Over18", hue="Attrition", data=cat)
num.head()
#Remove un used column
num=num.drop(['EmployeeID','EmployeeCount','StandardHours'],axis=1)
num.head()
#Convert categorical data into numrice form

#Label encode and hot encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat = cat.apply(LabelEncoder().fit_transform)
cat.head()
# and remove first column to avoid multicorlnarty
cat=pd.get_dummies(cat, columns=['Gender'],drop_first = True)
cat.head(5)
df=pd.concat([cat,num],axis=1)
df.head()
a=df['NumCompaniesWorked'].mode()
a
df['NumCompaniesWorked'].fillna(1.0, inplace=True)
#Outlier check
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(df)

z=pd.DataFrame(scaler.transform(df),columns=df.columns)
z.boxplot(vert=False,figsize=(15,10))
Q1 = df["YearsWithCurrManager"].quantile(0.25)
Q3 = df["YearsWithCurrManager"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["YearsWithCurrManager"] = np.where(df["YearsWithCurrManager"] >= (Q3 + 1.5 * IQR), Q3, df["YearsWithCurrManager"])
df["YearsWithCurrManager"] = np.where(df["YearsWithCurrManager"] <= (Q1 - 1.5 * IQR), Q1, df["YearsWithCurrManager"])
Q1 = df["YearsSinceLastPromotion"].quantile(0.25)
Q3 = df["YearsSinceLastPromotion"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["YearsSinceLastPromotion"] = np.where(df["YearsSinceLastPromotion"] >= (Q3 + 1.5 * IQR), Q3, df["YearsSinceLastPromotion"])
df["YearsSinceLastPromotion"] = np.where(df["YearsSinceLastPromotion"] <= (Q1 - 1.5 * IQR), Q1, df["YearsSinceLastPromotion"])
Q1 = df["YearsAtCompany"].quantile(0.25)
Q3 = df["YearsAtCompany"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["YearsAtCompany"] = np.where(df["YearsAtCompany"] >= (Q3 + 1.5 * IQR), Q3, df["YearsAtCompany"])
df["YearsAtCompany"] = np.where(df["YearsAtCompany"] <= (Q1 - 1.5 * IQR), Q1, df["YearsAtCompany"])
Q1 = df["TrainingTimesLastYear"].quantile(0.25)
Q3 = df["TrainingTimesLastYear"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["TrainingTimesLastYear"] = np.where(df["TrainingTimesLastYear"] >= (Q3 + 1.5 * IQR), Q3, df["TrainingTimesLastYear"])
df["TrainingTimesLastYear"] = np.where(df["TrainingTimesLastYear"] <= (Q1 - 1.5 * IQR), Q1, df["TrainingTimesLastYear"])
Q1 = df["MonthlyIncome"].quantile(0.25)
Q3 = df["MonthlyIncome"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["MonthlyIncome"] = np.where(df["MonthlyIncome"] >= (Q3 + 1.5 * IQR), Q3, df["MonthlyIncome"])
df["MonthlyIncome"] = np.where(df["MonthlyIncome"] <= (Q1 - 1.5 * IQR), Q1, df["MonthlyIncome"])
Q1 = df["TotalWorkingYears"].quantile(0.25)
Q3 = df["TotalWorkingYears"].quantile(0.75)

IQR = Q3 - Q1
print(IQR)
df["TotalWorkingYears"] = np.where(df["TotalWorkingYears"] >= (Q3 + 1.5 * IQR), Q3, df["TotalWorkingYears"])
df["TotalWorkingYears"] = np.where(df["TotalWorkingYears"] <= (Q1 - 1.5 * IQR), Q1, df["TotalWorkingYears"])
#Outlier check
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(df)

z=pd.DataFrame(scaler.transform(df),columns=df.columns)
z.boxplot(vert=False,figsize=(15,10))
x=df.drop(['Attrition'],axis=1)
y=df['Attrition']
d=x.corr()#Again Check multicollinearity b/w Indenpendent variable
e=d.iloc[:-1,:-1]
threshold = 0.5
important_corrs = (e[abs(e) > threshold][e != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs
x=x.drop(['TotalWorkingYears'],axis=1)
#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_test.shape)
#model=LogisticRegression(random_state=0)
#model.fit(x_train,y_train)
#y_pred=model.predict(x_test)
#print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier
DTREE = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTREE.fit(x_train, y_train)
y_pred=DTREE.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
