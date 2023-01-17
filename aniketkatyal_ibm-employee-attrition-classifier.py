import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
#printing the first 5 rows

df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()
#checking for null values in the dataset

df.isnull().sum()

#No null values in the dataset
#datatypes of all the fields in the dataset

df.dtypes
correlation_df = df.corr()

# The below correlation coefficients is NaN for Employee Count and Standard Hours Fields

#This may be because of the zero variance in those fields

employee_count_var = df["EmployeeCount"].var() #this is 0

standard_hours_var = df["StandardHours"].var() #this is 0
#Hence we drop these 2 rows

new_df = df.drop(["EmployeeCount","StandardHours"],axis = 1)
new_df.head()
correlation_new_df = new_df.corr()

correlation_new_df
#The above given matrix can also be drawn on MatplotLib for better visual Interpretation



sns.set()

f, ax = plt.subplots(figsize=(10, 8))

corr = new_df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.show()
#After removing the strongly correlated variables

df_numerical = new_df[['Age','DailyRate','DistanceFromHome','Education',

                       'EnvironmentSatisfaction', 'HourlyRate',                     

                       'JobInvolvement', 'JobLevel','MonthlyRate',

                       'JobSatisfaction',

                       'RelationshipSatisfaction', 

                       'StockOptionLevel',

                        'TrainingTimesLastYear','WorkLifeBalance']].copy()

df_numerical.head()
df_numerical = abs(df_numerical - df_numerical.mean())/df_numerical.std()  

df_numerical.head()
df_categorical = new_df[['Attrition', 'BusinessTravel','Department',

                       'EducationField','Gender','JobRole',

                       'MaritalStatus',

                       'Over18', 'OverTime']].copy()

df_categorical.head()
df_categorical["Over18"].value_counts()

#Since all values are Y, we can drop this column
df_categorical = df_categorical.drop(["Over18"],axis = 1)
# We now Label Encode the Attrition data 

lbl = LabelEncoder()

lbl.fit(['Yes','No'])

df_categorical["Attrition"] = lbl.transform(df_categorical["Attrition"])

df_categorical.head()
# We create dummies for the remaining categorical variables



df_categorical = pd.get_dummies(df_categorical)

df_categorical.head()
#Now we finally join both the numerical and categorical dataframes for model evaluation



final_df = pd.concat([df_numerical,df_categorical], axis= 1)

final_df.head()

X = final_df.drop(['Attrition'],axis= 1)

y = final_df["Attrition"]





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 4)
lr = LogisticRegression(solver = 'liblinear',random_state = 0) #Since this a small dataset, we use liblinear solver and Regularization strength as

# default i.e C = 1.0

lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
accuracy_score_lr = accuracy_score(y_pred_lr,y_test)

accuracy_score_lr 

#Logistic Regression shows 85.7 percent accuracy

dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state = 0)
dtree.fit(X_train,y_train)
y_pred_dtree = dtree.predict(X_test)

accuracy_score_dtree = accuracy_score(y_pred_dtree,y_test)

accuracy_score_dtree
rf = RandomForestClassifier(criterion = 'gini',random_state = 0)

rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

accuracy_score_rf = accuracy_score(y_pred_rf,y_test)

accuracy_score_rf
sv = svm.SVC(kernel= 'linear',gamma =2)

sv.fit(X_train,y_train)
y_pred_svm = sv.predict(X_test)

accuracy_score_svm = accuracy_score(y_pred_svm,y_test)

accuracy_score_svm
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

accuracy_score_knn = accuracy_score(y_pred_knn,y_test)

accuracy_score_knn
scores = [accuracy_score_lr,accuracy_score_dtree,accuracy_score_rf,accuracy_score_svm,accuracy_score_knn]

scores = [i*100 for i in scores]

algorithm  = ['Logistic Regression','Decision Tree','Random Forest','SVM', 'K-Means']

index = np.arange(len(algorithm))

plt.bar(index, scores)

plt.xlabel('Algorithm', fontsize=10)

plt.ylabel('Accuracy Score', fontsize=5)

plt.xticks(index, algorithm, fontsize=10, rotation=30)

plt.title('Accuracy scores for each classification algorithm')

plt.ylim(80,100)

plt.show()    
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)

feat_importances = feat_importances.nlargest(20)

feat_importances.plot(kind='barh')

plt.show()