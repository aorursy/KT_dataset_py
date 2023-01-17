import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
df.head()
df.info()
df.describe()
sns.pairplot(df)
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), annot=True)
plt.figure(figsize=(15,12))

plt.subplot(2,2,1)

sns.distplot(df['GRE Score'], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,2)

sns.distplot(df['TOEFL Score'], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,3)

sns.distplot(df['University Rating'], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,4)

sns.distplot(df['CGPA'], color='Orange')

plt.grid(alpha=0.5)
plt.figure(figsize=(10,6))

sns.countplot(y=df['Research'])

plt.grid(alpha=0.5)

plt.xlabel('Students')

plt.show()
print("Total number of students with Research : ",(df['Research']==1).sum())

print("Total number of students with-out Research : ",len(df)-(df['Research']==1).sum())

print("Percentage of students with Research : ",round(((df['Research']==1).sum()/len(df))*100,2),'%')
plt.figure(figsize=(10,6))

sns.countplot(y=df['University Rating'])

plt.grid(alpha=0.5)

plt.xlabel('Students Count')

plt.show()
plt.figure(figsize=(10,6))

uni_influence = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts()

uni_influence.plot(kind='barh')

plt.grid(alpha=0.5)

plt.xlabel('Student Count')

plt.ylabel('University Rating')

plt.show()
print('From given University Rating each university has a Student count of:')

print('University Rating 1 : ',(df['University Rating']==1).sum())

print('University Rating 2 : ',(df['University Rating']==2).sum())

print('University Rating 3 : ',(df['University Rating']==3).sum())

print('University Rating 4 : ',(df['University Rating']==4).sum())

print('University Rating 5 : ',(df['University Rating']==5).sum())
print('From given University Rating and Student count in each university, number of Students having chance >75% of Admit:')

print('University Rating 1 : ',uni_influence.iloc[4])

print('University Rating 2 : ',uni_influence.iloc[3])

print('University Rating 3 : ',uni_influence.iloc[2])

print('University Rating 4 : ',uni_influence.iloc[0])

print('University Rating 5 : ',uni_influence.iloc[1])
gre_avg = df['GRE Score'].mean()

gre_std = df['GRE Score'].std()

print("Maximum GRE Score : 340")

print("Average GRE Score : ",gre_avg)

print("Standard Deaviation : ",gre_std)



diff = df['GRE Score']-gre_avg

df['SD_GRE'] = diff/gre_std
toefl_avg = df['TOEFL Score'].mean()

toefl_std = df['TOEFL Score'].std()

print("Maximum TOEFL Score : 120")

print("Average TOEFL Score : ",toefl_avg)

print("Standard Deaviation : ",toefl_std)



diff = df['TOEFL Score']-toefl_avg

df['SD_TOEFL'] = diff/toefl_std
cgpa_avg = df['CGPA'].mean()

cgpa_std = df['CGPA'].std()

print("Maximum CGPA Score : 10")

print("Average CGPA Score : ",cgpa_avg)

print("Standard Deaviation : ",cgpa_std)



diff = df['CGPA']-cgpa_avg

df['SD_CGPA'] = diff/cgpa_std
df.head()
sns.pairplot(df, x_vars=['GRE Score','TOEFL Score','CGPA','SD_GRE','SD_TOEFL','SD_CGPA'], y_vars='Chance of Admit')
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(), annot=True)
x = df.drop(['Chance of Admit'], axis=1)

y = df['Chance of Admit']
x.info()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
coef = pd.DataFrame(lr.coef_, x_test.columns, columns = ['Co-efficient'])
coef
y_pred_mlr = lr.predict(x_test)
len(x_test)
fig = plt.figure()

c = [i for i in range(1,101,1)]

plt.plot(c,y_test, color = 'green', linewidth = 2.5, label='Test')

plt.plot(c,y_pred_mlr, color = 'orange', linewidth = 2.5, label='Predicted')

plt.grid(alpha = 0.3)

plt.legend()

fig.suptitle('Actual vs Predicted')
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred_mlr)

r_square_score = r2_score(y_test, y_pred_mlr)
print('Mean Square Error = ',mse)

print('R_Square Score = ',r_square_score)
fig = plt.figure()

plt.plot(c,y_test-y_pred_mlr, color = 'orange', linewidth = 2.5)

plt.grid(alpha = 0.3)

fig.suptitle('Error Terms')
import statsmodels.api as sm
x_train_sm = x_train

x_train_sm = sm.add_constant(x_train_sm)

lml = sm.OLS(y_train, x_train_sm).fit()

lml.params
print(lml.summary())
x_new = df.drop(['Serial No.','University Rating','SOP','Chance of Admit'], axis=1)

y_new = df['Chance of Admit']
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new,y_new, train_size = 0.7, random_state = 100)
lr.fit(x_train_new, y_train_new)
y_pred_new = lr.predict(x_test_new)
len(x_test_new)
# Actual vs Predicted after removing GRE

fig = plt.figure()

c = [i for i in range(1,121,1)]

plt.plot(c,y_test_new, color = 'green', linewidth = 2.5, label='Test')

plt.plot(c,y_pred_new, color = 'orange', linewidth = 2.5, label='Predicted')

plt.grid(alpha = 0.3)

plt.legend()

fig.suptitle('Actual vs Predicted')
mse_new = mean_squared_error(y_test_new, y_pred_new)

r_square_score_new = r2_score(y_test_new, y_pred_new)

print('Mean Square Error = ',mse_new)

print('R_Square Score = ',r_square_score_new)
fig = plt.figure()

plt.plot(c,y_test_new-y_pred_new, color = 'orange', linewidth = 2.5)

plt.grid(alpha = 0.3)

fig.suptitle('Error Terms')
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
y_train_label = [1 if each > 0.8 else 0 for each in y_train]

y_test_label  = [1 if each > 0.8 else 0 for each in y_test]
logmodel.fit(x_train, y_train_label)
y_pred_log = logmodel.predict(x_test)
from sklearn.metrics import classification_report

print(classification_report(y_test_label, y_pred_log))
from sklearn.metrics import confusion_matrix

cm_log = confusion_matrix(y_test_label, y_pred_log)
sns.heatmap(cm_log, annot=True)

plt.xlabel("Predicted")

plt.ylabel("Actual")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_log))

print("precision_score: ", precision_score(y_test_label,logmodel.predict(x_test)))

print("recall_score: ", recall_score(y_test_label,logmodel.predict(x_test)))

print("f1_score: ",f1_score(y_test_label,logmodel.predict(x_test)))
from sklearn.svm import SVC

svmmodel = SVC()
svmmodel.fit(x_train,y_train_label)
y_pred_svm = svmmodel.predict(x_test)
from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_test_label, svmmodel.predict(x_test))
sns.heatmap(cm_svm, annot=True)

plt.xlabel("Predicted")

plt.ylabel("Actual")
from sklearn.metrics import precision_score, recall_score, f1_score

print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_svm))

print("precision_score: ", precision_score(y_test_label,svmmodel.predict(x_test)))

print("recall_score: ", recall_score(y_test_label,svmmodel.predict(x_test)))

print("f1_score: ",f1_score(y_test_label,svmmodel.predict(x_test)))
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor()

dt_model.fit(x_train,y_train)
y_pred_dt = dt_model.predict(x_test)
from sklearn.metrics import r2_score

print('R_Squared Score = ',r2_score(y_test, y_pred_dt))
from sklearn.preprocessing import StandardScaler 

from sklearn.neighbors import KNeighborsClassifier
import math

math.sqrt(len(y_test_label))
knnc = KNeighborsClassifier(n_neighbors = 11, p=2, metric = 'euclidean')
knnc.fit(x_train, y_train_label)
y_pred_knn = knnc.predict(x_test)
y_pred_knn
cm = confusion_matrix(y_test_label, y_pred_knn)

sns.heatmap(cm, annot=True)

plt.xlabel("Predicted")

plt.ylabel("Actual")
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

print("Accuracy Score = ",accuracy_score(y_test_label, y_pred_knn))

print("precision_score: ", precision_score(y_test_label,knnc.predict(x_test)))

print("recall_score: ", recall_score(y_test_label,knnc.predict(x_test)))

print("f1_score: ",f1_score(y_test_label,knnc.predict(x_test)))
x = ["Linear_Reg","Decision_Tree_Reg"]

y = np.array([r2_score(y_test,y_pred_mlr),r2_score(y_test,y_pred_dt)])

plt.barh(x,y, color='#225b46')

plt.xlabel("R_Squared_Score")

plt.ylabel("Regression Models")

plt.title("Best R_Squared Score")

plt.grid(alpha=0.5)

plt.show()
x = ["KNN","SVM","Logistic_Reg"]

y = np.array([accuracy_score(y_test_label, y_pred_knn),accuracy_score(y_test_label, y_pred_svm),accuracy_score(y_test_label, y_pred_log)])

plt.barh(x,y, color='#225b46')

plt.xlabel("Accuracy Score")

plt.ylabel("Classification Models")

plt.title("Best Accuracy Score")

plt.grid(alpha=0.5)

plt.show()