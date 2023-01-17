# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas_profiling as pp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from mlxtend.classifier import StackingCVClassifier
#Reading the dataset
df = pd.read_csv('/kaggle/input/heartdisease-dataset/processed.cleveland.csv')
df.info()
df.head()
df.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']
df.head()
#Finding the special characters in the data frame
df.isin(['?']).sum(axis=0)
#Replacing the special character to nan and then drop the columns
df['MajorVessels'] = df['MajorVessels'].replace('?',np.nan)
df['Thalassemia'] = df['Thalassemia'].replace('?',np.nan)
#Dropping the NaN rows now 
df.dropna(how='any',inplace=True)
bg_color = (0.25, 0.25, 0.25)
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,
            "text.color":"white",
            "xtick.color":"white",
            "ytick.color":"white",
            "axes.labelcolor":"white",
            "axes.grid":False,
            'axes.labelsize':25,
            'figure.figsize':(10.0,5.0),
            'xtick.labelsize':15,
            'ytick.labelsize':10})
plt.figure(figsize=(12,12))
correlation_matrix = df.corr()

ax = sns.heatmap(
    correlation_matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(30, 150, n=500),
    square=True,
    annot=True,
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_title("Correlation Plot");
df.nunique()
y = df["Target"]
X = df.drop('Target',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)
print(y_test.unique())
Counter(y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
score_LR = LR.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print('The accuracy of the Logistic Regression model is', score_LR)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
# Support Vector Classifier (SVM/SVC)
svc = SVC(gamma=0.22)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
score_svc = svc.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print('The accuracy of SVC model is', score_svc)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
# Random Forest Classifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
score_RF = RF.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print('The accuracy of the Random Forest Model is', score_RF)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)
score_DT = DT.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print("The accuracy of the Decision tree model is ",score_DT)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
# Gaussian Naive Bayes
GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred = GNB.predict(X_test)
score_GNB = GNB.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print('The accuracy of Gaussian Naive Bayes model is', score_GNB)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
score_knn = knn.score(X_test,y_test)
lr_conf_matrix = confusion_matrix(y_test, y_pred)
print("confussion matrix")
print(lr_conf_matrix)
print('The accuracy of the KNN Model is',score_knn)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True);
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [score_LR*100,
                    score_GNB*100,score_RF*100,score_knn*100,score_DT*100,score_svc*100]})
model_ev
typical_angina_cp = [k for k in df['ChestPain'] if k ==0]
atypical_angina_cp = [k for k in df['ChestPain'] if k ==1]
non_anginal_cp = [k for k in df['ChestPain'] if k ==2]
none_cp = [k for k in df['ChestPain'] if k ==3]

typical_angina_cp_total = len(typical_angina_cp)*100/len(df)
atypical_angina_cp_total = len(atypical_angina_cp)*100/len(df)
non_anginal_cp_total = len(non_anginal_cp)*100/len(df)
none_cp_total = len(none_cp)*100/len(df)

labels=['Typical angina','Atypical angina','Non-anginal','Asymptomatic']
values = [typical_angina_cp_total,atypical_angina_cp_total,non_anginal_cp_total,none_cp_total]

plt.pie(values,labels=labels,autopct='%1.1f%%')

plt.title("Chest Pain Type Percentage")    
plt.show()
heart=[]
for k in df['Target']:
    if k == 0:
        heart.append('Healthy Heart')
    elif k == 1:
        heart.append('Heart Disease_1')
    elif k == 2:
        heart.append("Heart Disease_2")
    elif k == 3:
        heart.append("Heart Disease_3")
    elif k == 4:
        heart.append("Heart Disease_4")
ax = sns.countplot(x='Gender',hue=heart,data=df,palette='mako_r')

plt.title("Heart-Health Vs Gender")    
plt.ylabel("")
plt.yticks([])
plt.xlabel("")

for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))
ax.set_xticklabels(['Male','Female']);
plt.title("Heart-Health Vs Chest Pain Type")
    
ax = sns.countplot(x='ChestPain',hue=heart,data=df,palette='Set1')

plt.ylabel("")
plt.yticks([])
plt.xlabel("")
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))
age_group=[]
for k in df['Age']:
    if (k >=29) & (k<40):
        age_group.append(0)
    elif (k >=40)&(k<55):
        age_group.append(1)
    else:
        age_group.append(2)
df['Age-Group'] = age_group
plt.title("Heart-Health Vs Age group")
ax = sns.countplot(x=age_group,hue=heart,palette='bwr')

plt.ylabel("")
plt.yticks([])
plt.xlabel("")
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))
    
ax.set_xticklabels(['Young (29-40)','Mid-Age(40-55)','Old-Age(>55)']);
ax = sns.countplot(x='FastingBloodSugar',data=df)
plt.title("Fasting Blood Sugar")
plt.ylabel("")
plt.yticks([])
plt.xlabel("")

for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))
ax.set_xticklabels(["Fasting Blood Sugar < 120 mg/dl","Fasting Blood Sugar > 120 mg/dl"]);
serum_chol=[]
for k in df['Cholestrol']:
    if k > 200:
        serum_chol.append(1) #not healthy
    else:
        serum_chol.append(0) #healthy

ax = sns.countplot(x=serum_chol,palette='bwr')

plt.title("Serum Cholestrol")
plt.ylabel("")
plt.yticks([])
plt.xlabel("")

for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))
ax.set_xticklabels(["Serum Cholestrol > 200 mg/dL","Serum Cholestrol < 200 mg/dL"]);
bp=[]
for k in df['RestingBloodPressure']:
    if (k > 130):
        bp.append(1) #high bp
    else:
        bp.append(0) #normal

ax = sns.countplot(x=bp,palette='Set3')

plt.title("Resting Blood Pressure Count")
plt.ylabel("")
plt.yticks([])
plt.xlabel("")

for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))
    
ax.set_xticklabels(["Normal BP","Abnormal BP"]);
(df[
     (df['Age'] > 40) & 
     (df['ChestPain'] == 0) &
     (df['Cholestrol'] >=250) &
     (df['RestingBloodPressure'] > 120) &
     (df['Thalassemia']==2) &
     (df['RestingECG']==1) &
     (df['ExerciseIndusedAngina']==0) &
    (df['MaxHeartRateAchivied']>100)]
    )