import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
heart_data = pd.read_csv('../input/heart.csv')

heart_data.head()
heart_data.isnull().sum().any()
heart_data.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',

       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']
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

correlation_matrix = heart_data.corr()



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
heart_data.nunique()
typical_angina_cp = [k for k in heart_data['ChestPain'] if k ==0]

atypical_angina_cp = [k for k in heart_data['ChestPain'] if k ==1]

non_anginal_cp = [k for k in heart_data['ChestPain'] if k ==2]

none_cp = [k for k in heart_data['ChestPain'] if k ==3]



typical_angina_cp_total = len(typical_angina_cp)*100/len(heart_data)

atypical_angina_cp_total = len(atypical_angina_cp)*100/len(heart_data)

non_anginal_cp_total = len(non_anginal_cp)*100/len(heart_data)

none_cp_total = len(none_cp)*100/len(heart_data)



labels=['Typical angina','Atypical angina','Non-anginal','Asymptomatic']

values = [typical_angina_cp_total,atypical_angina_cp_total,non_anginal_cp_total,none_cp_total]



plt.pie(values,labels=labels,autopct='%1.1f%%')



plt.title("Chest Pain Type Percentage")    

plt.show()
result=[]

for k in heart_data['ChestPain']:

    if k == 0:

        result.append('Typical Angina')

    elif k == 1:

        result.append('Atypical Angina')

    elif k == 2:

        result.append('Non-Anginal')

    elif k == 3:

        result.append('Asymptomatic')



heart_data['Chest Pain Type'] = result



ax = sns.countplot(hue=result,x='Gender',data=heart_data,palette='husl')



plt.title("Chest Pain Type Vs Gender")    

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.05, p.get_height()+1))



ax.set_xticklabels(['Female','Male']);
age = []

for i in range(len(heart_data)):    

    if heart_data['ChestPain'][i] == 0:        

        age.append(heart_data['Age'][i])    



sns.swarmplot(age)

plt.title("Chest Pain across different Age Groups")    

plt.xlabel("Age")

plt.ylabel("")

plt.yticks([]);
heart_health=[]

for k in heart_data['Target']:

    if k == 0:

        heart_health.append('Healthy Heart')

    elif k == 1:

        heart_health.append('Heart Disease')
ax = sns.countplot(x='Gender',hue=heart_health,data=heart_data,palette='mako_r')



plt.title("Heart-Health Vs Gender")    

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))

ax.set_xticklabels(['Female','Male']);
f = ((heart_data['Gender'] == 0) & (heart_data['Target'] == 1)).sum()

fd =  (heart_data['Target'] == 1).sum()

print('Percent of Female having heart disease : {:.2f}%'.format(f/fd * 100))



m = ((heart_data['Gender'] == 1) & (heart_data['Target'] == 1)).sum()

md =  (heart_data['Target'] == 1).sum()

print('Percent of Male having heart disease : {:.2f}%'.format(m/md * 100))
plt.title("Heart-Health Vs Chest Pain Type")

    

ax = sns.countplot(x='Chest Pain Type',hue=heart_health,data=heart_data,palette='Set1')



plt.ylabel("")

plt.yticks([])

plt.xlabel("")

for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))
age_group=[]

for k in heart_data['Age']:

    if (k >=29) & (k<40):

        age_group.append(0)

    elif (k >=40)&(k<55):

        age_group.append(1)

    else:

        age_group.append(2)

heart_data['Age-Group'] = age_group

plt.title("Heart-Health Vs Age group")

ax = sns.countplot(x=age_group,hue=heart_health,palette='bwr')



plt.ylabel("")

plt.yticks([])

plt.xlabel("")

for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))

    

ax.set_xticklabels(['Young (29-40)','Mid-Age(40-55)','Old-Age(>55)']);
y = ((heart_data['Age-Group'] == 0) & (heart_data['Target'] == 1)).sum()

yd =  (heart_data['Target'] == 1).sum()

print('Percent of Youth having heart disease   : {:.2f}%'.format(y/yd * 100))



m = ((heart_data['Age-Group'] == 1) & (heart_data['Target'] == 1)).sum()

md =  (heart_data['Target'] == 1).sum()

print('Percent of Mid-Age having heart disease : {:.2f}%'.format(m/md * 100))



o = ((heart_data['Age-Group'] == 2) & (heart_data['Target'] == 1)).sum()

od =  (heart_data['Target'] == 1).sum()

print('Percent of Old-Age having heart disease : {:.2f}%'.format(o/od * 100))

ax = sns.countplot(x='FastingBloodSugar',data=heart_data)

plt.title("Fasting Blood Sugar")

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))

ax.set_xticklabels(["Fasting Blood Sugar < 120 mg/dl","Fasting Blood Sugar > 120 mg/dl"]);
serum_chol=[]

for k in heart_data['Cholestrol']:

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

for k in heart_data['RestingBloodPressure']:

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
(heart_data[

     (heart_data['Age'] > 40) & 

     (heart_data['ChestPain'] == 0) &

     (heart_data['Cholestrol'] >=250) &

     (heart_data['RestingBloodPressure'] > 120) &

     (heart_data['Thalassemia']==2) &

     (heart_data['RestingECG']==1) &

     (heart_data['ExerciseIndusedAngina']==0) &

    (heart_data['MaxHeartRateAchivied']>100)]

    )
X_data = heart_data.drop(columns=['Chest Pain Type','Age-Group','Target'], axis=1)

Y = heart_data['Target']



#normalize the data

Y = ((Y - np.min(Y))/ (np.max(Y) - np.min(Y))).values

X = ((X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))).values



x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)
lr = LogisticRegression()

lr.fit(x_train,y_train)

lr_pred = lr.predict(x_test)



lr_accuracy = accuracy_score(y_test, lr_pred)

print('Logistic Regression Accuracy: {:.2f}%'.format(lr_accuracy*100))



cm = confusion_matrix(y_test,lr_pred)

sns.heatmap(cm,annot=True);
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 3)



rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)

rfc_accuracy = accuracy_score(y_test, rfc_pred)

 

print('Random Forest Classifier Accuracy: {:.2f}%'.format(rfc_accuracy*100))

cm = confusion_matrix(y_test,rfc_pred)

sns.heatmap(cm,annot=True);
knn = KNeighborsClassifier(n_neighbors = 4)  # n_neighbors means k

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print('KNeighborsClassifier Accuracy: {:.2f}%'.format(knn_accuracy*100))



cm = confusion_matrix(y_test,knn_pred)

sns.heatmap(cm,annot=True);
svm = SVC(random_state = 1)

svm.fit(x_train, y_train)

svm_pred = svm.predict(x_test)

svm_accuracy = accuracy_score(y_test, svm_pred)



print('SVM Accuracy: {:.2f}%'.format(svm_accuracy*100))



cm = confusion_matrix(y_test,svm_pred)

sns.heatmap(cm,annot=True);
nb = GaussianNB()

nb.fit(x_train, y_train)

nb_pred = nb.predict(x_test)

nb_accuracy = accuracy_score(y_test, nb_pred)



print('Naive Bayes Accuracy: {:.2f}%'.format(nb_accuracy*100))

cm = confusion_matrix(y_test,nb_pred)

sns.heatmap(cm,annot=True);
models = ['Logistic Regression','KNeighborsClassifier','Random Forest Classifier','Naive Bayes','SVM']

accuracy = [lr_accuracy,knn_accuracy,rfc_accuracy,nb_accuracy,svm_accuracy]



ax = sns.barplot(models,accuracy)



plt.title("Accuracy of differet models")



ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right');