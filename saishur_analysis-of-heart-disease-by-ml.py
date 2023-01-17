# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
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



heart= pd.read_csv('/kaggle/input/heart-disease-dataset/heart.csv')

print(heart.shape)

heart.head()



heart.info()
print('unique entries in columns')

heart.nunique()
heart.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',

       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']
heart.head()
import seaborn as sns

sns.countplot(x='Age',data=heart)

plt.show()
sns.swarmplot(heart['Age'])
result=[]

for i in heart['ChestPain']:

    if i == 0:

        result.append('Typical Angina')

    if i ==1:

        result.append('Atypical Angina')

    if i ==2:

        result.append('Non-Anginal')

    if i==3:

        result.append('Asymptomatic')

        

heart['ChestPainType']=pd.Series(result)



sns.swarmplot(x='ChestPainType', y='Age', data=heart)
ax=sns.countplot(hue=result,x='MajorVessels',data=heart,palette='husl')
ChestPain=(heart['ChestPainType']).value_counts()

percent_typAng= ChestPain[0] *100/ len(heart)

percent_AtypAng=ChestPain[1]*100/len(heart)

percent_nonAng=ChestPain[2]*100/len(heart)

percent_none=ChestPain[3]*100/len(heart)



values= [percent_typAng, percent_AtypAng, percent_nonAng, percent_none]

labels=['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic']

plt.pie(values, labels=labels,autopct='%1.1f%%')

plt.title("Chest Pain Type Percentage")    

plt.show()
import matplotlib.pyplot as plt

ax = sns.countplot(hue=result,x='Gender',data=heart,palette='husl')



plt.title("Chest Pain Type Vs Gender")    

plt.ylabel("")

plt.yticks([])

plt.xlabel("")

for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.05, p.get_height()+1))

ax.set_xticklabels(['Female','Male'])

print(ax.patches)
ax = sns.regplot(x='RestingBloodPressure', y='Cholestrol',data=heart, color="g")
import itertools

columns=heart.columns[:14]

plt.subplots(figsize=(28,25))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    heart[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
heartAT=heart[heart['Target']==1]

columns=heart.columns[:13]

plt.subplots(figsize=(28,25))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    heartAT[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
sns.pairplot(data=heart,hue='Target',diag_kind='kde')

plt.show()
heart.isnull().sum()
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
X_data = heart.drop(columns=['ChestPainType','Age','Target'], axis=1)

Y = heart['Target']



#normalize the data

Y = ((Y - np.min(Y))/ (np.max(Y) - np.min(Y))).values

X = ((X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))).values



x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)
types=['rbf','linear']

for i in types:

    model=svm.SVC(kernel=i)

    model.fit(x_train,y_train)

    svm_prediction=model.predict(x_test)

    print('Accuracy for SVM kernel=',i,'is',metrics.accuracy_score(svm_prediction,y_test))
model = LogisticRegression()

model.fit(x_train,y_train)

lr_prediction=model.predict(x_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(lr_prediction,y_test))
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=39, n_estimators=50)

model.fit(x_train, y_train)

gb_pred = model.predict(x_test)

accuracy = np.mean(gb_pred == y_test)

print('accuracy: ', accuracy*100, '%')
model=DecisionTreeClassifier()

model.fit(x_train, y_train)

dt_prediction=model.predict(x_test)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction,y_test))

a_index=list(range(1,11))

a=pd.Series()

x=[0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(x_train,y_train)

    kn_prediction=model.predict(x_test)

    a=a.append(pd.Series(metrics.accuracy_score(kn_prediction,y_test)))

plt.plot(a_index, a)

plt.xticks(x)

plt.show()

print('Accuracies for different values of n are:',a.values)
abc=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree', 'Gradient Boost']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier(),GradientBoostingClassifier() ]

for i in models:

    model = i

    model.fit(x_train,y_train)

    prediction=model.predict(x_test)

    abc.append(metrics.accuracy_score(prediction,y_test))

models_dataframe=pd.DataFrame(abc,index=classifiers)   

models_dataframe.columns=['Accuracy']

models_dataframe
sns.heatmap(heart[heart.columns[:13]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(20,18)

plt.show()
Bloodpress=[]

for k in heart['RestingBloodPressure']:

    if (k > 130):

        Bloodpress.append(1) #high bp

    else:

        Bloodpress.append(0) #normal



ax = sns.countplot(x=Bloodpress,palette='Set3')



plt.title("Resting Blood Pressure Count")

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))

    

ax.set_xticklabels(["Normal Blood Pressure","Abnormal Blood Pressure"]);
Cholestrol=[]

for k in heart['Cholestrol']:

    if (k <= 200):

         Cholestrol.append(1) #high Cholestrol

    else:

         Cholestrol.append(0) #normal 



ax = sns.countplot(x=Cholestrol,palette='Set3')



plt.title("Cholestrol Level")

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))

    

ax.set_xticklabels(["Normal Cholestrol","Abnormal Cholestrol"]);
Thalassemia=[]

for k in heart['Thalassemia']:

    if (k >= 9.5):

         Thalassemia.append(1) #high Thalassemia

    else:

         Thalassemia.append(0) #normal 



ax = sns.countplot(x=Cholestrol,palette='Set3')



plt.title("Thalassemia Level")

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))

    

ax.set_xticklabels(["Normal Thalassemia","Abnormal Thalassemia"]);
MaxHeartRateAchivied=[]

for k in heart['MaxHeartRateAchivied']:

    if (k > 100):

         MaxHeartRateAchivied.append(1) #high MaxHeartRateAchivied

    else:

         MaxHeartRateAchivied.append(0) #normal 



ax = sns.countplot(x=MaxHeartRateAchivied,palette='Set3')



plt.title("MaxHeartRateAchivied Level")

plt.ylabel("")

plt.yticks([])

plt.xlabel("")



for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))

    

ax.set_xticklabels(["Normal Max-HeartRate","Abnormal Max-HeartRate"]);
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,svm_prediction)

sns.heatmap(cm,annot=True);
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,lr_prediction)

sns.heatmap(cm,annot=True);
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,gb_pred)

sns.heatmap(cm,annot=True);
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,dt_prediction)

sns.heatmap(cm,annot=True);
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,kn_prediction)

sns.heatmap(cm,annot=True);