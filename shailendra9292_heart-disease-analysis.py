import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier

from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import plot_importance

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import cross_val_score,ShuffleSplit

import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
df = pd.read_csv('../input/heart.csv')
df.columns
df.rename(columns={'cp':'chest_pain','chol':'cholesterol','restecg':'resting_ecg','trestbps':'resting_blood_pressure',

                   'fbs':'fasting_blood_sugar','thalach':'max_heart_rate','exang':'exercise_induced_angina','oldpeak':

          'st_depression','thal':'thalassemia','ca':'num_major_vessels'},inplace=True)
df.head()
df.isnull().sum()
df.info()
df.describe()
df.age.value_counts()[:10]
plt.figure(figsize=(19,5))

sns.barplot(x=df.age.value_counts().index,y=df.age.value_counts().values)

plt.xlabel("Age")

plt.ylabel("Counter")
plt.figure(figsize=(11,5),dpi = 70)

sns.distplot(df['age'],color='lightseagreen',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 2})

plt.show()
print("Total number of Females :",len(df[df.sex==0]))

print("Total number of Males :",len(df[df.sex==1]))
plt.figure(dpi=100)

plt.subplot(1,2,1)

plt.title('0:Female   1:Male',fontdict={'fontsize':9})

sns.countplot(df.sex)

plt.subplot(1,2,2)

plt.pie(df.sex.value_counts(),autopct='%.2f%%',explode=[0.1,0],shadow=True,labels=['Male','Female'],startangle=90,

        colors=['darkcyan','tomato'])

plt.show()
print("Total number of person infected with Heart Disease:",len(df[df.target==1]))

print("Total number of person without Heart Disease :",len(df[df.target==0]))
plt.figure(figsize=(5,4))

sns.countplot(df.target)
plt.figure(dpi=100)

plt.subplot(1,2,1)

plt.title("0: Person Without Disease\n 1:Person Affected with Heart Disease \n(Age vs Target)",fontdict={'fontsize':8})

sns.countplot(df.target)

plt.subplot(1,2,2)

plt.pie(df.sex.value_counts(),autopct='%.2f%%',explode=[0.1,0],shadow=True,labels=['Male','Female'],startangle=90,

        colors=['darkcyan','tomato'])

plt.show()
chest_type=['Typical Angina','Atypical Angina', 'Non-Anginal Pain','Asymptomatic']

for types,counts in zip(chest_type,df.chest_pain.value_counts()):

    print('Number of Person with Chest Pain Type "{}" are: {}'.format(types,counts))
plt.figure(dpi=100)

plt.pie(df.chest_pain.value_counts().values,labels=chest_type,autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0],startangle=90)

plt.title('Chest Pain Types\n')

plt.show()
plt.figure(dpi=100)

sns.swarmplot(df.sex,df.age,hue=df.chest_pain,edgecolor='black',s=6)

plt.legend(loc=3)

plt.show()
plt.figure(dpi=100)

sns.boxenplot(x=df.sex,y=df.age,hue=df.chest_pain)

plt.title("0: female, 1:male",fontdict={'fontsize':11})

plt.show()
plt.figure(dpi=80)

sns.regplot(x=df.age,y=df.cholesterol,color='indianred',line_kws={"color":"b","alpha":0.5,"lw":3})

plt.title('Age vs Cholesterol')

plt.show()
plt.figure(dpi=80)

sns.regplot(x=df.age,y=df.max_heart_rate,color='lightseagreen',line_kws={"color":"r","alpha":0.7,"lw":3},scatter_kws={'alpha':0.5})

plt.title('Age vs Max Heart Ratr')

plt.show()
plt.figure(figsize=(15,4),dpi=110)

palette = sns.color_palette("mako_r", 2)

sns.lineplot(x=df.age,y=df.max_heart_rate,hue=df.target,palette=palette,markers=True,style=df.target)

plt.legend(['Non Heart Patients','Heart Patients'],loc=1)

plt.title("0: Person Without Disease\n 1:Person Affected with Heart Disease ")

plt.show()
plt.figure(figsize=(15,4),dpi=110)

palette = sns.diverging_palette(220, 20, n=2)

sns.lineplot(x=df.age,y=df.resting_blood_pressure,hue=df.target,palette=palette,markers=True,style=df.target)

plt.legend(['Non Heart Patients','Heart Patients'],loc=1)

plt.title("0: Person Without Disease\n 1:Person Affected with Heart Disease ")

plt.show()
plt.figure(figsize=(19,5),dpi=100)

sns.countplot(df.age,data=df,hue='target',palette='GnBu')

plt.legend(['Non Heart Patients','Heart Patients'],loc=1)

plt.title('Classification of Disease with Ages')

plt.show()
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True,fmt='.1f',cmap='GnBu')

plt.show()
df['sex']=df['sex'].astype('category')

df['chest_pain']=df['chest_pain'].astype('category')

df['fasting_blood_sugar']=df['fasting_blood_sugar'].astype('category')

df['resting_ecg']=df['resting_ecg'].astype('category')

df['exercise_induced_angina']=df['exercise_induced_angina'].astype('category')

df['slope']=df['slope'].astype('category')

df['num_major_vessels']=df['num_major_vessels'].astype('category')

df['thalassemia']=df['thalassemia'].astype('category')

df['target']=df['target'].astype('category')
y = df.target
x = pd.get_dummies(df.iloc[:,:-1],drop_first=True)


x.head()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
classifiers=[['Logistic Regression :',LogisticRegression()],

             ['Decision Tree Classification :',DecisionTreeClassifier()],

       ['Random Forest Classification :',RandomForestClassifier(n_estimators=10)],

       ['Gradient Boosting Classification :', GradientBoostingClassifier()],

       ['Ada Boosting Classification :',AdaBoostClassifier()],

       ['Extra Tree Classification :', ExtraTreesClassifier()],

       ['K-Neighbors Classification :',KNeighborsClassifier()],

       ['Support Vector Classification :',SVC(kernel='linear')],

       ['Gaussian Naive Bayes :',GaussianNB()],

            ['XGBoost :',XGBClassifier()]]



for name,model in classifiers:

    model=model

    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    print(name,'Test score is',accuracy_score(y_test,predictions)*100,'and',end =' ',sep= ' ')

    print('Training score is',accuracy_score(y_train,model.predict(x_train))*100)
classifier=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=10),

            GradientBoostingClassifier(),AdaBoostClassifier(),

             ExtraTreesClassifier(),KNeighborsClassifier(),SVC(kernel='linear'),GaussianNB(),XGBClassifier()]

plt.figure(figsize=(23,9),dpi=100)

for model,j in zip(classifier,range(len(classifier))):

    

    model.fit(x_train,y_train)

    pred = model.predict(x_test)

    plt.subplot(3,4,j+1)

    plt.title(classifiers[j][0])

    sns.heatmap(confusion_matrix(y_test,pred),annot=True,cmap='GnBu')
# Feature importance Visualisation 

model =XGBClassifier()

model.fit(x_train,y_train)

plot_importance(model)
cross_val_score(LogisticRegression(penalty='l1'),x_train,y_train,cv=ShuffleSplit(n_splits=5,test_size=0.2)).mean()
print(round(accuracy_score(y_test,LogisticRegression(penalty='l2').fit(x_train,y_train).predict(x_test))*100,2),'%')