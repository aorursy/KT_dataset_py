# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
dataset= pd.read_csv("../input/train.csv")
%matplotlib inline

import seaborn

seaborn.set() 



#-------------------class별 생존자/사망자 비율-------------------------------------

survived_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()

dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class,dead_class])

df_class.index = ['Survived','Died']

df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")



Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100

print("Percentage of Class 1 that survived:" ,round(Class1_survived),"%")

print("Percentage of Class 2 that survived:" ,round(Class2_survived), "%")

print("Percentage of Class 3 that survived:" ,round(Class3_survived), "%")



# 표로 나타내기

from IPython.display import display

display(df_class)
#-------------------sex별 생존자/사망자 비율------------------------------------

   

Survived = dataset[dataset.Survived == 1]['Sex'].value_counts()

Died = dataset[dataset.Survived == 0]['Sex'].value_counts()

df_sex = pd.DataFrame([Survived , Died])

df_sex.index = ['Survived','Died']

df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")





female_survived= df_sex.female[0]/df_sex.female.sum()*100

male_survived = df_sex.male[0]/df_sex.male.sum()*100

print("Percentage of female that survived:" ,round(female_survived), "%")

print("Percentage of male that survived:" ,round(male_survived), "%")



# 표로 나타내기

from IPython.display import display

display(df_sex) 
#-------------------- Embarked별 생존자/사망자 비율 ----------------------------



survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()

dead_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()

df_embark = pd.DataFrame([survived_embark,dead_embark])

df_embark.index = ['Survived','Died']

df_embark.plot(kind='bar',stacked=True, figsize=(5,3))



Embark_S= df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100

Embark_C = df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100

Embark_Q = df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100

print("Percentage of Embark S that survived:", round(Embark_S), "%")

print("Percentage of Embark C that survived:" ,round(Embark_C), "%")

print("Percentage of Embark Q that survived:" ,round(Embark_Q), "%")



from IPython.display import display

display(df_embark)
X = dataset.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)

y = X.Survived                       # 라벨로 이루어진 벡터(종속변수)

X=X.drop(['Survived'], axis=1)       # 데이터프레임 X로부터 종속변수를 제거



X.head(20)
# ----------------- 범주형 데이터 코딩 -------------------------



# "Sex" 코딩

from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

X.Sex=labelEncoder_X.fit_transform(X.Sex)





# "Embarked" 코딩



# embarked의 결측치 개수:

print ('Number of null values in Embarked:', sum(X.Embarked.isnull()))



# 2개의 결측치에 옵션 중 하나를 채워 넣기(S, C 혹은 Q)

row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked']='S' 



Embarked  = pd.get_dummies(  X.Embarked , prefix='Embarked'  )

X = X.drop(['Embarked'], axis=1)

X= pd.concat([X, Embarked], axis=1)  

# we should drop one of the columns

X = X.drop(['Embarked_S'], axis=1)



X.head()
#-------------- 결측값들 관리  -----------------------------



print ('Number of null values in Age:', sum(X.Age.isnull()))

 



# -------- Name -> Title (이름->호칭)으로 변경----------------------------

got= dataset.Name.str.split(',').str[1]

X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# ---------------------------------------------------------- 





#------------------ 호칭별 평균 나이 -------------------------------------------------------------

ax = plt.subplot()

ax.set_ylabel('Average age')

X.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)



title_mean_age=[]

title_mean_age.append(list(set(X.Name)))  #호칭의 고유값을 위한 셋 그리고 리스트로 변환

title_mean_age.append(X.groupby('Name').Age.mean())

title_mean_age

#------------------------------------------------------------------------------------------------------





#------------------ 빈 나이를 메꾸기 ---------------------------

n_traning= dataset.shape[0]   #행 개수

n_titles= len(title_mean_age[1])

for i in range(0, n_traning):

    if np.isnan(X.Age[i])==True:

        for j in range(0, n_titles):

            if X.Name[i] == title_mean_age[0][j]:

                X.Age[i] = title_mean_age[1][j]

#--------------------------------------------------------------------    



X=X.drop(['Name'], axis=1)



       
for i in range(0, n_traning):

    if X.Age[i] > 18:

        X.Age[i]= 0

    else:

        X.Age[i]= 1



X.head()
#-----------------------Logistic Regression---------------------------------------------

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l2',random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Logistic Regression:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")







#-----------------------------------K-NN --------------------------------------------------



# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)





# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("K-NN:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")





#---------------------------------------SVM -------------------------------------------------



# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("SVM:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")





#---------------------------------Naive Bayes-------------------------------------------



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Naive Bayes:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")







#----------------------------Random Forest------------------------------------------



# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())
