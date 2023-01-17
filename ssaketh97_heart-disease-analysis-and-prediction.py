from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt





import os

print(os.listdir("../input"))

heart_dataset = pd.read_csv('../input/heart.csv')

heart_dataset.head()
heart_dataset.describe()
data_with_disease = heart_dataset[heart_dataset['target'] == 1]

data_without_disease = heart_dataset[heart_dataset['target'] == 0]
data_with_disease['age'] = ((data_with_disease['age']-(data_with_disease['age']%10))/10)

data_without_disease['age'] = ((data_without_disease['age']-(data_without_disease['age']%10))/10) 



N=8

ind =np.arange(N)



disease_data = [0]*8

no_disease_data = [0]*8



for i in range(0,len(data_with_disease)):

    disease_data[int(data_with_disease['age'][i])] += 1

for i in range(165,165+len(data_without_disease)):

    no_disease_data[int(data_without_disease['age'][i])] += 1 



plt.figure(figsize = (6,6))

p1 = plt.bar(ind,no_disease_data)

p2 = plt.bar(ind,disease_data,bottom = no_disease_data)

plt.xticks(ind, ('0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80'))

plt.yticks(np.arange(0,150,10))

plt.xlabel('age_groups')

plt.ylabel('count of people')

plt.legend((p1[0], p2[0]), ('not_suffered', 'Suffered'))

plt.title('age group wise data comparision between  suffered and un-suffered people in dataset')

plt.show()

plt.figure(figsize = (9,9))

lables = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']

sizes = disease_data

explode = [0, 0, 0, 0, 0, 0, 0, 0]

plt.pie(sizes, labels = lables, explode = explode, shadow = True, startangle=90, autopct='%1.1f%%')

plt.legend()

plt.title('percent of different age groups who are suffered in dataset')

plt.show()
suffered_data = [0]*2

un_suffered_data = [0]*2  

for i in range(0,len(data_with_disease)):

    suffered_data[int(data_with_disease['sex'][i])] += 1

    

for i in range(165,165+len(data_without_disease)):

    un_suffered_data[int(data_without_disease['sex'][i])] += 1 



N=2

ind=np.arange(N)



p1 = plt.bar(ind,suffered_data)

p2 = plt.bar(ind,un_suffered_data,bottom = suffered_data)

plt.xticks(ind, ('female','male'))

plt.yticks(np.arange(0,210,20))

plt.xlabel('gender')

plt.ylabel('count of people')

plt.legend((p1[0], p2[0]), ('suffered', 'not-suffered'))

plt.title('gender wise data comparision in dataset')

plt.show()
chest_pains_count=[0]*4

chest_pains_count_suffered=[0]*4

chest_pains_count_unsuffered=[0]*4

chest_pains_with_gender = [0]*8

types = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']

gender_with_types = ['female','male','female','male','female','male','female','male']



#make data

for i in range(0,len(heart_dataset)):

    chest_pains_count[heart_dataset['cp'][i]]+= 1

    if(heart_dataset['sex'][i] == 0):

        chest_pains_with_gender[heart_dataset['cp'][i]*2]+=1

    else:

        chest_pains_with_gender[(heart_dataset['cp'][i]*2)+1]+=1

#choose colors

a, b, c ,d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges]



#outer ring



fig, ax = plt.subplots()

ax.axis('equal')

out_pie, _ =ax.pie(chest_pains_count, radius =2, labels =types, colors = [a(0.8), b(0.8), c(0.8), d(0.8)])

plt.setp(out_pie, width= 1.2)



#inner ring

in_pie, _ =ax.pie(chest_pains_with_gender, radius =1.4, labels = gender_with_types, labeldistance=0.7,

                  colors = [a(0.4),a(0.6),b(0.4),b(0.6),c(0.4),c(0.6),d(0.4),d(0.6)]) 

plt.setp(in_pie, width = 1)

plt.show() 



for i in range(0,len(data_with_disease)):

    chest_pains_count_suffered[data_with_disease['cp'][i]]+= 1

for i in range(165,165+len(data_without_disease)):

    chest_pains_count_unsuffered[data_without_disease['cp'][i]]+= 1  



    types = ['typical', 'atypical', 'non-anginal', 'asymptomatic']

plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

plt.bar(types,chest_pains_count_suffered,width =0.8)

plt.ylabel('count of different chest pain types who are suffered')



plt.subplot(1,2,2)

plt.bar(types,chest_pains_count_unsuffered,width =0.8)

plt.ylabel('count of different chest pain types who are unsuffered')

plt.show()   

thal_types_suffered = [0]*4

thal_types_unsuffered = [0]*4



N =4

ind = np.arange(N)

for i in range(0,len(heart_dataset)):

    if(heart_dataset['target'][i] == 1):thal_types_suffered[heart_dataset['thal'][i]] +=1

    else:thal_types_unsuffered[heart_dataset['thal'][i]] += 1



plt.figure(figsize = (7,7))

p1 = plt.bar(ind,thal_types_suffered)

p2 = plt.bar(ind,thal_types_unsuffered,bottom = thal_types_suffered)

plt.xticks(ind, ('None', 'normal', 'fixed defect', 'reversable defect'))

plt.yticks(np.arange(0,175,30))

plt.xlabel('thal types in suffered and un-suffered data')

plt.ylabel('count of different thal types who are suffering and not suffering')

plt.legend((p1[0], p2[0]), ('suffered', 'not_suffered'))

plt.title('Thal type wise data comparision between  suffered and un-suffered people in dataset')

plt.show()

heart_dataset['cp'][heart_dataset['cp'] == 0] = 'typical angina'

heart_dataset['cp'][heart_dataset['cp'] == 1] = 'atypical angina'

heart_dataset['cp'][heart_dataset['cp'] == 2] = 'non-anginal pain'

heart_dataset['cp'][heart_dataset['cp'] == 3] = 'asymptomatic'



heart_dataset['restecg'][heart_dataset['restecg'] == 0] = 'ecg1'

heart_dataset['restecg'][heart_dataset['restecg'] == 1] = 'ecg2'

heart_dataset['restecg'][heart_dataset['restecg'] == 2] = 'ecg3'



heart_dataset['slope'][heart_dataset['slope'] == 0] = 'slope1'

heart_dataset['slope'][heart_dataset['slope'] == 1] = 'slope2'

heart_dataset['slope'][heart_dataset['slope'] == 2] = 'slope3'



heart_dataset['thal'][heart_dataset['thal'] == 0] = 'None'

heart_dataset['thal'][heart_dataset['thal'] == 1] = 'normal'

heart_dataset['thal'][heart_dataset['thal'] == 2] = 'fixed'

heart_dataset['thal'][heart_dataset['thal'] == 3] = 'reversable'
plt.figure(figsize = (18,18))

plt.subplot(2,3,1)

plt.boxplot('age', data = heart_dataset)

plt.ylabel('age')

plt.subplot(2,3,2)

plt.boxplot('chol', data = heart_dataset) 

plt.ylabel('chol')

plt.subplot(2,3,3)

plt.boxplot('trestbps', data = heart_dataset )

plt.ylabel('trestbps')

plt.subplot(2,3,4)

plt.boxplot('thalach', data = heart_dataset)

plt.ylabel('thalach')

plt.subplot(2,3,5)

plt.boxplot('oldpeak', data = heart_dataset)

plt.ylabel('oldpeak')

plt.subplot(2,3,6)

plt.boxplot('exang', data = heart_dataset)

plt.ylabel('exang')
for i in range(0,len(heart_dataset)):

    if(heart_dataset['chol'][i] > 360):

        heart_dataset['chol'][i] = int(heart_dataset['chol'][i] + heart_dataset['chol'].median())/2    

    if(heart_dataset['trestbps'][i] > 170):

        heart_dataset['trestbps'][i] = int(heart_dataset['trestbps'][i] + heart_dataset['trestbps'].median())/2    

    if(heart_dataset['oldpeak'][i] > 4):

        heart_dataset['oldpeak'][i] = int(heart_dataset['oldpeak'][i] + heart_dataset['oldpeak'].median())/2 



heart_dataset = heart_dataset.drop(85) # chol value is extremely outside the range

heart_dataset = heart_dataset.drop(48)  # thal is none

heart_dataset = heart_dataset.drop(281) # thal is none



heart_dataset = pd.get_dummies(heart_dataset, drop_first=True)



X = heart_dataset

X=X.drop(['target'],axis =1)

y = heart_dataset['target']



# train test split

from sklearn.model_selection import train_test_split



train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = 0.15, random_state =0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

train_x = sc_X.fit_transform(train_x)

test_x = sc_X.transform(test_x)

#1. logistic regression----------------------------------------------------------------------------------



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state =0)

classifier.fit(train_x,train_y)

y_pred = classifier.predict(test_x)



from sklearn.metrics import confusion_matrix

cm_logistic = confusion_matrix(test_y,y_pred)

print(cm_logistic)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv=10)

print(' ')

print('accuracy')

print(accuracies.mean())

print(' ')

print('standard deviation')

print(accuracies.std())



from sklearn.metrics import f1_score

f1_score(test_y,y_pred)
#2.K-Nearest Neighbours----------------------------------------------------------------------------



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p=2)

classifier.fit(train_x,train_y)

y_pred = classifier.predict(test_x)





cm_knn = confusion_matrix(test_y,y_pred)

print(cm_knn)





accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv=10)

print(' ')

print('accuracy')

print(accuracies.mean())

print(' ')

print('standard deviation')

print(accuracies.std())



f1_score(test_y,y_pred)



from sklearn.model_selection import GridSearchCV

parameters = [{'n_neighbors':[2,3,4,5,6,7,8], 'metric':['minkowski'],'p':[2]}]

gridsearch = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'f1',

                          cv=10)

gridsearch = gridsearch.fit(train_x,train_y)

best_params_knn = gridsearch.best_params_

print(best_params_knn)
#3.SVM-------------------------------------------------------------------------------------------



from sklearn.svm import SVC

classifier = SVC(C=10,kernel = 'rbf',

                 gamma=0.01,

                 random_state = 0)

classifier.fit(train_x,train_y)

y_pred = classifier.predict(test_x)



cm_svm = confusion_matrix(test_y,y_pred)

print(cm_svm)



accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv=10)

print(' ')

print('accuracy')

print(accuracies.mean())

print(' ')

print('standard deviation')

print(accuracies.std())





f1_score(test_y,y_pred)

parameters = [ {'C':[1, 10, 100], 'kernel':['linear']},

               {'C':[1, 10, 100],'kernel' : ['rbf'], 'gamma':[0.1,0.01,0.05,0.005]},

               {'C':[1, 10, 100], 'kernel' :['poly'], 'degree' :[1,2,3,4]}]

gridsearch = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'f1',

                          cv=10)

gridsearch = gridsearch.fit(train_x,train_y)

best_accuracy_SVM = gridsearch.best_score_

best_params_SVM = gridsearch.best_params_

print(best_params_SVM)
#4.Decision Tree --------------------------------------------------------------------------------------



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion ='entropy', random_state =0)

classifier.fit(train_x,train_y)

y_pred = classifier.predict(test_x)





cm_DecisionTree = confusion_matrix(test_y,y_pred)

print(cm_DecisionTree)



accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv=10)

print(' ')

print('accuracy')

print(accuracies.mean())

print(' ')

print('standard deviation')

print(accuracies.std())





f1_score(test_y,y_pred)
#5.Random Forest-------------------------------------------------------------------------------------------------



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators =100, criterion ='gini', random_state = 0)

classifier.fit(train_x,train_y)

y_pred = classifier.predict(test_x)





cm_RandomForest = confusion_matrix(test_y,y_pred)

print(cm_RandomForest)



accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv=10)

print(' ')

print('accuracy')

print(accuracies.mean())

print(' ')

print('standard deviation')

print(accuracies.std())



f1_score(test_y,y_pred)



parameters = [ {'n_estimators':[10,100,200,300,400,500,600,700], 'criterion':['entropy']},

                {'n_estimators':[10,100,200,300,400,500,600,700], 'criterion':['gini']}]

gridsearch = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'f1',

                          cv=10)

gridsearch = gridsearch.fit(train_x,train_y)

best_accuracy_forest = gridsearch.best_score_

best_params_forest = gridsearch.best_params_

print(best_params_forest)