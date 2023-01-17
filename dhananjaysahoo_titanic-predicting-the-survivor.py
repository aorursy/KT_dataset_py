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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from prettytable import PrettyTable



import warnings

warnings.filterwarnings('ignore')
originalDF = pd.read_csv('/kaggle/input/titanic/train.csv')

print(originalDF.shape)

originalDF.head()
originalDF.info()
plt.figure(figsize=(8,6))



plot = sns.countplot(x='Survived',data=originalDF,hue='Sex')



for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+10))



plt.xlabel('Survival')

plt.ylabel('Count')

plt.title("Survivality vs Gender")

#plt.legend()

plt.grid()

plt.show()



total_female = originalDF[originalDF['Sex'] == 'female'].count()

female_surv = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Survived'] == 1)].count()

total_male = originalDF[originalDF['Sex'] == 'male'].count()

male_surv = originalDF[(originalDF['Sex'] == 'male') & (originalDF['Survived'] == 1)].count()



female_survival_rate = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Survived'] == 1)].count()/originalDF[originalDF['Survived'] == 1].count()

female_deceased_rate = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Survived'] == 0)].count()/originalDF[originalDF['Survived'] == 0].count()



male_dustribution = np.round(total_male['Sex']/(total_male['Sex']+total_female['Sex']),2)

female_distribution = np.round(total_female['Sex']/(total_male['Sex']+total_female['Sex']),2)



print("Total Gender Distribution: Male: {} and Female: {}".format(male_dustribution,female_distribution))

#print("%age of female who survived:",np.round(female_surv['Sex']/total_female['Sex'],2))

#print("%age of male who survived:",np.round(male_surv['Sex']/total_male['Sex'],2))



print("Survival Gender Distribution: Male: {} and Female: {}".format(np.round(1-female_survival_rate['Sex'],2),np.round(female_survival_rate['Sex'],2)))

print("Deceased Gender Distribution: Male: {} and Female: {}".format(np.round(1-female_deceased_rate['Sex'],2),np.round(female_deceased_rate['Sex'],2)))
plt.figure(figsize=(8,6))

plot = sns.countplot(x='Survived',data=originalDF,hue='Pclass')



for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+5))



plt.xlabel('Survival')

plt.ylabel('Count')

plt.title("Survivality vs PClass")

#plt.legend()

plt.grid()

plt.show()



total_3rd_Pclass = originalDF[originalDF['Pclass'] == 3].count()

total_3rd_Pclass_Surv = originalDF[(originalDF['Pclass'] == 3) & (originalDF['Survived'] == 1)].count()



total_2nd_Pclass = originalDF[originalDF['Pclass'] == 2].count()

total_2nd_Pclass_Surv = originalDF[(originalDF['Pclass'] == 2) & (originalDF['Survived'] == 1)].count()



total_1st_Pclass = originalDF[originalDF['Pclass'] == 1].count()

total_1st_Pclass_Surv = originalDF[(originalDF['Pclass'] == 1) & (originalDF['Survived'] == 1)].count()



print("%age of 1st Class passengers who survived: {}".format(np.round(total_1st_Pclass_Surv['Pclass']/total_1st_Pclass['Pclass'],2)))

print("%age of 2nd Class passengers who survived: {}".format(np.round(total_2nd_Pclass_Surv['Pclass']/total_2nd_Pclass['Pclass'],2)))

print("%age of 3rd Class passengers who survived: {}".format(np.round(total_3rd_Pclass_Surv['Pclass']/total_3rd_Pclass['Pclass'],2)))
femaleDF = originalDF[(originalDF['Sex']=='female') & (originalDF['Survived']==1)]



plt.figure(figsize=(8,6))

plot = sns.countplot(x='Sex',data=femaleDF,hue='Pclass')



for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))



plt.xlabel('Gender')

plt.ylabel('Count')

plt.title("Female Surviability Across PClass")

#plt.legend()

plt.grid()

plt.show()



total_female_in_1stPC = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Pclass'] == 1)].count()

total_female_surv_1stPC = femaleDF[femaleDF['Pclass'] == 1].count()



total_female_in_2ndPC = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Pclass'] == 2)].count()

total_female_surv_2ndPC = femaleDF[femaleDF['Pclass'] == 2].count()



total_female_in_3rdPC = originalDF[(originalDF['Sex'] == 'female') & (originalDF['Pclass'] == 3)].count()

total_female_surv_3rdPC = femaleDF[femaleDF['Pclass'] == 3].count()



print("%age of women survivided in 1st PClass:",np.round(total_female_surv_1stPC['Sex']/total_female_in_1stPC['Sex'],2))

print("%age of women survivided in 2nd PClass:",np.round(total_female_surv_2ndPC['Sex']/total_female_in_2ndPC['Sex'],2))

print("%age of women survivided in 3rd PClass:",np.round(total_female_surv_3rdPC['Sex']/total_female_in_3rdPC['Sex'],2))
plt.figure(figsize=(8,6))

plot = sns.countplot(x='Embarked',data=originalDF,hue='Survived')



for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))



plt.xlabel('Embarked')

plt.ylabel('Count')

plt.title("Embarked vs Survived")

#plt.legend()

plt.grid()

plt.show()
cabin_nan = originalDF[originalDF['Cabin'].isnull()]



plt.figure(figsize=(8,6))

plot = sns.countplot(x='Embarked',data=cabin_nan,hue='Survived')



for p in plot.patches:

    plot.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))



plt.xlabel('Embarked')

plt.ylabel('Count')

plt.title("Embarked vs Survived For Passenger without Cabin")

#plt.legend()

plt.grid()

plt.show()



passenger_wo_cabin = cabin_nan['Cabin'].isnull().count()

passenger_wo_cabin_and_surv = cabin_nan[(cabin_nan['Cabin'].isnull()) & (cabin_nan['Survived'] == 1)].count()

#print(passenger_wo_cabin,passenger_wo_cabin_and_surv['PassengerId'])



print("%age of Survivability for passenger without cabin:",np.round(passenger_wo_cabin_and_surv['PassengerId']/passenger_wo_cabin,2))
sns.FacetGrid(data=originalDF,size=8,hue='Survived').map(sns.scatterplot,'PassengerId','Age').add_legend()

plt.grid()

plt.title("Survival vs Age")

plt.show()
# Fill in the missing data for Cabin attribute. This will be a simple Yes/No categorical feature.

originalDF.loc[originalDF['Cabin'].isnull(),'Cabin'] = 'No'

originalDF.loc[originalDF['Cabin'].notnull(), 'Cabin'] = 'Yes'



# Fill in the missing data for Embarked attribute. The missing data was updated based on Pclass, Sex and the Port of Embarkation of the passenger.

originalDF.loc[originalDF['Embarked'].isnull(),'Embarked'] = 'S'



# Fill in the missing data for Age attribute. The missing data was update with the mean age of the passenger depending of female or male for whom the data was missing

passenger = originalDF[(originalDF['Age'].isnull()) & (originalDF['Sex'] == 'male')].count()

passenger = originalDF[(originalDF['Age'].isnull()) & (originalDF['Sex'] == 'female')].count()



male_passenger = originalDF[(originalDF['Age'].notnull()) & (originalDF['Sex'] == 'male')].mean()

female_passenger = originalDF[(originalDF['Age'].notnull()) & (originalDF['Sex'] == 'female')].mean()



originalDF.loc[((originalDF['Age'].isnull()) & (originalDF['Sex'] == 'female')),'Age'] = female_passenger['Age']

originalDF.loc[((originalDF['Age'].isnull()) & (originalDF['Sex'] == 'male')),'Age'] = male_passenger['Age']



# Removing the non essential features

originalDF.drop(['Name','Ticket'],axis=1,inplace=True)
#Prepare the traing data set

train_df = originalDF.drop(['Survived','PassengerId'],axis=1)

train_num = train_df.drop(['Sex','Cabin','Embarked'],axis=1)

train_label = originalDF['Survived'].copy()



# Perform normalization of the numerical data and onehot encoding for the categorical data.

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



num_pipeline = Pipeline([('std_scaler',StandardScaler())])



num_attribs = list(train_num)

cat_attribs = ['Sex','Cabin','Embarked']



full_pipeline = ColumnTransformer([

                                  ('num',num_pipeline,num_attribs),

                                  ('cat',OneHotEncoder(),cat_attribs)

                                  ])



train_prepared = full_pipeline.fit_transform(train_df)

print("Shape of train_prepared:",train_prepared.shape)

print("Shape of train_df:",train_df.shape)
testDF = pd.read_csv('/kaggle/input/titanic/test.csv')

print(testDF.shape)

testDF.head()
# Fill in the missing data for Cabin attribute. This will be a simple Yes/No categorical feature.

testDF.loc[testDF['Cabin'].isnull(),'Cabin'] = 'No'

testDF.loc[testDF['Cabin'].notnull(), 'Cabin'] = 'Yes'



# Fill in the missing data for Age attribute. The missing data was update with the mean age of the passenger depending of female or male for whom the data was missing

passenger = testDF[(testDF['Age'].isnull()) & (testDF['Sex'] == 'male')].count()

passenger = testDF[(testDF['Age'].isnull()) & (testDF['Sex'] == 'female')].count()



male_passenger = testDF[(testDF['Age'].notnull()) & (testDF['Sex'] == 'male')].mean()

female_passenger = testDF[(testDF['Age'].notnull()) & (testDF['Sex'] == 'female')].mean()



testDF.loc[((testDF['Age'].isnull()) & (testDF['Sex'] == 'female')),'Age'] = female_passenger['Age']

testDF.loc[((testDF['Age'].isnull()) & (testDF['Sex'] == 'male')),'Age'] = male_passenger['Age']



# Fill in the missing data for Fare attribute. The missing data was update with the mean Fare of passenger based on their Pclass, Sex, SibSp, Parch and Cabin

passenger = testDF[(testDF['Fare'].notnull()) & (testDF['Pclass'] == 3) & 

       (testDF['Sex'] == 'male') & (testDF['SibSp'] == 0) & 

       (testDF['Parch'] == 0) & (testDF['Cabin'] == 'Yes')

      ].mean()

testDF.loc[testDF['Fare'].isnull(),'Fare'] = passenger['Fare']



# Preparing the test dataset

test_df = testDF.drop(['Name','Ticket','PassengerId'],axis=1)

test_num = test_df.drop(['Sex','Cabin','Embarked'],axis=1)



#Shape of Train and Test dataset

test_prepared = full_pipeline.transform(test_df)

print("Shape of train_prepared:",train_prepared.shape)

print("Shape of original DF:",test_prepared.shape)
# Functions to capture performace scores:



from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import confusion_matrix,precision_recall_curve,precision_score,recall_score,f1_score,roc_auc_score,roc_curve





def crossValScore(model,x,y,cv,score):

    return (cross_val_score(model,x,y,cv=cv,scoring=score))



def first_metrics(model,x,y,cv):

    model_predict = cross_val_predict(model,x,y,cv=cv)

    model_confmat = confusion_matrix(y,model_predict)

    model_precScore = precision_score(y,model_predict)

    model_recallScore = recall_score(y,model_predict)

    model_f1Score = f1_score(y,model_predict)

    

    return model_confmat,model_precScore,model_recallScore,model_f1Score

    

def crossValDecFunc(model,x,y,cv,method):

    return(cross_val_predict(model,x,y,cv=cv,method=method))



def second_metrics(x,y):

    roc_score = roc_auc_score(x,y)

    #print("RoC Score:",roc_score)

    #print("\n")

    fpr,tpr,thresholds = roc_curve(x,y)

    plt.figure(figsize=(8,6))

    plt.plot(fpr,tpr,linewidth=2)

    plt.plot([0,1],[0,1],'r--')

    plt.xlabel("FPR")

    plt.ylabel("TPR")

    plt.title("Area Under Curve")

    plt.legend()

    plt.grid()

    plt.show()

    

    return roc_score
# Model 1: Logistic Regression



from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(train_prepared,train_label)
#Logistic Regression Metrics:



log_reg_score = crossValScore(log_reg,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",log_reg_score)

log_conf_mat, log_prec, log_recall, log_f1 = first_metrics(log_reg,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",log_conf_mat)

print("\nPrecision Score:",log_prec)

print("\nRecall Score:",log_recall)

print("\nF1 Score:",log_f1)



log_reg_df = crossValDecFunc(log_reg,train_prepared,train_label,3,'decision_function')

log_reg_roc = second_metrics(train_label,log_reg_df)

print("Log RoC Score:",log_reg_roc)
# Model 2: SGDClassifier



from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()

sgd_clf.fit(train_prepared,train_label)
#SGDClassifier Metrics:



sgd_clf_score = crossValScore(sgd_clf,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",sgd_clf_score)

sgd_conf_mat, sgd_prec, sgd_recall, sgd_f1 = first_metrics(sgd_clf,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",sgd_conf_mat)

print("\nPrecision Score:",sgd_prec)

print("\nRecall Score:",sgd_recall)

print("\nF1 Score:",sgd_f1)



sgd_reg_df = crossValDecFunc(sgd_clf,train_prepared,train_label,3,'decision_function')

sgd_roc = second_metrics(train_label,sgd_reg_df)

print("SGD RoC Score:",sgd_roc)
# Model 3: KNNClassifier



from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5)

knn_clf.fit(train_prepared,train_label)
#KNeighborsClassifier Metrices



knn_clf_score = crossValScore(knn_clf,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",knn_clf_score)

knn_conf_mat, knn_prec, knn_recall, knn_f1 = first_metrics(knn_clf,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",knn_conf_mat)

print("\nPrecision Score:",knn_prec)

print("\nRecall Score:",knn_recall)

print("\nF1 Score:",knn_f1)



knn_reg_df = crossValDecFunc(knn_clf,train_prepared,train_label,3,'predict_proba')

knn_roc = second_metrics(train_label,knn_reg_df[:,1])

print("KNN RoC Score:",knn_roc)
# Model 4: SVC



from sklearn.svm import SVC

svc_clf = SVC()

svc_clf.fit(train_prepared,train_label)
#SVC Metrics:



svc_clf_score = crossValScore(svc_clf,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",svc_clf_score)

svc_conf_mat, svc_prec, svc_recall, svc_f1 = first_metrics(svc_clf,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",svc_conf_mat)

print("\nPrecision Score:",svc_prec)

print("\nRecall Score:",svc_recall)

print("\nF1 Score:",svc_f1)



svc_reg_df = crossValDecFunc(svc_clf,train_prepared,train_label,3,'decision_function')

svc_roc = second_metrics(train_label,svc_reg_df)

print("SVC RoC Score:",svc_roc)
# Model 5: Random Forest



from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

forest_clf.fit(train_prepared,train_label)
#RandomForestClassifier Metrics:



forest_clf_score = crossValScore(forest_clf,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",forest_clf_score)

forest_conf_mat, forest_prec, forest_recall, forest_f1 = first_metrics(forest_clf,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",forest_conf_mat)

print("\nPrecision Score:",forest_prec)

print("\nRecall Score:",forest_recall)

print("\nF1 Score:",forest_f1)



forest_reg_df = crossValDecFunc(forest_clf,train_prepared,train_label,3,'predict_proba')

forest_roc = second_metrics(train_label,forest_reg_df[:,1])

print("Forest RoC Score:",forest_roc)
# Model 6: Grid Search With SVC Hyper Tunning



from sklearn.model_selection import GridSearchCV

svc_grid_clf = SVC(random_state=42)



params = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':[1,10,100],'gamma':['scale', 'auto']}

grid_search = GridSearchCV(svc_grid_clf,params,cv=3,scoring='precision',return_train_score=True)

grid_search.fit(train_prepared,train_label)
print(grid_search.best_params_)

print(grid_search.best_estimator_)

final_model = grid_search.best_estimator_

final_model.fit(train_prepared,train_label)



#Grid Search With SVC Hyper Tunning metrics:



grid_svc_clf_score = crossValScore(final_model,train_prepared,train_label,3,'accuracy')

print("Cross Val Accuracy Scores:",grid_svc_clf_score)

grid_svc_conf_mat, grid_svc_prec, grid_svc_recall, grid_svc_f1 = first_metrics(final_model,train_prepared,train_label,3)

print("\nConfusion Matrix:\n",grid_svc_conf_mat)

print("\nPrecision Score:",grid_svc_prec)

print("\nRecall Score:",grid_svc_recall)

print("\nF1 Score:",grid_svc_f1)



grid_svc_reg_df = crossValDecFunc(final_model,train_prepared,train_label,3,'decision_function')

grid_svc_roc = second_metrics(train_label,grid_svc_reg_df)

print("RoC Score:",grid_svc_roc)
model_compare = PrettyTable()

model_compare.field_names = ["Model", "Precision", "Recall", "F1Score","RoCScore"]

model_compare.add_row(['LogReg',np.round(log_prec,2),np.round(log_recall,2),np.round(log_f1,2),np.round(log_reg_roc,2)])

model_compare.add_row(['SGD',np.round(sgd_prec,2),np.round(sgd_recall,2),np.round(sgd_f1,2),np.round(sgd_roc,2)])

model_compare.add_row(['KNN',np.round(knn_prec,2),np.round(knn_recall,2),np.round(knn_f1,2),np.round(knn_roc,2)])

model_compare.add_row(['SVC',np.round(svc_prec,2),np.round(svc_recall,2),np.round(svc_f1,2),np.round(svc_roc,2)])

model_compare.add_row(['Forest',np.round(forest_prec,2),np.round(forest_recall,2),np.round(forest_f1,2),np.round(forest_roc,2)])

model_compare.add_row(['Grid+SVC',np.round(grid_svc_prec,2),np.round(grid_svc_recall,2),np.round(grid_svc_f1,2),np.round(grid_svc_roc,2)])





print(model_compare)
test_predict_gridsvc = final_model.predict(test_prepared)



test_predict_gridsvc_list = test_predict_gridsvc.tolist()

deceased = 0

survived = 0

for i in test_predict_gridsvc_list:

    if i == 0:

        deceased += 1

    else:

        survived += 1

        

print("Test Grid+SVC: Survived - {} and Deceased - {}".format(survived,deceased))