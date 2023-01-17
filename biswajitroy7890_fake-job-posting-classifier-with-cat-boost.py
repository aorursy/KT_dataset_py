import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
from google.colab import files
files.upload()
pd.set_option('display.max_rows',14000)
Job_post=pd.read_csv('fake_job_postings.csv', encoding='latin')
Job_post=Job_post.dropna()
(Job_post.isnull().sum()/(Job_post.shape[0]))*100
Droplabels=['job_id','location','salary_range']
Droplabels1=['function']
Job_post=Job_post.drop(Droplabels1, axis=1)
Job_post['AllText']=Job_post['department']+" "+Job_post['company_profile']+" "+Job_post['description']+" "+Job_post['requirements']+" "+Job_post['benefits']
Job_post
Job_post=Job_post.dropna()
Job_post.head()
Droplabels3=['company_profile','description','requirements','benefits']
Job_post=Job_post.drop(Droplabels3, axis=1)
def stripping(Inpdata):
    cleanedArticle1=re.sub(r'[?|$|.@#=><|!]Â&*/',r' ',Inpdata)
    cleanedArticle2=re.sub(r'[^a-z A-Z]',r' ',cleanedArticle1)
    cleanedArticle3=cleanedArticle2.lower()
    cleanedArticle4=re.sub(r'\b\w{1,2}\b', ' ',cleanedArticle3)
    cleanedArticle5=re.sub(r'https?://\S+|www\.\S+',r' ',cleanedArticle4)
    cleanedArticle6=re.sub(r' +', ' ',cleanedArticle5)
    return(cleanedArticle6)
Job_post['ALL']=Job_post['AllText'].apply(stripping)
Job_post=Job_post.drop('AllText', axis=1)
Job_post.head()
Job_post.groupby('fraudulent').size().plot(kind='bar',figsize=(15,6))
Job_post.shape
(Job_post.isnull().sum()/(Job_post.shape[0]))*100
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = vectorizer.fit_transform(Job_post['ALL'])
Job_post_ML=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
Job_post_ML.head()
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
Job_post['title']=Le.fit_transform(Job_post['title'])
Job_post['department']=Le.fit_transform(Job_post['department'])
Job_post['employment_type']=Le.fit_transform(Job_post['employment_type'])
Job_post['required_experience']=Le.fit_transform(Job_post['required_experience'])
Job_post['required_education']=Le.fit_transform(Job_post['required_education'])
Job_post['industry']=Le.fit_transform(Job_post['industry'])

Job_post_ML['telecommuting']=Job_post['telecommuting']
Job_post_ML['has_company_logo']=Job_post['has_company_logo']
Job_post_ML['has_questions']=Job_post['has_questions']
Job_post_ML['fraudulent']=Job_post['fraudulent']
Job_post_ML['title']=Job_post['title']
Job_post_ML['department']=Job_post['department']
Job_post_ML['employment_type']=Job_post['employment_type']
Job_post_ML['required_experience']=Job_post['required_experience']
Job_post_ML['required_education']=Job_post['required_education']
Job_post_ML['industry']=Job_post['industry']
Job_post_ML['fraudulent']=Job_post['fraudulent']
Mod=Job_post_ML['fraudulent'].mode()[0]
Job_post_ML['fraudulent']=Job_post_ML['fraudulent'].fillna(Mod)
cols=['telecommuting','has_company_logo','department','employment_type','required_education','industry']
for i in cols:
    print(Job_post_ML[i].isnull().sum())
Mod=Job_post_ML['telecommuting'].mode()[0]
Job_post_ML['telecommuting']=Job_post_ML['telecommuting'].fillna(Mod)
Mod=Job_post_ML['has_company_logo'].mode()[0]
Job_post_ML['has_company_logo']=Job_post_ML['has_company_logo'].fillna(Mod)
Mod=Job_post_ML['has_questions'].mode()[0]
Job_post_ML['has_questions']=Job_post_ML['has_questions'].fillna(Mod)
Mod=Job_post_ML['fraudulent'].mode()[0]
Job_post_ML['fraudulent']=Job_post_ML['fraudulent'].fillna(Mod)
Mod=Job_post_ML['title'].mode()[0]
Job_post_ML['title']=Job_post_ML['title'].fillna(Mod)
Mod=Job_post_ML['department'].mode()[0]
Job_post_ML['department']=Job_post_ML['department'].fillna(Mod)
Mod=Job_post_ML['employment_type'].mode()[0]
Job_post_ML['employment_type']=Job_post_ML['employment_type'].fillna(Mod)
Mod=Job_post_ML['required_experience'].mode()[0]
Job_post_ML['required_experience']=Job_post_ML['required_experience'].fillna(Mod)
Mod=Job_post_ML['required_education'].mode()[0]
Job_post_ML['required_education']=Job_post_ML['required_education'].fillna(Mod)
Mod=Job_post_ML['industry'].mode()[0]
Job_post_ML['industry']=Job_post_ML['industry'].fillna(Mod)


Job_post_ML.groupby('fraudulent').size().plot(kind='bar', figsize=(15,6))
Target=['fraudulent']
y=Job_post_ML[Target].values
Job_post_ML=Job_post_ML.drop(labels='fraudulent', axis=1)
Predictors=Job_post_ML.columns
X=Job_post_ML[Predictors].values
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Lda=LinearDiscriminantAnalysis(n_components=1)
reduced_data = Lda.fit(X,y).transform(X)
reduced_data [:]
principalDf = pd.DataFrame(data = reduced_data, columns = ['PC-1'])
finalDf = pd.concat([principalDf,Job_post_ML['fraudulent']], axis = 1)
finalDf.groupby('fraudulent').size().plot(kind='bar', figsize=(15,6))
from sklearn.model_selection import train_test_split
Predictors=['PC-1']
Target=['fraudulent']
X=finalDf[Predictors].values
y=finalDf[Target].values

print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=3500)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
def RandomForest_classifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.26,0.33,0.36,0.42,0.45]
    Random_state=[521457,7505,32578,5,2567,4789,8547,657]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            RFC =RandomForestClassifier(max_depth=3, n_estimators=300,criterion='entropy')
            predictModel=RFC.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy) 
length,AVGAccuracy=RandomForest_classifier(X,y)
Sum_of_Acc_Dtree=sum(AVGAccuracy)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
def Logistic_Regression(Data1,Data2):
    from sklearn.linear_model import LogisticRegression
    Test_size=[0.30,0.20,0.23,0.26,0.33,0.36,0.42,0.45]
    Random_state=[521457,50,32578,5,2567,4789,8547,657]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            lgf=LogisticRegression(C=2,penalty='l2', solver='liblinear')
            predictModel=lgf.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy)  
length,AVGAccuracy=Logistic_Regression(X,y)
Sum_of_Acc=sum(AVGAccuracy)
print("The Average of All acuracies",(Sum_of_Acc/length))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
def Adaboost_Classifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[521457,7505,32578,5,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            DTC=DecisionTreeClassifier(max_depth=5)
            ADA = AdaBoostClassifier(n_estimators=150, base_estimator=DTC ,learning_rate=0.01)
            predictModel=ADA.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy,ADA ) 
length,AVGAccuracy,AD=Adaboost_Classifier(X,y)
Sum_of_Acc_Dtree=sum(AVGAccuracy)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
from xgboost import XGBClassifier
def Xgboost_classifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[521457,7505,32578,5,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            xgb=XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=400, objective='binary:logistic', booster='gbtree')
            predictModel=xgb.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy,xgb )
length,AvgACC,XG=Xgboost_classifier(X,y)
Sum_of_Acc_Dtree=sum(AvgACC)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
from sklearn.ensemble import GradientBoostingClassifier
def GradientBoostClassifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[521457,7505,32578,5,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            GBC=GradientBoostingClassifier(max_depth=5,learning_rate=0.01,n_estimators=300,)
            predictModel=GBC.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy, GBC ) 
length,AverageAcc,GB=GradientBoostClassifier(X,y)
Sum_of_Acc_Dtree=sum(AverageAcc)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
from lightgbm import LGBMClassifier
def lightbgmclassifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[521457,7505,32578,5,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            lgb=LGBMClassifier(max_depth=-1,learning_rate=0.1,n_estimators=300)
            predictModel=lgb.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy, lgb ) 
length,AverageAcc,LGB=lightbgmclassifier(X,y)
Sum_of_Acc_Dtree=sum(AverageAcc)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
def NaiveBayesClassifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[525478,7505,32578,3257,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            GNB=GaussianNB()
            predictModel=GNB.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy,GNB)             
                                  
length,AverageAcc,GNB=NaiveBayesClassifier(X,y)
Sum_of_Acc_Dtree=sum(AverageAcc)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))
!pip install catboost
import catboost
def CatboostClassifier(Data1,Data2):
    Test_size=[0.30,0.25,0.21,0.22,0.33,0.36,0.42,0.45,0.34]
    Random_state=[525478,7505,32578,3257,2567,4789,8547,657,42]
    AverageAccuracy=[]
    k=0
    print('/'*60)
    for i in Test_size:
        for j in Random_state:
            print('/'*60)
            print("The Test_size",i,"and random state is",j) 
            X_train,X_test,y_train,y_test=train_test_split(Data1,Data2,test_size=i,random_state=j)
            cat=catboost.CatBoostClassifier(iterations=200,learning_rate=0.01,depth=2,loss_function='Logloss')
            predictModel=cat.fit(X_train,y_train)
            predictions=predictModel.predict(X_test)
            print(metrics.classification_report(y_test, predictions))
            print(metrics.confusion_matrix(y_test, predictions))
            F1score=metrics.classification_report(y_test, predictions).split()[-2]
            F1=float(F1score)
            AverageAccuracy.append(F1)
            k=k+1
            print("Accuracy is ",F1score)
    return(k,AverageAccuracy)             
                                   
length,AverageAcc=CatboostClassifier(X,y)
Sum_of_Acc_Dtree=sum(AverageAcc)
print("The Average of All acuracies",(Sum_of_Acc_Dtree/length))