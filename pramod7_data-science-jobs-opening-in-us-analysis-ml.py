# Import all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
# Load CSV file into a DataFrame
raw_data=pd.read_csv('../input/alldata.csv')
#PART 1, Data Analysis
#Now, we are finding total number of compnaies who require data scientist
total_no_company=raw_data['company'].nunique()
print('Toatl number of firms with data science job vacancies',total_no_company)

#finding highest number of vacancy in a company
most_vacancy= raw_data.groupby(['company'])['position'].count()
most_vacancy=most_vacancy.reset_index(name='position')
most_vacancy=most_vacancy.sort_values(['position'],ascending=False)
pareto_df=most_vacancy
most_vacancy=most_vacancy.head(25)
print('Top 10 firms with most vacancies',most_vacancy)

# Plot graph for top most vacancy
fig, ax = plt.subplots(figsize = (10,6))
ax=seaborn.barplot(x="company", y="position", data=most_vacancy)    
ax.set_xticklabels(most_vacancy['company'],rotation=90)  
ax.set_xlabel('COMPANY WITH MOST JOBS',fontsize=16, color='red')
ax.set_ylabel('# OF JOBS',fontsize=16,color='red') 
# Finding total number of unique roles in data science domain from the given dataset
total_no_roles=raw_data['position'].nunique()
print('Toatl number of roles across all the firms',total_no_roles)

# most offered roles across all the firms
most_offd_roles=raw_data.groupby(['position'])['company'].count()   
most_offd_roles=most_offd_roles.reset_index(name='company')
most_offd_roles=most_offd_roles.sort_values(['company'],ascending=False)
most_offd_roles=most_offd_roles.head(30)   
print('Top 15 most wanted roles across firms',most_offd_roles)

# Plot graph for top most offered roles
fig,ax=plt.subplots(figsize=(12,6))
ax=seaborn.barplot(x="position", y="company", data=most_offd_roles)    
ax.set_xticklabels(most_offd_roles['position'],rotation=90)
ax.set_xlabel('MOST WANTED JOB ROLES',fontsize=20,color='red')
ax.set_ylabel('NO OF ROLES ACROSS INDUSTRY',fontsize=16,color='red')#
# Finding total number of cities with Data science jobs
total_no_cities=raw_data['location'].nunique()

#cities and total no of openings w.r.t companies
city_and_roles=raw_data.groupby(['location','company'])['position'].count()     
city_and_roles=city_and_roles.reset_index()
city_and_roles=city_and_roles.sort_values(['position'],ascending=False)
city_and_roles=city_and_roles.head(15) 

# Plot graph for top most cities and no of roles
fig,a=plt.subplots(figsize=(10,6))             
a=seaborn.barplot(x="company", y="position", hue="location", data=city_and_roles);    
a.set_xticklabels(city_and_roles['company'],rotation=90)   
a.set_ylabel('No Of Positions',fontsize=16,color='red')
a.set_xlabel('Company Name',fontsize=16,color='red')   
#PART 2, PARETO CHART
# trying to find if the given data set follows Pareto(80,20) rule

#find the total job openings
total_job_openings=len(raw_data['position'])

#find 70% total job (total_job_openings) openings, pareto rule,can be (80,20) or (70,20)
sum_70_percent_job_openings=total_job_openings/100*70

#find 20% total number of companies
sum_20_percent_companies=total_no_company/100*20

#now find the total number of job openings from those 20% top comapnies 
top_20_companies_job_openings=pareto_df.head(443)

sum_=top_20_companies_job_openings['position'].sum()

print('70% of the total job openings is :',sum_70_percent_job_openings )

print('total job openings from top 20% of the companies :',sum_)

print ('so, 70% of the total job openings and total job openings from top 20% of the companies are almost equal. Therefore we can say the data set follows Pareto Rule')

#PART 3 MACHINE LEARNING
#there are so many job profiles in teh given dataset so lets Categories them into 5; Data Scientist, Machine Learning Engineer, Data Analyst, Data Science Manager and Others

# Creating only 5 datascience roles among all
data=raw_data.copy()
data.dropna(subset=['position'], how='all', inplace = True)
data['position']=[x.upper() for x in data['position']]
data['description']=[x.upper() for x in data['description']]

data.loc[data.position.str.contains("SCIENTIST"), 'position'] = 'Data Scientist'

data.loc[data.position.str.contains('ENGINEER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('PRINCIPAL STATISTICAL PROGRAMMER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('PROGRAMMER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('DEVELOPER'),'position']='Machine Learning Engineer'

data.loc[data.position.str.contains('ANALYST'), 'position'] = 'Data Analyst'
data.loc[data.position.str.contains('STATISTICIAN'), 'position'] = 'Data Analyst'

data.loc[data.position.str.contains('MANAGER'),'position']='Data Science Manager'
data.loc[data.position.str.contains('CONSULTANT'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DATA SCIENCE'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DIRECTOR'),'position']='Data Science Manager'

data.position=data[(data.position == 'Data Scientist') | (data.position == 'Data Analyst') | (data.position == 'Machine Learning Engineer') | (data.position == 'Data Science Manager')]
data.position=['Others' if x is np.nan else x for x in data.position]

position=data.groupby(['position'])['company'].count()   
position=position.reset_index(name='company')
position=position.sort_values(['company'],ascending=False)

print('Here is  the count of each new roles we created :', '\n\n', position)

# Next Part in ML Algorithm is Data Cleansing
X=data.description
Y=data.position

X=[re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in X]
X=[re.sub("[0-9]+",' ',k) for k in X]

#applying stemmer
ps =PorterStemmer()
X=[ps.stem(k) for k in X]

#Note: I have not removed stop words because there are important key words mentioned in job description which are of length 2, I feel they have weightage while classifing
tfidf=TfidfVectorizer()
label_enc=LabelEncoder()

X=tfidf.fit_transform(X)
Y=label_enc.fit_transform(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y,test_size=0.3)

# first algorithm SVM
#SVM classification
#svm=SVC(kernel='rbf')
#svm.fit(x_train,y_train)

#svm_y=svm.predict(x_test)

#print('Accuracy of SVM :', accuracy_score(y_test,svm_y))
#print ('Confusion Matrix of SVM : ', '\n\n', confusion_matrix(y_test,svm_y))

#crossfold Validation of 7 folds for SVM
#cross_val_SVM=sklearn.model_selection.cross_validate(svm, x_train, y=y_train,cv=7)

#print ('SVM Train fit score is : ', '\n\n', cross_val_SVM ['train_score'])
#print ('SVM TEST score is : ', '\n\n', cross_val_SVM ['test_score'])
#Naive Bayes classification
NB=MultinomialNB()
NB.fit(x_train,y_train)
NB_y=NB.predict(x_test)

print('Accuracy of NB :', accuracy_score(y_test,NB_y))
print ('Confusion Matrix of NB : ', '\n\n', confusion_matrix(y_test,NB_y))

#crossfold Validation of 7 folds for NB
cross_val_NB=sklearn.model_selection.cross_validate(NB, x_train, y=y_train,cv=7)

print ('NB Train fit score is : ', '\n\n', cross_val_NB ['train_score'])
print ('NB TEST score is : ', '\n\n', cross_val_NB ['test_score'])
#3rd Classifier SGDC
#SGD classification
sgd=SGDClassifier()
sgd.fit(x_train,y_train)
sgd_y=sgd.predict(x_test)

print('Accuracy of SGD :', accuracy_score(y_test,sgd_y))
print ('Confusion Matrix of SGD : ', '\n\n', confusion_matrix(y_test,sgd_y))

#crossfold Validation of 7 folds for SGD
cross_val_SGD=sklearn.model_selection.cross_validate(sgd, x_train, y=y_train,cv=7)

print ('SGD Train fit score is : ', '\n\n', cross_val_SGD ['train_score'])
print ('SGD TEST score is : ', '\n\n', cross_val_SGD ['test_score'])
#4th Classifier 
#XGBOOST classification
#xgboost=GradientBoostingClassifier(n_estimators=90)
#xgboost.fit(x_train,y_train)
#xgboost_y=xgboost.predict(x_test)

#print('Accuracy of XGBOOST :', accuracy_score(y_test,xgboost_y))
#print ('Confusion Matrix of XGBOOST : ', '\n\n', confusion_matrix(y_test,xgboost_y))

#crossfold Validation of 7 folds for SGD
#cross_val_xgboost=sklearn.model_selection.cross_validate(xgboost, x_train, y=y_train,cv=7)

#print ('XGBOOST Train fit score is : ', '\n\n', cross_val_xgboost ['train_score'])
#print ('XGBOOST TEST score is : ', '\n\n', cross_val_xgboost ['test_score']) 
# Inverse Transform of label Encoder
print (label_enc.inverse_transform([0,1,2,3,4]))
"""END NOTES

•	Graphs can be improved, these are designed for beginner purpose

•	Pareto graph is not plotted since I had issues with my Work laptop to install certain packages, but I have given the numbers which means mostly the same

•	ML can be improved with the following ideas;

        Categories the data in a better way, so that they are much meaningful
        Balance the categories so that we will not have any skewness as we have here
        Do not run SVM or XGBOOST if you do  not have an hour of patience
        Rather run Naïve Bayes and SGDC which are almost performing same as SVM and XGBOOST respectively
        The fit is not good in SVM & Naïve Bayes because the categories (Job Description) are very closely correlated and may be due to biased data 
        Look at the confusion matrix of SVM and NB, clearly due to biase the data is pulled towards DATA SCIENTIST class
        I have tried to change kernel and otehr parameters for SVM & NB but no improvements
        XGBOOST & SGDC are doing good with 79-80% accuracy consistently but we should look at the confusion matrix and try to improve recall and precision
        
        
•	Please drop a note if you feel I can do any better work 

You can contact me on
mmpramod7@gmail.com or
Pramod Manjegowda @ linkedIn 


THANKS A TON """

