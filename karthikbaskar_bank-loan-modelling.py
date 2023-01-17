import numpy as np # Numpy for neumeric algebra{

import pandas as pd # pandas for data 

import seaborn as sns # seaborn and matplotlib for data visualization 

from matplotlib import pyplot as plt
df=pd.read_excel("../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx","Data")
df.head()
print("There are {} rows & {} columns in our dataset.".format(df.shape[0],df.shape[1]))
df.info()
print("+++++++++++Check for null values in our dataset+++++++++++")



print(df.isna().apply(pd.value_counts))
df.drop(['ID','ZIP Code'],axis=1,inplace=True)
df.describe().T
df.boxplot(return_type='axes',figsize=(10,8),column=['Age','Experience','Income','Family','Education']);
df.skew(numeric_only=True)
columns=list(df)

df[columns].hist(stacked=True,density=True, bins=100,color='Orange', figsize=(16,30), layout=(10,3)); 
sns.distplot(df['Age'],kde=True,hist=False,color='Red')

sns.distplot(df['Income'],kde=True,hist=False,color='Green')

sns.distplot(df['Experience'],kde=True,hist=False,color='blue')

plt.show()
Negative_values_in_exp=df[df['Experience']< 0]

sns.distplot(Negative_values_in_exp['Age'],kde=True,label="Distribution of Age which has negative values");

avg_exp=df['Experience'].mean()

print("The Average exp is : {}".format(avg_exp))

N1=Negative_values_in_exp[Negative_values_in_exp['Age'].between(22,31)]

avg_exp2=N1['Experience'].mean()

positive_values_for_age_group=df[df['Experience']>0]

get_mean_value=positive_values_for_age_group[positive_values_for_age_group['Age'].between(22,31)]

mean_value=get_mean_value['Experience'].mean()



print("The Avg exp for people in this age grouop:{}".format(avg_exp2))

perct=df.shape[0]

perct=Negative_values_in_exp.shape[0]/perct

perct = perct * 100

print("There are {} records which has negative values for Experience, approx {}% among age group between 22 to 30".format(Negative_values_in_exp.shape[0],perct))

print("The mean value that we can use is :{} ".format(mean_value))



# I am using mask function to change the negative values to mean value derived from data with the same age group 

df['Experience']=df['Experience'].mask(df['Experience']<0,mean_value)    

print("After updating the negative records we get the mean value of exp to be:{}".format(df['Experience'].mean()))
def plot_corr(df, size=10):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, 25))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(df)
df=df.drop(['Experience'],axis=1)
def edu(row):

    if row['Education']==1:

        return "Undergrad"

    elif row['Education']==2:

        return "Graduate"

    else:

        return "Advanced/Professional"

df['EDU']=df.apply(edu,axis=1)
EDU_dis=df.groupby('EDU')["Age"].count()

EDU_dis.plot.pie(shadow=True, startangle=170,autopct='%.2f')
def SD_CD(row):

    if (row['Securities Account']==1) & (row['CD Account']==1):

        return"Holds Securities & deposit"

    elif(row['Securities Account']==0) & (row['CD Account']==0):

        return"Does not hold any securities or deposit"

    elif(row['Securities Account']==1) & (row['CD Account']==0):

        return "Holds only Securities Account"

    elif(row['Securities Account']==0) & (row['CD Account']==1):

        return"Holds only deposit"  
df['Account_Holder_Category']=df.apply(SD_CD,axis=1)
df['Account_Holder_Category'].value_counts().plot.pie(shadow=True, startangle=125,autopct='%.2f')
sns.boxplot(df['Education'],df['Income'],hue=df['Personal Loan']);
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan']==0]['Income'],kde=True,color='r',hist=False,label="Income distribution for customers with no personal Loan")

sns.distplot(df[df['Personal Loan']==1]['Income'],kde=True,color='G',hist=False,label="Income distribution for customers with personal Loan")

plt.legend()

plt.title("Income Distribution")
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan']==0]['CCAvg'],kde=True,hist=False,color='r',label="Credit card average for customers with no personal Loan")

sns.distplot(df[df['Personal Loan']==1]['CCAvg'],kde=True,hist=False,color='G',label="Credit card average for customers with personal Loan")

plt.legend()

plt.title("CCAvg Distribution")
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan']==0]['Mortgage'],kde=True,hist=False,color='r',label="Mortgage of customers with no personal Loan")

sns.distplot(df[df['Personal Loan']==1]['Mortgage'],kde=True,hist=False,color='G',label="Mortgage of customers with personal Loan")

plt.legend()

plt.title("Mortgage Distribution")
col_names=['Securities Account','Online','CreditCard']



for i in col_names:

    plt.figure(figsize=(14,12))

j=2

k=0

for i in col_names:

    plt.subplot(2,j,j*(k+1)//j)

    sns.countplot(x=i,hue='Personal Loan',palette="Blues", data=df)

    k=k+1

    plt.grid(True)

plt.show()

plt.figure(figsize=(14,12))

sns.countplot(df['Account_Holder_Category'],hue=df['Personal Loan'],palette='Blues')

plt.show();
df.head()
Data=df.drop(['EDU','Account_Holder_Category'],axis=1)
Data.head()
Data.describe().T
import scipy.stats as stats

sns.scatterplot(Data['Age'],Data['Personal Loan'],hue=Data['Family'],alpha=0.8);



#Frame the Hypothesis

H0="Age does not have any impact on availing personal Loan"

Ha="Age does have phenomenal significance on availing personal Loan"



Age_PL_Yes=np.array(Data[Data['Personal Loan']==1].Age)

Age_PL_No=np.array(Data[Data['Personal Loan']==0].Age)



t,p_value=stats.ttest_ind(Age_PL_Yes,Age_PL_No,axis=0)



if p_value < 0.05:

    #We reject Null

    print(Ha,"As the P_value is less than 0.05 with a value of :{}".format(p_value))

else:

    #We fail to reject Null

    print (H0,"As the P_value is Greater than 0.05 with a value of :{}".format(p_value))
sns.scatterplot(Data['Age'],Data['Personal Loan'],hue=Data['Income'],alpha=0.8);

Income_PL_Yes=np.array((Data[Data['Personal Loan']==1]).Income)

Income_PL_No=np.array((Data[Data['Personal Loan']==0]).Income)



H0="Income of a person does not have an impact on availing Personal Loan"

Ha="Income of a person has significant impact on availing Personal Loan"



t,p_value=stats.ttest_ind(Income_PL_Yes,Income_PL_No,axis=0)



if p_value < 0.05:

    #Reject Null

    print(Ha,"As the P_value is less than 0.05 with a value of :{}".format(p_value))

    print("As you can see from the plot, those who availed Personal Loan tend to have higher income")

else:

    #We fail to reject Null

    print (H0,"As the P_value is Greater than 0.05 with a value of :{}".format(p_value))

    
sns.scatterplot(Data['Age'],Data['Personal Loan'],hue=Data['Family'],alpha=0.8);

Family_PL_Yes=np.array((Data[Data['Personal Loan']==1]).Family)

Family_PL_No=np.array((Data[Data['Personal Loan']==0]).Family)



H0="Number of persons in the family does not have an impact on availing Personal Loan"

Ha="Number of persons in the family has significant impact on availing Personal Loan"



t,p_value=stats.ttest_ind(Family_PL_Yes,Family_PL_No,axis=0)



if p_value < 0.05:

    #Reject Null

    print(Ha,"As the P_value is less than 0.05 with a value of :{}".format(p_value))

    #print("As you can see from the plot, those who availed Personal Loan tend to have higher income")

else:

    #We fail to reject Null

    print (H0,"As the P_value is Greater than 0.05 with a value of :{}".format(p_value))
TAB=pd.crosstab(Data['Education'],Data['Personal Loan'])

chi,p_value,dof,expected=stats.chi2_contingency(TAB)



H0="Educational qualification does not have an influence on a person to avail Loan"

Ha="Educational qualification does have an influence on a person to avail Loan"



if p_value < 0.05:

    #Reject Null

    print(Ha)

else:

    #Fail to reject Null

    print(H0)
print(Data['Personal Loan'].value_counts())

No_of_customers_availed_PL=Data[Data['Personal Loan']==1].shape[0]

No_of_customers_availed_PL

Total_Cust=Data.shape[0]

percet=(No_of_customers_availed_PL * 100)/Total_Cust 

print("Overall percentage of customers who have availed personal Loan:{}".format(percet),"%")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



#Split our Data into Dependent Variables & Independent Variables 



X=Data.drop(['Personal Loan'],axis=1)

y=Data['Personal Loan']

#Split dataset into train and test, as suggested this method will 



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)
Model1_raw=LogisticRegression(solver='liblinear')

Model1_raw.fit(X_train,y_train)
Model1_raw_coef=pd.DataFrame(Model1_raw.coef_)

Model1_raw_coef
Model1_raw.score(X_train,y_train)
Model1_raw.score(X_test,y_test)
Model1_raw_prediction=Model1_raw.predict(X_test)
chk_model1=pd.DataFrame({"Actual":y_test,"Predicted": Model1_raw_prediction})

Top=chk_model1.nlargest(25,'Predicted')

Top.plot(kind='bar',figsize=(15,10));

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
cm_model1=confusion_matrix(y_test,Model1_raw_prediction,labels=[0,1])

print(cm_model1)

acc_score_log1=accuracy_score(y_test,Model1_raw_prediction)

f1_score_log1=f1_score(y_test,Model1_raw_prediction)

print("Accuracy Score  for Logistic Regression RAW DATA:{}".format(acc_score_log1*100))

print("F1 Score  for Logistic Regression RAW DATA:{}".format(f1_score_log1*100))

print(classification_report(y_test,Model1_raw_prediction))
from  sklearn.utils import resample

df_majority=Data[Data['Personal Loan']==0]

df_minority=Data[Data['Personal Loan']==1]

df_upsample_minority=resample(df_minority,replace=True,random_state=12,n_samples=4520)

df_upsample=pd.concat([df_majority,df_upsample_minority])

count=df_upsample['Personal Loan'].value_counts()

count.plot(kind='bar',figsize=(5,5));
X_upsampled=df_upsample.drop(['Personal Loan'],axis=1)

Y_upsampled=df_upsample['Personal Loan']
X_upsampled_test,X_upsampled_train,Y_upsampled_test,Y_upsampled_train=train_test_split(X_upsampled,Y_upsampled,random_state=1,test_size=0.3)
Model1_raw.fit(X_upsampled_train,Y_upsampled_train)
Pred_Upsample_Log=Model1_raw.predict(X_upsampled_test)

cm_model1_upsample=confusion_matrix(Y_upsampled_test,Pred_Upsample_Log,labels=[0,1])
acc_score_log1=accuracy_score(y_test,Model1_raw_prediction)

f1_score_log1=f1_score(y_test,Model1_raw_prediction)

print("Accuracy Score  for Logistic Regression with RAW DATA:{}".format(acc_score_log1*100))

print("F1 Score  for Logistic Regression with RAW DATA:{}".format(f1_score_log1*100))

print("+++++++++++++++THE CONFUSION MATRIX LOGISTIC REGRESSION  WITH RAW DATA++++++++++++++++")

print("Confusion Matrix: \n",cm_model1)

print("+++++++++++++++CLASSIFICATION REPORT LOGISTIC REGRESSION WITH RAW DATA++++++++++++++++")

print(classification_report(y_test,Model1_raw_prediction))

acc_score_upsampled=accuracy_score(Y_upsampled_test,Pred_Upsample_Log)

f1_score_upsampled=f1_score(Y_upsampled_test,Pred_Upsample_Log)

print("Accuracy Score  for Logistic Regression Upsampled Date:{}".format(acc_score_upsampled*100))

print("F1 Score  for Logistic Regression:{}".format(f1_score_upsampled*100))

print("+++++++++++++++THE CONFUSION MATRIX LOGISTIC REGRESSION WITH Upsampled Data+++++++++++++++++")

print("Confusion Matrix: \n",cm_model1_upsample)

print("+++++++++++++++CLASSIFICATION REPORT LOGISTIC REGRESSION WITH Upsampled Data++++++++++++++++")

print(classification_report(Y_upsampled_test,Pred_Upsample_Log))
from sklearn.naive_bayes import GaussianNB

Model2_nb=GaussianNB()
Model2_nb.fit(X_upsampled_train,Y_upsampled_train)

Model2_nb.fit(X_train,y_train)
Model2_nb.fit(X_upsampled_test,Y_upsampled_test)

Model2_nb.fit(X_test,y_test)
print("####################Raw Data Score####################")

print(Model2_nb.score(X_train,y_train))

print(Model2_nb.score(X_test,y_test))

print("####################Sample Data Score####################")

print(Model2_nb.score(X_upsampled_train,Y_upsampled_train))

print(Model2_nb.score(X_upsampled_test,Y_upsampled_test))
Pred_nb_raw=Model2_nb.predict(X_test)

Pred_nb_Upsampled=Model2_nb.predict(X_upsampled_test)

CM_NB_RAW=confusion_matrix(y_test,Pred_nb_raw)

CM_NB_UPSAMPLE=confusion_matrix(Y_upsampled_test,Pred_nb_Upsampled)
acc_score_NB_Raw=accuracy_score(y_test,Pred_nb_raw)

f1_score_NB_raw=f1_score(y_test,Pred_nb_raw)

print("Accuracy Score  for Naive Bayes Model RAW DATA:{}".format(acc_score_NB_Raw*100))

print("F1 Score  for Naive Bayes Model RAW DATA:{}".format(f1_score_NB_raw*100))

print("+++++++++++++++THE CONFUSION MATRIX RAW DATA++++++++++++++++")

print("Confusion Matrix: \n",CM_NB_RAW)

print("+++++++++++++++CLASSIFICATION REPORT RAW DATA++++++++++++++++")

print(classification_report(y_test,Pred_nb_raw))

acc_score_NB_upsampled=accuracy_score(Y_upsampled_test,Pred_nb_Upsampled)

f1_score_NB_upsampled=f1_score(Y_upsampled_test,Pred_nb_Upsampled)

print("Accuracy Score  for Naive Bayes Model Upsampled Data:{}".format(acc_score_NB_upsampled*100))

print("F1 Score  for Naive Bayes Model:{}".format(f1_score_NB_upsampled*100))

print("+++++++++++++++THE CONFUSION MATRIX Upsampled Data++++++++++++++++")

print("Confusion Matrix: \n",CM_NB_UPSAMPLE)

print("+++++++++++++++CLASSIFICATION REPORT Upsampled Data++++++++++++++++")

print(classification_report(Y_upsampled_test,Pred_Upsample_Log))
print("#############Inference from Logistic Regression & Naive bayes Model###############")

#print("Precision score of Logistic Regression is 84% , whereas Precsion of Naive bayes is 44%\nThis means,out of all customers who are likely to buy personal loan Logistic model predicted 84% to be right.\nRecall score for both Logistic and Naive bayes model is same (57%)\nThis means out of all customers who will buy loan both models predicted only 57% correctly, which means both miss to provide insight about the other 43% data which is crucial")



if (acc_score_log1 > acc_score_upsampled):

    print("Accuracy score of Logistic Regression with Raw Data is higher than Upsampled Data")

else:

    print("Accuracy score of Logistic Model with Upsampled Data is greater")

if (f1_score_log1 > f1_score_upsampled):

    print("F1 score of Logistic model with Raw Data is higher than Upsampled Data")

else:

    print("F1 score of Logistic Model with Upsampled Data is greater")

   



        
from sklearn.neighbors import KNeighborsClassifier
Model3_KNN=KNeighborsClassifier(n_neighbors=11)
Model3_KNN.fit(X_train,y_train)
Predict_knn=Model3_KNN.predict(X_test)
CM_KNN=confusion_matrix(y_test,Predict_knn)
Model3_KNN.fit(X_upsampled_train,Y_upsampled_train)
Predict_KNN_Sampled=Model3_KNN.predict(X_upsampled_test)
CM_KNN_SAMPLED=confusion_matrix(Y_upsampled_test,Predict_KNN_Sampled)
acc_score_KNN_Raw=accuracy_score(y_test,Predict_knn)

f1_score_KNN_raw=f1_score(y_test,Predict_knn)

print("Accuracy Score  for KNN with RAW DATA:{}".format(acc_score_KNN_Raw*100))

print("F1 Score for KNN with RAW DATA:{}".format(f1_score_KNN_raw*100))

print("+++++++++++++++THE CONFUSION MATRIX for KNN RAW DATA++++++++++++++++")

print("Confusion Matrix: \n",CM_KNN)

print("+++++++++++++++CLASSIFICATION REPORT for KNN RAW DATA++++++++++++++++")

print(classification_report(y_test,Predict_knn))

acc_score_KNN_upsampled=accuracy_score(Y_upsampled_test,Predict_KNN_Sampled)

f1_score_KNN_upsampled=f1_score(Y_upsampled_test,Predict_KNN_Sampled)

print("Accuracy Score  for KNN with Upsampled Data:{}".format(acc_score_KNN_upsampled*100))

print("F1 Score  for for KNN with Upsampled Data:{}".format(f1_score_KNN_upsampled*100))

print("+++++++++++++++THE CONFUSION MATRIX KNN Upsampled Data++++++++++++++++")

print("Confusion Matrix: \n",CM_KNN_SAMPLED)

print("+++++++++++++++CLASSIFICATION REPORT KNN Upsampled Data++++++++++++++++")

print(classification_report(Y_upsampled_test,Predict_KNN_Sampled))
#Import metrics



from sklearn.metrics import roc_curve,auc,roc_auc_score



#Perform Predictions for all models both with raw data and upsampled data



PRED_PROB_LOG_RAW=Model1_raw.predict_proba(X_test)

PRED_PROB_LOG_SAMPLED=Model1_raw.predict_proba(X_upsampled_test)



PRED_PROB_NB_RAW=Model2_nb.predict_proba(X_test)

PRED_PROB_NB_SAMPLED=Model2_nb.predict_proba(X_upsampled_test)



PRED_PROB_KNN_RAW=Model3_KNN.predict_proba(X_test)

PRED_PROB_KNN_SAMPLED=Model3_KNN.predict_proba(X_upsampled_test)



#calculate fpr,tpr,threshold

fpr1, tpr1, thresh1 = roc_curve(y_test, PRED_PROB_LOG_RAW[:,1], pos_label=1)

fpr2,tpr2,thresh2= roc_curve(Y_upsampled_test,PRED_PROB_LOG_SAMPLED[:,1],pos_label=1)

fpr3,tpr3,thresh3=roc_curve(y_test,PRED_PROB_NB_RAW[:,1],pos_label=1)

fpr4,tpr4,thresh4=roc_curve(Y_upsampled_test,PRED_PROB_NB_SAMPLED[:,1],pos_label=1)

fpr5,tpr5,thresh5=roc_curve(y_test,PRED_PROB_KNN_RAW[:,1],pos_label=1)

fpr6,tpr6,thresh6=roc_curve(Y_upsampled_test,PRED_PROB_KNN_SAMPLED[:,1],pos_label=1)





random_probs = [0 for i in range(len(y_test))]

p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)





AUC_LOG_RAW=roc_auc_score(y_test,PRED_PROB_LOG_RAW[:,1])

AUC_LOG_SAMPLED=roc_auc_score(Y_upsampled_test,PRED_PROB_LOG_SAMPLED[:,1])

AUC_NB_RAW=roc_auc_score(y_test,PRED_PROB_NB_RAW[:,1])

AUC_NB_UPSAMPLED=roc_auc_score(Y_upsampled_test,PRED_PROB_NB_SAMPLED[:,1])

AUC_KNN_RAW=roc_auc_score(y_test,PRED_PROB_KNN_RAW[:,1])

AUC_KNN_UPSAMPLED=roc_auc_score(Y_upsampled_test,PRED_PROB_KNN_SAMPLED[:,1])



AUC_SCORES=pd.array([AUC_LOG_RAW,AUC_LOG_SAMPLED,AUC_NB_RAW,AUC_NB_UPSAMPLED,AUC_KNN_RAW,AUC_KNN_UPSAMPLED])







#Plot Area Under Curve



plt.plot(fpr1,tpr1,linestyle='--',color='orange', label='Logistic Regression RAW')

plt.plot(fpr2,tpr2,linestyle='solid',color='blue', label='Logistic Regression Sampled')

plt.plot(fpr3,tpr3,linestyle='--',color='Green', label='Naive Bayes RAW')

plt.plot(fpr4,tpr4,linestyle='solid',color='Red', label='Naive Bayes Sampled')

plt.plot(fpr5,tpr5,linestyle='--',color='violet',label='KNN RAW')

plt.plot(fpr6,tpr6,linestyle='solid',color='black',label='KNN Upsampled')



plt.plot(p_fpr, p_tpr, linestyle=':', color='red')

# title

plt.title('ROC curve')

# x label

plt.xlabel('False Positive Rate')

# y label

plt.ylabel('True Positive rate')



plt.legend(loc='best')

plt.savefig('ROC',dpi=300)

plt.show();



print("ROC_AUC_Score for Logistic Regression with Raw Data:{}".format(AUC_LOG_RAW))

print("ROC_AUC_Score for Logistic Regression with Upsample  Data:{}".format(AUC_LOG_SAMPLED))

print("ROC_AUC_Score for Naive Bayes with Raw Data:{}".format(AUC_NB_RAW))

print("ROC_AUC_Score for Naive Bayes with Upsample  Data:{}".format(AUC_NB_UPSAMPLED))

print("ROC_AUC_Score for KNN with Raw Data:{}".format(AUC_KNN_RAW))

print("ROC_AUC_Score for KNN with Upsample  Data:{}".format(AUC_KNN_UPSAMPLED))  

print("=======================================================================")

print("The Best AUC_SCORE that we have got is :{}".format(AUC_SCORES.max()))
