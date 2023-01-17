# Importing the libraries
import pandas as pd        # for data manipulation
import seaborn as sns      # for statistical data visualisation
import numpy as np         # for linear algebra
import matplotlib.pyplot as plt      # for data visualization
from scipy import stats        # for calculating statistics

# Importing various machine learning algorithm from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,roc_curve,auc,accuracy_score
from  sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
dataframe= pd.read_csv("Bank_Personal_Loan_Modelling.csv")  # Reading the data
dataframe.head()   # showing first 5 datas
dataframe.shape
dataframe.info()
dataframe.apply(lambda x: len(x.unique()))
dataframe.iloc[:,1:].describe()
sns.pairplot(dataframe.iloc[:,1:])
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
sns.boxplot(dataframe.Experience)
plt.subplot(3,1,2)
sns.boxplot(dataframe.Income)
plt.subplot(3,1,3)
sns.boxplot(dataframe.CCAvg)
dataframe.iloc[:,1:9].skew()
dataframe.Experience[dataframe.Experience<0].count()
neg_ids=dataframe.loc[dataframe.Experience<0].ID.tolist()
pos_exp_data=dataframe.loc[dataframe.Experience>0]
for i in neg_ids:
    education=dataframe.Education[dataframe.ID==i].tolist()[0]
    age=dataframe.Age[dataframe.ID==i].tolist()[0]
    pos_record=pos_exp_data[(pos_exp_data.Age==age) & (pos_exp_data.Education==education)]
    x=pos_record['Experience'].median()
    dataframe.loc[(dataframe.ID==i),'Experience']=x
    
dataframe.Experience[dataframe.Experience<0].count()
dataframe.Experience.describe()
dataframe["Personal Loan"].hist(bins=2)
dataframe["Personal Loan"].value_counts()
sns.countplot(x='Education',data=dataframe,hue='Personal Loan')
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=dataframe)
sns.countplot(x="Family", data=dataframe,hue="Personal Loan")
fs_takenloan = np.mean( dataframe[dataframe['Personal Loan']== 0].Family )
fs_nottaken_loan = np.mean( dataframe[dataframe['Personal Loan'] == 1].Family )
print("Family size of those taken loan  is",fs_takenloan )
print("Family size of those not taken loan  is",fs_nottaken_loan )

stats.ttest_ind(dataframe[dataframe['Personal Loan'] == 1]['Family'], dataframe[dataframe['Personal Loan'] == 1]['Family'])
sns.boxplot(x='Family',y='Income',data=dataframe,hue='Personal Loan')
sns.boxplot(x='Education',y='Mortgage',data=dataframe,hue='Personal Loan')
sns.countplot(x='Securities Account',data=dataframe,hue='Personal Loan')
dataframe[dataframe['Personal Loan']==1].CCAvg.hist()
dataframe[dataframe['Personal Loan']==0].CCAvg.hist()
sns.distplot(dataframe[dataframe['Personal Loan']==1].CCAvg)
sns.distplot(dataframe[dataframe['Personal Loan']==0].CCAvg)
corelation=dataframe.corr()
plt.figure(figsize=(20,20))
a=sns.heatmap(corelation,annot=True)


dataframe.columns
features=['Age', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']
X=dataframe[features]
Y=dataframe['Personal Loan']         
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 
train_X.head()
test_X.count()
test_X.head()
LR_Model=LogisticRegression()
Logestic_Model=LR_Model.fit(train_X,train_y)
Logestic_Model
predict=LR_Model.predict(test_X)
print(predict[0:1000])
metrics=confusion_matrix(test_y,predict)
metrics
sns.heatmap(metrics,annot=True,fmt='g',cmap='Blues')
print(classification_report(test_y,predict))
probability=Logestic_Model.predict_proba(test_X)
pred=probability[:,1]
fpr,tpr,thresh=roc_curve(test_y,pred)
roc_auc=auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='AUC =%0.2f'%roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
LR_accuracy=accuracy_score(test_y,predict)
LR_accuracy
LR_AUC=roc_auc
LR_AUC
LR_Gini = 2*roc_auc - 1
LR_Gini

X=dataframe[features].apply(zscore)
Y=dataframe['Personal Loan']
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
KNN_Model=KNeighborsClassifier()
KNearestN_Model=KNN_Model.fit(train_X,train_y)
KNearestN_Model
predict=KNN_Model.predict(test_X)
predict[0:200,]
n=[1,3,5,7,11,13,15,17,19,21,23,25,27,29,31,33,35]
accuracy_scores=[]
for i in n:
    KNN_Model=KNeighborsClassifier(n_neighbors=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores
    
p=[1,2]
accuracy_scores=[]
for i in p:
    KNN_Model=KNeighborsClassifier(n_neighbors=5,p=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores

    
KNN_Model=KNeighborsClassifier(n_neighbors=5,p=2)  
KNN_Model.fit(train_X,train_y)
predict=KNN_Model.predict(test_X)
print(predict[0:200,])
Knn_matrics=confusion_matrix(test_y,predict)
Knn_matrics
print(classification_report(test_y,predict))
sns.heatmap(Knn_matrics,annot=True,cmap='Blues',fmt='g')
probs = KNN_Model.predict_proba(test_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
KNN_accuracy=accuracy_score(test_y,predict)
KNN_accuracy
KNN_Gini=2*roc_auc-1
KNN_Gini
KNN_AUC=roc_auc
KNN_AUC
X=dataframe[features]
Y=dataframe['Personal Loan']
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
NB_Model=GaussianNB()
naiveB_Model=NB_Model.fit(train_X,train_y)
naiveB_Model
predict=NB_Model.predict(test_X)
predict[0:200,]
ac_score=accuracy_score(test_y,predict)
ac_score
print(classification_report(test_y,predict))
NB_matrics=confusion_matrix(test_y,predict)
NB_matrics
sns.heatmap(NB_matrics,annot=True,cmap='Blues',fmt='g')
probs=NB_Model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
NB_accuracy=accuracy_score(test_y,predict)
NB_accuracy
NB_Gini=2*roc_auc-1
NB_Gini
NB_AUC=roc_auc
NB_AUC
data=[[LR_accuracy,LR_Gini,LR_AUC],[KNN_accuracy,KNN_Gini,KNN_AUC],[NB_accuracy,NB_Gini,NB_AUC]]
comparison=pd.DataFrame(data,index=['Logestic','KNN','Naive Bayes'],columns=['Accuracy','Gini','AUC'])
comparison