#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import  DecisionTreeClassifier
from scipy.stats import zscore

%matplotlib inline
#reading the data-pandas dataframe
bankdata=pd.read_csv("/kaggle/input/portuguese-bank-marketing-data-set/bank-full.csv",sep=';')
bankdata.head() #first 5 records of file for sample
print("Shape",bankdata.shape) #Shape- no of rows and columns
print("Size",bankdata.size) #Size- number of elements in the data file
bankdata.info() #file attributes and metadata details
for col in bankdata.columns:
    if bankdata[col].dtype=='object':
        bankdata[col]= pd.Categorical(bankdata[col])
bankdata =bankdata.rename(columns={'y': 'Target'})
bankdata.info()
# checking for null values in the files
bankdata.isna().sum()
bankdata.describe() # Summary of numerical attributes of the file
bankdata.head(20) #check data for anamolies
# Missing values and categorical data treatment with convenient data for analysis

replace_struct={"marital": {"single":0 ,"married":1,"divorced":2},
                "contact": {"unknown":0,"telephone":1,"cellular":2},
                "poutcome":{"other":-1,"unknown":0,"success":1,"failure":2},
                "month": {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12},
                "default":{"yes":1, "no":0},
                "loan":{"yes":1, "no":0},
                "housing":{"yes":1, "no":0},
                "Target": {"no":0,"yes":1}
                }
                
df1=bankdata.replace(replace_struct)
df1
#checking the correlation between few columns of interest
corr_data= bankdata[['age','balance','day','duration','campaign','pdays','previous','Target']]
corr_data.corr()
df1.corr() # checking correlation between data on missing values and relevant data replacement
# checking the correlation 
corr_data= df1[['age','balance','duration','campaign','month','previous','Target']]
corr_data.corr()
# correlation matrix/ graph
plt.figure(figsize=(10,8))
sns.heatmap(corr_data.corr(), annot=True, fmt='0.3f', center=0,linewidths=.5)
#checking for outliers
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('age')
plt.boxplot(df1['age'])
plt.subplot(1,2,2)
plt.title('balance')
plt.boxplot(df1['balance'])
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('duration')
plt.boxplot(df1['duration'])
plt.subplot(1,2,2)
plt.title('campaign')
plt.boxplot(df1['campaign'])
df1['balance_zscore']=df1['balance']
df1
df1['balance_zscore']=zscore(df1['balance_zscore'])
df1.sample(12)
df1=df1.drop(df1[(df1['balance_zscore']>3)|(df1['balance_zscore']<-3)].index, axis=0, inplace=False)
print(df1.shape)
corr_data= df1[['age','balance','duration','campaign','month','previous','Target']]
print(corr_data.corr())
plt.figure(figsize=(10,8))
sns.heatmap(corr_data.corr(), annot=True, fmt='0.3f', center=0,linewidths=.5)
plt.style.use('ggplot')
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('age')
plt.hist(df1['age'],bins=8)
plt.subplot(1,2,2)
plt.title('balance')
plt.hist(df1['balance'],bins=8)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1) 
plt.title('duration')
plt.hist(df1['duration'])
plt.subplot(1,2,2)
plt.title('campaign')
plt.hist(df1['campaign'])
df1['campaign'].describe()
df1['duration'].describe()
sns.distplot(df1['month'])
sns.scatterplot(x='age',y='balance',data=df1)
plt.figure(figsize=(8,8))
sns.scatterplot(x='age',y='balance', data=df1, hue='Target')
plt.figure(figsize=(8,8))
 
sns.countplot(x='poutcome', hue='Target' , data=df1)
df2=df1.drop('balance_zscore',axis=1 )
df2
sns.countplot(x='default', hue='Target' , data=df2) #drop default
df2=df2.drop('default',axis=1)
df2
sns.countplot(x='loan', hue='Target' , data=df2)
sns.countplot(x='housing', hue='Target' , data=df2)
sns.countplot(x='contact', hue='Target' , data=df2)
plt.figure(figsize=(6,8))
sns.countplot(x='month', hue='Target' , data=df2)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(x='month',y='duration' , data=df1)
plt.subplot(1,2,2)
sns.barplot(x='month',y='duration', hue='Target', data=df1)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(x='month',y='campaign' , data=df1)
plt.subplot(1,2,2)
sns.barplot(x='month',y='campaign', hue='Target', data=df1)
print(df2.shape)
df2=df2.drop(df2[df2['education']=='unknown'].index, axis=0)
print(df2.shape)
df2['education'].unique()
plt.figure(figsize=(16,8))
sns.countplot(x='job', hue='Target' , data=df2) # can drop unknown values
print(df2.shape)
df2=df2.drop(df2[df2['job']=='unknown'].index, axis=0)
print(df2.shape)
df2['job'].unique()
plt.figure(figsize=(10,5))
sns.countplot(x='marital', hue='Target' , data=df2)
plt.figure(figsize=(20,5))
sns.barplot(x='day', y='duration' ,hue='Target' , data=df2)
plt.figure(figsize=(8,8))
sns.scatterplot(x='campaign',y='duration', data=df1, hue='Target')
df2
df2=df2.drop(['pdays','poutcome'],axis=1) #not significant in classifying the customer
df2.shape
oneHotcode =['job','education']
df2=pd.get_dummies(df2,columns=oneHotcode)
df2.shape
bankdataset=df2.drop(['job_unknown','education_unknown'],axis=1) 
bankdataset.shape
bankdataset.columns
x=bankdataset.drop(['Target'], axis=1)
Y=bankdataset['Target']
x_train, x_test, Y_train, Y_test = train_test_split(x,Y, train_size = 0.7, test_size = 0.3, random_state = 1)
LogRegModel= LogisticRegression(solver='sag',max_iter=10000)
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))
LogRegModel= LogisticRegression(solver='lbfgs', max_iter=10000)
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))
LogRegModel= LogisticRegression(solver='liblinear')
LogRegModel.fit(x_train,Y_train)
print(LogRegModel.score(x_train, Y_train))
print(LogRegModel.score(x_test, Y_test))
pred=LogRegModel.predict(x_test)
print(LogRegModel.intercept_)
print(LogRegModel.coef_)
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(ConfMat_DF, annot=True )
#AUC ROC curve
logit_roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, LogRegModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, LogRegModel.predict_proba(x_test)[:,1])
print("Logistic Regression AUC Score:",auc_score)

myList = list(range(1,200))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))
# empty list that will hold accuracy scores
ac_scores = []
rl_scores =[]

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, Y_train)
    # predict the response
    Y_pred = knn.predict(x_test)
    # evaluate accuracy
    scores = accuracy_score(Y_test, Y_pred)
    ac_scores.append(scores)
   # changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
KNNModel = KNeighborsClassifier(n_neighbors = 9)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))

KNNModel = KNeighborsClassifier(n_neighbors = 23)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))
KNNModel = KNeighborsClassifier(n_neighbors = 41)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))
KNNModel = KNeighborsClassifier(n_neighbors =95)

# fitting the model
KNNModel.fit(x_train,Y_train)

# predict the response
pred = KNNModel.predict(x_test)

# evaluate accuracy
 
print(KNNModel.score(x_test, Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True )
knn_roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, KNNModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('KNN_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, KNNModel.predict_proba(x_test)[:,1])
print("KNN AUC Score:",auc_score)
NBModel = GaussianNB()
NBModel.fit(x_train, Y_train)
pred= NBModel.predict(x_test)
print(NBModel.score(x_train, Y_train))
print(NBModel.score(x_test, Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, NBModel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Baye''s (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Naive_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Y_test, NBModel.predict_proba(x_test)[:,1])
print("Naive Bayes AUC Score:",auc_score)

# Building a Support Vector Machine on train data
SVCModel = SVC(C= .1, kernel='rbf', gamma= 'auto')
SVCModel.fit(x_train, Y_train)

pred= SVCModel.predict(x_test)
# check the accuracy on the training set
print(SVCModel.score(x_train, Y_train))
# check the accuracy on the test set
print(SVCModel.score(x_test, Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )


dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(x_train, Y_train)
print(dTree.score(x_train, Y_train))
print(dTree.score(x_test, Y_test))
dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state=1)
#(The importance of a feature is computed as the normalized total reduction of the criterion brought by that feature. 
#It is also known as the Gini importance )

dTreeR.fit(x_train, Y_train)
print(dTreeR.score(x_train, Y_train))
print(dTreeR.score(x_test, Y_test))
# importance of features in the tree building 
print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = x_train.columns))
print(dTreeR.score(x_test , Y_test))
pred = dTreeR.predict(x_test)

ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )

# Ensemble Techniques

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))


ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
bgcl = BaggingClassifier(base_estimator=dTreeR, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))

ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
bgcl = BaggingClassifier(base_estimator=LogRegModel, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(x_train, Y_train)
pred = bgcl.predict(x_test)
print(bgcl.score(x_test , Y_test))

ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
abcl = AdaBoostClassifier(base_estimator=dTree,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
abcl = AdaBoostClassifier(base_estimator=dTreeR,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
abcl = AdaBoostClassifier(base_estimator=LogRegModel,n_estimators=30, random_state=1)
abcl = abcl.fit(x_train, Y_train)
pred = abcl.predict(x_test)
print(abcl.score(x_test , Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )
rfcl = RandomForestClassifier(n_estimators = 50, random_state=1,max_depth=5)
rfcl = rfcl.fit(x_train, Y_train)
pred= rfcl.predict(x_test)
print(rfcl.score(x_test , Y_test))
ConfMat=metrics.confusion_matrix(Y_test, pred, labels=[1, 0])
print (ConfMat)

ConfMat_DF = pd.DataFrame(ConfMat, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,4))
sns.heatmap(ConfMat_DF, annot=True, cmap='GnBu',center=0 )