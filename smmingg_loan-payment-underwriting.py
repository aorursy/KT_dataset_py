import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
loan = pd.read_csv('../input/loan-approval/loan.csv')
payment = pd.read_csv('../input/payment/payment.csv')
underwriting = pd.read_csv('../input/underwriting/clarity_underwriting_variables.csv')
loan.head()
loan.info()
loan['loanStatus'].value_counts()
plt.figure(figsize= (15,8))
plt.xlabel('Loan Status')
plt.ylabel('Count')
loan['loanStatus'].value_counts().plot(kind = 'barh', grid = True)
plt.show()
withdraw = len(loan[(loan.loanStatus == 'Withdrawn Application')])
print ('Withdrawn Application Ratio: %.2f%%'  % (withdraw/len(loan)*100))
loan_amount = loan["loanAmount"].values
loan_amount
plt.figure(figsize=(12,6))
sns.violinplot(loan["loanAmount"], inner="point", palette="bone")
plt.figure(figsize=(12,10))
sns.barplot(x='loanAmount', y='state', data=loan)
plt.title('Issuance of Loans', fontsize=16)
plt.xlabel('Average loan amount issued', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.figure(figsize=(12,10))
sns.barplot(x= loan['apr'], y= loan['state'], palette = 'Blues_d')
plt.title('APR in different State', fontsize=16)
plt.xlabel('APR', fontsize=14)
plt.ylabel('State', fontsize=14)
loan.corr()
sns.scatterplot(x= loan['apr'], y= loan['loanAmount'])
payment.head()
payment['paymentStatus'].value_counts()
good_loan = ['Checked','None','Complete','Pending']
#Condition to classify the good and bad loans
payment['loan_condition'] = np.nan

def loan_condition(status):
    if status in good_loan:
        return 'Good Loan'
    else:
        return 'Bad Loan'
    
payment['loan_condition'] = payment['paymentStatus'].apply(loan_condition)
plt.figure(figsize=(8,8))
payment["loan_condition"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', colors = ["#3791D7", "#D72626"], shadow=True, labels=("Good Loans", "Bad Loans"), fontsize=14, startangle=30)
plt.ylabel('% of Condition of Loans', fontsize=14)
underwriting.info()
underwriting_corr = underwriting.corr()
underwriting_corr
plt.figure(figsize=(12,10))
sns.heatmap(underwriting_corr, annot=True, cmap="YlGnBu")
loan_state = pd.get_dummies(loan['state'])
loan_new = pd.concat([loan, loan_state], axis=1)
loan_new
merge = pd.merge(loan_new, underwriting,left_on='clarityFraudId', right_on='underwritingid')
merge
merge.drop(['loanId','anon_ssn','payFrequency','clarityFraudId','apr','state','applicationDate','originated','originatedDate','isFunded','loanStatus','originallyScheduledPaymentAmount','leadType','leadCost','fpStatus','hasCF'], axis=1, inplace=True)
merge.drop(['.underwritingdataclarity.clearfraud.clearfraudindicator.bestonfilessnissuedatecannotbeverified',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.bestonfilessnrecordedasdeceased',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.creditestablishedbeforeage18',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.creditestablishedpriortossnissuedate',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.currentaddressreportedbynewtradeonly',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.currentaddressreportedbytradeopenlt90days',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.driverlicenseformatinvalid',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.driverlicenseinconsistentwithonfile',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inputssnissuedatecannotbeverified',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inputssnrecordedasdeceased',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inputssninvalid',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquiryaddresscautious',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquiryaddresshighrisk',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquiryaddressnonresidential',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquiryageyoungerthanssnissuedate',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquirycurrentaddressnotonfile',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.morethan3inquiriesinthelast30days',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.inquiryonfilecurrentaddressconflict',           
            '.underwritingdataclarity.clearfraud.clearfraudindicator.onfileaddresscautious',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.onfileaddresshighrisk',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.onfileaddressnonresidential',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.telephonenumberinconsistentwithaddress',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.telephonenumberinconsistentwithstate',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.highprobabilityssnbelongstoanother',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.ssnreportedmorefrequentlyforanother',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.workphonepreviouslylistedashomephone',
            '.underwritingdataclarity.clearfraud.clearfraudindicator.workphonepreviouslylistedascellphone',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressmatch',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncodedescription',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.overallmatchresult',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssndobreasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.phonematchtypedescription',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamematch',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncodedescription',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.phonematchresult',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.phonematchtype',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.phonetype',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssndobmatch',
            'underwritingid'], axis=1, inplace=True)
merge = merge[pd.notnull(merge['clearfraudscore'])]
merge
merge.sort_values(['clearfraudscore'],ascending = False)
merge.info()
merge.nPaidOff = merge.nPaidOff.fillna(merge.nPaidOff.median())
merge.loanAmount = merge.loanAmount.fillna(merge.loanAmount.median())
merge.info()
from sklearn.model_selection import train_test_split
y = merge['approved']
X = merge.loc[:, merge.columns != 'approved']
from sklearn import metrics
from sklearn.model_selection import train_test_split

y = merge['approved']
X = merge.loc[:, merge.columns != 'approved']

X_trainSVM, X_testSVM, y_trainSVM, y_testSVM = train_test_split (X, y, test_size=0.2, random_state=5)
print ('Train set:', X_trainSVM.shape,  y_trainSVM.shape)
print ('Test set:', X_testSVM.shape,  y_testSVM.shape)
from sklearn import svm
clf = svm.SVC(kernel='rbf', max_iter = -1)
clf.fit(X_trainSVM, y_trainSVM) 
yhatSVM = clf.predict(X_testSVM)
yhatSVM[0:10]
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_trainLR, X_testLR, y_trainLR , y_testLR = train_test_split(X, y, test_size=0.2, random_state=5)
logreg = LogisticRegression()
logreg.fit(X_trainLR, y_trainLR)
yhatLR = logreg.predict(X_testLR)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_testLR, y_testLR)))
LR_yhat_prob = logreg.predict_proba(X_testLR)
LR_yhat_prob
from sklearn.tree import DecisionTreeClassifier
X_trainDT, X_testDT, y_trainDT, y_testDT = train_test_split(X, y, test_size=0.2, random_state=5)
print ('Train set:', X_trainDT.shape,  y_trainDT.shape)
print ('Test set:', X_testDT.shape,  y_testDT.shape)
mergeDT = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
mergeDT # it shows the default parameters
mergeDT.fit(X_trainDT,y_trainDT)
yhatDT = mergeDT.predict(X_testDT)
yhatDT
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testDT, yhatDT))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 7
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
kNN_model
yhat = kNN_model.predict(X_test)
yhat[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(y_train, kNN_model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
#Trying different K
Ks=15
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
ConfustionMx=[];
for n in range(1,Ks):
    
    #Train Model and Predict  
    kNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = kNN_model.predict(X_test)
    
    
    mean_acc[n-1]=np.mean(yhat==y_test);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    
mean_acc
k = 13
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat = kNN_model.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, kNN_model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
from sklearn.metrics import jaccard_similarity_score, f1_score,log_loss
print("SVM F1-score: %.2f" % f1_score(y_testSVM, yhatSVM, average='weighted'))
print("SVM Jaccard index: %.2f" % jaccard_similarity_score(y_testSVM, yhatSVM))
print("LR F1-score: %.2f" % f1_score(y_testLR, yhatLR, average='weighted'))
print("LR Jaccard index: %.2f" % jaccard_similarity_score(y_testLR, yhatLR))
print("LR LogLoss: %.2f" % log_loss(y_testLR, LR_yhat_prob))
print("DT F1-score: %.2f" % f1_score(y_testDT, yhatDT, average='weighted'))
print("DT Jaccard index: %.2f" %jaccard_similarity_score(y_testDT, yhatDT))
print("KNN F1-score: %.2f" % f1_score(y_test, yhat, average='weighted'))
print("KNN Jaccard index: %.2f" % jaccard_similarity_score(y_test, yhat))