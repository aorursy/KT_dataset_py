#import libraries
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
%matplotlib inline
#Test Train Split
from sklearn.model_selection import train_test_split
#Feature Scaling library
from sklearn.preprocessing import StandardScaler
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
df_bank = pd.read_csv("../input/bank-personal-loan-modellingcsv/Bank_Personal_Loan_Modelling .csv")
# 1. Read the column description and ensure you understand each attribute well  
df_bank.shape # Check number of columns and rows in data frame
df_bank.dtypes
df_bank.isnull().values.any() # If there are any null values in data set
df_bank.head() # To check first 5 rows of data set
#Using t test to test attributes('Age','Experience','Income','ZIP Code','CCAvg','Mortgage') 
# individually make any difference to customers taking the loan.

cols =['Age','Experience','Income','ZIP Code','CCAvg','Mortgage']
for i in np.arange(len(cols)):
    tx = df_bank[df_bank['Personal Loan'] == 1][cols[i]]  # taken Personal Loan
    ty = df_bank[df_bank['Personal Loan'] == 0][cols[i]]  # not taken Personal Loan
    t_stat, pval = stats.ttest_ind(tx,ty, axis = 0)
    if pval < 0.05:
        print(cols[i], f' individually impacts customer behaviour as the p_value {round(pval,4)} < 0.05')
    else:
        print(cols[i], f' individually does not impact customer behaviour as the p_value {round(pval,4)} > 0.05')
# Chi_square test to check if the categorical/boolean variables individually make any difference to customers taking the loan
cols =['Family','Education','Securities Account','CD Account','Online','CreditCard']
for i in np.arange(len(cols)):
    crosstab=pd.crosstab(df_bank['Personal Loan'],df_bank[cols[i]])
    chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)
    if p_value < 0.05:
         print(cols[i], f' individually impacts customer behaviour as the p_value {round(p_value,4)} < 0.05')
    else:
         print(cols[i], f' individually does not impact customer behaviour as the p_value {round(p_value,4)} > 0.05')  
# 2. Study the data distribution in each attribute, share your findings  (15 marks)

df_bank.describe()
# studying the distribution of continuous attributes
cols = ['Age','Experience','Income','ZIP Code','CCAvg','Mortgage']
for i in np.arange(len(cols)):
    sns.distplot(df_bank[cols[i]], color='blue')
    #plt.xlabel('Experience')
    plt.show()
    print('Distribution of ',cols[i])
    print('Mean is:',df_bank[cols[i]].mean())
    print('Median is:',df_bank[cols[i]].median())
    print('Mode is:',df_bank[cols[i]].mode())
    print('Standard deviation is:',df_bank[cols[i]].std())
    print('Skewness is:',df_bank[cols[i]].skew())
    print('Maximum is:',df_bank[cols[i]].max())
    print('Minimum is:',df_bank[cols[i]].min())
# Distribution of categorical columns 'Family','Education','Securities Account','CD Account','Online'and 
#'CreditCard' individually against 'Personal Loan'
cols =['Family','Education','Securities Account','CD Account','Online','CreditCard']
for i in np.arange(len(cols)):
    sns.countplot(x= df_bank[cols[i]],data=df_bank,hue='Personal Loan')
    plt.show()
    # calculating counts
    print(pd.pivot_table(data=df_bank,index='Personal Loan',columns=[cols[i]],aggfunc='size'))                                                   
   
#plt.figure(figsize = (50,50))
sns.pairplot(df_bank)

# studying correlation between the attributes
b_corr=df_bank.corr()
plt.subplots(figsize =(12, 7)) 
sns.heatmap(b_corr,annot=True)
# Identify the true and false for Personal Loan
loan_true = len(df_bank.loc[df_bank['Personal Loan'] == 1])
loan_false = len(df_bank.loc[df_bank['Personal Loan'] == 0])

print (f"\nCustomer percentage who accepted loan offer: {loan_true} in total {len(df_bank)} which is {loan_true/len(df_bank)*100}%")
print (f"Customer percentage who did not accept loan offer: {loan_false} in total {len(df_bank)} which is  {loan_false/len(df_bank)*100}%")

# Distribution of 'Personal Loan'
sns.countplot(x= 'Personal Loan',data=df_bank)
plt.show()
print('Counts when Personal Loan is:\n',df_bank['Personal Loan'].value_counts())
# Checking the presence of outliers
l = len(df_bank)
col = ['Age','Income','ZIP Code','CCAvg','Mortgage']
for i in np.arange(len(col)):
    sns.boxplot(x= df_bank[col[i]], color='cyan')
    plt.show()
    print('Boxplot of ',col[i])
    #calculating the outiers in attribute 
    Q1 = df_bank[col[i]].quantile(0.25)
    Q2 = df_bank[col[i]].quantile(0.50)
    Q3 = df_bank[col[i]].quantile(0.75)
    print('Q1 is : ',Q1)
    print('Q2 is : ',Q2)
    print('Q3 is : ',Q3)
    IQR = Q3 - Q1
    print('IQR is:',IQR)
    bools = (df_bank[col[i]] < (Q1 - 1.5 *IQR)) |(df_bank[col[i]] > (Q3 + 1.5 * IQR))
    print('Out of ',l,' rows in data, number of outliers are:',bools.sum())   #calculating the number of outliers
    n_zeros = len(df_bank.loc[df_bank[col[i]] == 0])
    print(' Number of zeros in',col[i],'is ', n_zeros)
    lw_tc = len(df_bank.loc[df_bank[col[i]] < (Q1 - 1.5 *IQR)])  # Total number less than Lower Whisker
    print(' Total number less than Lower Whisker',col[i],'is ', lw_tc)
    uw_tc = len(df_bank.loc[df_bank[col[i]] > (Q3 + 1.5 * IQR)]) # Total number more than Upper Whisker
    print(' Total number more than Upper Whisker',col[i],'is ', uw_tc)
    total_c = l - (n_zeros+lw_tc+uw_tc)                         # Total non zero data within IQR is
    print(' Total non zero data within IQR is :',total_c )
    print(' Number of records in Outlier where Personal loan was taken', len(df_bank.loc[(df_bank[col[i]] > (Q3 + 1.5 * IQR)) & (df_bank['Personal Loan'] == 1)]))
#Dropping column 'ID' and 'Experience'
# ID is just an identifier for the row and has no contrbution to personal loan, so we drop it.
df_bank.drop('ID',axis=1,inplace=True) 

# Experience is in high correlation with 'Age', so the contribution of both these columns towards personal loan 
# would be same, hence dropping 'Experience'. Since, we are dropping 'Experience' no need to correct it for negative values.
df_bank.drop('Experience',axis=1,inplace=True)
# shows 'Age' and 'Experience' have been dropped
df_bank.dtypes
col = ['Income','ZIP Code','CCAvg','Mortgage']
for i in np.arange(len(col)):
    df_bank['zscore']= np.abs(stats.zscore(df_bank[col[i]]))
    df_bank= df_bank[df_bank['zscore']<= 3]
    df_bank.drop('zscore',axis=1,inplace=True) #Rows where col[i] was in outliers have been dropped 
X_bank = df_bank.drop(['Personal Loan'], axis=1)
X = df_bank.drop(['Personal Loan'], axis=1)
y = df_bank['Personal Loan']
# 4. Split the data into training and test set in the ratio of 70:30 respectively 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#since we have just 480 records in 'Personal Loan' column, verifying to ensure there are some records with 'Personal Loan' = 1 for training and testing both.
print('Number of records in y_train with values 0 & 1 are:\n',y_train.value_counts())
print('Number of records in y_test with values 0 & 1 are:\n',y_test.value_counts())
# Fit model on the Train-set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict Test-set
#y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Coefficient and intercept of model
coef = pd.DataFrame(logreg.coef_)
coef['intercept'] = logreg.intercept_
print('\n\nCoefficient :',coef)

# Score of model
model_score = logreg.score(X_test,y_test)
print('\nScore of the model '+str(model_score))

print('\nLogistic Regression Classification report:\n',classification_report(y_test, logreg.predict(X_test)))

LGRcm_matrix = metrics.confusion_matrix(y_test,logreg.predict(X_test))

print('\nConfusion metrics :\n', metrics.confusion_matrix(y_test,logreg.predict(X_test)))
#true positives (TP): These are cases in which we predicted yes, and actually took loan.
TP=58
#true negatives (TN): We predicted no, and they actually did not took loan.
TN=1289
#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")
FP=22
#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")
FN=64


Accuracy=(TP+TN)/(TP+TN+FP+FN)
print('Accuracy of logistic regression classifier on test set: {:.2%}'.format(Accuracy))

Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)
print('Logistic regression Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))

#Recall
Sensitivity=TP/(FN+TP)
print('Logistic regression Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))

Specificity=TN/(TN+FP)
print('Logistic regression Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))

Precision=TP/(FP+TP)
print('Logistic regression Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))

#Area Under the ROC Curv
print('Logistic regression AUC: ',round(roc_auc_score(y_test,logreg.predict(X_test))*100))


#ROC Curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#----------------------------------------------------------------------------------------------------------------
# For better understanding of confusion matrix when 3 nearest Neighbours we plot it on heatmap

LGRcm_matrix = metrics.confusion_matrix(y_test,logreg.predict(X_test))

report_LGR = classification_report(y_test,logreg.predict(X_test),labels=[1,0])

#Converting all attributes to z-score
XScaled = X.apply(zscore)

# 2) Spliting randomly in train and test set
X_train1,X_test1,y_train1,y_test1 = train_test_split(XScaled, y, test_size=0.30, random_state=1)
# Call Nearest Neighbour algorithm, keeping number of neighbours as 3
NNH = KNeighborsClassifier(n_neighbors= 3 , weights = 'uniform' )
NNH.fit(X_train1, y_train1)

# Score of the Model
print('NNH score when 3 nearest neighbours:', NNH.score(X_test1, y_test1))


# For every test data point, predict it's label based on 3 nearest neighbours in this model. The majority class will 
# be assigned to the test data point

print('\nConfusion metrics when 3 nearest neighbour:\n', metrics.confusion_matrix(y_test1,NNH.predict(X_test1)))

print('\nAUC: ',roc_auc_score(y_test1, NNH.predict(X_test1)*100))

#-------------------------------------------------------------------------------------------------------------------
#Iteration 1:
# Call Nearest Neighbour algorithm, keeping number of neighbours as 5
NNH1 = KNeighborsClassifier(n_neighbors= 5 , weights = 'uniform' )
NNH1.fit(X_train1, y_train1)

# Score of the Model
print('\nNNH score when 5 nearest neighbours :', NNH1.score(X_test1, y_test1))

# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 
# be assigned to the test data point

print('\nConfusion metrics when 5 nearest neighbour:\n',metrics.confusion_matrix(y_test1,NNH1.predict(X_test1)))

print('\nAUC: ',roc_auc_score(y_test1, NNH1.predict(X_test1)*100))

#-----------------------------------------------------------------------------------------------------------------

#Iteration 2:

# Call Nearest Neighbour algorithm, keeping number of neighbours as 9
NNH2 = KNeighborsClassifier(n_neighbors= 9, weights = 'uniform' )
NNH2.fit(X_train1, y_train1)

print('\nNNH score when 9 nearest neighbours :', NNH2.score(X_test1, y_test1))

print('\nConfusion metrics when 9 nearest neighbour:\n',metrics.confusion_matrix(y_test1,NNH2.predict(X_test1)))

print('\nAUC: ',roc_auc_score(y_test1, NNH2.predict(X_test1)*100))

#----------------------------------------------------------------------------------------------
#Classification reports

print('\nClassification report for 3 nearest neighbour\n',classification_report(y_test1,NNH.predict(X_test1)))
print('\nClassification report for 5 nearest neighbour\n',classification_report(y_test1,NNH2.predict(X_test1)))
print('\nClassification report for 9 nearest neighbour\n',classification_report(y_test1,NNH2.predict(X_test1)))

report_KNN = classification_report(y_test1,NNH.predict(X_test1))
#Since AUC and NNH score is more for 3 nearest neighbour calculating metrices

#Confusion metrics when 3 nearest neighbour:
# [[1307    3]
# [  44   80]]

#true positives (TP): These are cases in which we predicted yes, and actually took loan.
TP=80
#true negatives (TN): We predicted no, and they actually did not took loan.
TN=1307
#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")
FP=3
#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")
FN=44

Accuracy=(TP+TN)/(TP+TN+FP+FN)
print('Accuracy of KNN classifier on test set: {:.2%}'.format(Accuracy))

Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)
print('KNN Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))

#Recall
Sensitivity=TP/(FN+TP)
print('KNN Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))

Specificity=TN/(TN+FP)
print('KNN Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))

Precision=TP/(FP+TP)
print('KNN Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))

print('KNN AUC: ',roc_auc_score(y_test1, NNH.predict(X_test1)*100))

#ROC Curve

KNN_roc_auc = roc_auc_score(y_test1, NNH.predict(X_test1))
fpr, tpr, thresholds = roc_curve(y_test1, NNH.predict_proba(X_test1)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % KNN_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()

#----------------------------------------------------------------------------------------------------------------
# For better understanding of confusion matrix when 3 nearest Neighbours we plot it on heatmap

KNNcm_matrix = metrics.confusion_matrix(y_test1,NNH.predict(X_test1))

report_KNN = classification_report(y_test1,NNH.predict(X_test1),labels=[1,0])
#Iteration 1 - Fitting all variables, cleaned and normalized data
GNB1 = GaussianNB()
GNB1.fit(X_train, y_train)

# Score of the Model
print('GNB score :', GNB1.score(X_test, y_test))


# For every test data point, predict it's label based on 3 nearest neighbours in this model. The majority class will 
# be assigned to the test data point

print('\nConfusion metrics :\n', metrics.confusion_matrix(y_test, GNB1.predict(X_test)))

print('\nAUC: ',roc_auc_score(y_test, GNB1.predict(X_test)*100))
#CALCULATING METRICES FOR CHECING MODEL

#Confusion metrics :
# [[1224   87]
# [  50   73]]

#true positives (TP): These are cases in which we predicted yes, and actually took loan.
TP=73
#true negatives (TN): We predicted no, and they actually did not took loan.
TN=1224
#false positives (FP): We predicted yes, but they don't actually took loan.(Also known as a "Type I error.")
FP=87
#false negatives (FN): We predicted no, but they actually took loan.(Also known as a "Type II error.")
FN=50


Accuracy=(TP+TN)/(TP+TN+FP+FN)
print('Accuracy of GNB classifier on test set: {:.2%}'.format(Accuracy))

Misclassification_Rate=(FP+FN)/(TP+TN+FP+FN)
print('GNB Misclassification Rate: It is often wrong: {:.2%}'.format(Misclassification_Rate))

#Recall
Sensitivity=TP/(FN+TP)
print('GNB Sensitivity: When its actually yes how often it predicts yes: {:.2%}'.format(Sensitivity))

Specificity=TN/(TN+FP)
print('GNB Specificity: When its actually no, how often does it predict no: {:.2%}'.format(Specificity))

Precision=TP/(FP+TP)
print('GNB Precision: When it predicts yes, how often is it correct: {:.2%}'.format(Precision))

print('GNB AUC: ',roc_auc_score(y_test, GNB1.predict(X_test)*100))
#ROC Curve
GNB_roc_auc = roc_auc_score(y_test, GNB1.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, GNB1.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='GNB (area = %0.2f)' % GNB_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()
#-------------------------------------------------------------------------------------------------------------

GNBcm_matrix = metrics.confusion_matrix(y_test, GNB1.predict(X_test))

report_GNB = classification_report(y_test,GNB1.predict(X_test),labels=[1,0])
 
# For better understanding of logistic regression confusion matrix we plot it on heatmap

HM = pd.DataFrame(LGRcm_matrix, index = [i for i in ['0','1']],
                    columns = [i for i in ['Predict 0', 'Predict 1']])
plt.figure(figsize=(7,5))
print('Confusion matrix for logistic regression ')
sns.heatmap(HM,annot=True, fmt='g')
plt.show()

# For better understanding of KNN confusion matrix we plot it on heatmap
HM = pd.DataFrame(KNNcm_matrix, index = [i for i in ['0','1']],
                    columns = [i for i in ['Predict 0', 'Predict 1']])
plt.figure(figsize=(7,5))
print('KNN confusion matrix\n')
sns.heatmap(HM,annot=True, fmt='g') 
plt.show()

# For better understanding of GNB confusion matrix we plot it on heatmap
HM = pd.DataFrame(GNBcm_matrix, index = [i for i in ['0','1']],
                    columns = [i for i in ['Predict 0', 'Predict 1']])
plt.figure(figsize=(7,5))
print('GNB confusion matrix\n')
sns.heatmap(HM,annot=True, fmt='g') 
plt.show()
print("\nLogisitic Regression\n",report_LGR)
print("\nNaive Bayes\n",report_GNB)
print("\nKNN Classifier\n",report_KNN)