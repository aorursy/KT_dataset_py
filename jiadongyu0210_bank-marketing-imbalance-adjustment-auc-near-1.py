#library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection  #for k-fold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
bank = pd.read_csv('../input/bank-marketing/bank-additional-full.csv',sep=';')
bank.head()
bank.info()
#missing value check
bank.isnull().any()
bank.columns
#find wether the data set is imlalance or not
bank.y.value_counts(1).reset_index().rename(columns={'index':'Category','y':'Percentage'}).set_index('Category')
#numerical value outlier check
bank.describe()
bank_client=bank.iloc[:,0:7]
bank_client.head()
fig, ax = plt.subplots()
fig.set_size_inches(15,8)
sns.countplot(x='age',data=bank_client)
ax.set_xlabel('Age', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Age Count Distribution', fontsize=10)
sns.despine()
#Observe the max, min, 1,2,3,4 quartile and std.
bank_client['age'].describe()
#Box and Histgram
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 7))

sns.boxplot(x = 'age', data = bank_client, orient = 'v', ax = ax1)
ax1.set_ylabel('Age', fontsize=10)
ax1.set_title('Age Distribution Box Plot', fontsize=10)
ax1.tick_params(labelsize=15)

sns.distplot(bank_client['age'],ax = ax2)
ax2.set_xlabel('Age', fontsize=10)
ax2.set_title('Age Distribution Histgram', fontsize=10)
ax2.tick_params(labelsize=10)
sns.despine()
# Calcualte Outliers:
# Interquartile range, IQR = Q3 - Q1
# lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
# Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
print('Ages above: ', bank_client['age'].quantile(q = 0.75) + 1.5*(bank_client['age'].quantile(q = 0.75) - bank_client['age'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', bank_client[bank_client['age'] > 69.6]['age'].count())
print('Number of clients: ', len(bank_client))
#Outliers in %
print('Outliers are:', round(bank_client[bank_client['age'] > 69.6]['age'].count()*100/len(bank_client),2), '%')
# Calculating some values to evaluete this independent variable
print('MEAN:', round(bank_client['age'].mean(), 1))
# A low standard deviation indicates that the data points tend to be close to the mean or expected value
# A high standard deviation indicates that the data points are scattered
print('STD :', round(bank_client['age'].std(), 1))
# I thing the best way to give a precisly insight abou dispersion is using the CV (coefficient variation) (STD/MEAN)*100
#    cv < 15%, low dispersion
#    cv > 30%, high dispersion
print('CV  :',round(bank_client['age'].std()*100/bank_client['age'].mean(), 1), ', High middle dispersion')
# See jobs category:
print('Jobs:', bank_client['job'].unique())
#Job distribution with sorted count values
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
sns.countplot(x = 'job', data = bank_client,order = bank_client['job'].value_counts().index)
ax.set_xlabel('Job', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Job Distribution', fontsize=10)
sns.despine()
print('Marital:', bank_client['marital'].unique())
#Marital distribution with sorted count values
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
sns.countplot(x = 'marital', data = bank_client,order = bank_client['marital'].value_counts().index)
ax.set_xlabel('Marital', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Marital Distribution', fontsize=10)
sns.despine()
print('Education:', bank_client['education'].unique())
#Marital distribution with sorted count values
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
sns.countplot(x = 'education', data = bank_client,order = bank_client['education'].value_counts().index)
ax.set_xlabel('Education', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Education Distribution', fontsize=10)
sns.despine()
print('Default:\n', bank_client['default'].unique())
print('Housing:\n', bank_client['housing'].unique())
print('Loan:\n', bank_client['loan'].unique())
# Default, has credit in default ?
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))
sns.countplot(x = 'default', data = bank_client, ax = ax1, order = ['no', 'unknown', 'yes'])
ax1.set_title('Default Distribution', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.tick_params(labelsize=10)

# Housing, has housing loan ?
sns.countplot(x = 'housing', data = bank_client, ax = ax2, order = ['no', 'unknown', 'yes'])
ax2.set_title('Housing Distribution', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.tick_params(labelsize=10)

# Loan, has personal loan ?
sns.countplot(x = 'loan', data = bank_client, ax = ax3, order = ['no', 'unknown', 'yes'])
ax3.set_title('Loan Distribution', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.25)

print('Default:\n No credit in default:', bank_client[bank_client['default'] == 'no']['age'].count(), 
      '\n Yes to credit in default:' , bank_client[bank_client['default'] == 'yes']['age'].count(),
              '\n Unknown credit in default:', bank_client[bank_client['default'] == 'unknown']['age'].count()
             )

print('Housing:\n No housing in loan:', bank_client[bank_client['housing'] == 'no']['age'].count(),
      '\n Yes to housing in loan:' , bank_client[bank_client['housing'] == 'yes']['age'].count(),
              '\n Unknown housing in loan:', bank_client[bank_client['housing'] == 'unknown']['age'].count()
              )

print('Loan:\n No to personal loan:', bank_client[bank_client['loan'] == 'no']['age'].count(),
      '\n Yes to personal loan:'    , bank_client[bank_client['loan'] == 'yes']['age'].count(),
              '\n Unknown to personal loan:', bank_client[bank_client['loan'] == 'unknown']['age'].count()
              )
labelencoder_X = LabelEncoder()
# Label encoder order is alphabetical
#Slice the age segementation with value to represent different age groups.
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe

age(bank_client)
bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 
bank_client.head()
bank_related = bank.iloc[: , 7:11]
bank_related.head()
bank_related.duration.describe()
#Box and Histgram
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 7))

sns.boxplot(x = 'duration', data = bank_related, orient = 'v', ax = ax1)
ax1.set_ylabel('Duration', fontsize=10)
ax1.set_title('Duration Box Plot', fontsize=10)
ax1.tick_params(labelsize=15)

sns.distplot(bank_related['duration'], ax = ax2)
ax2.set_xlabel('Duration', fontsize=10)
ax2.set_title('Duration Distribution Histgram', fontsize=10)
ax2.tick_params(labelsize=10)
sns.despine()
print('Duration calls above: ', bank_related['duration'].quantile(q = 0.75) + 
                      1.5*(bank_related['duration'].quantile(q = 0.75) - bank_related['duration'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', bank_related[bank_related['duration'] > 644.5]['duration'].count())
print('Number of clients: ', len(bank_related))
#Outliers in %
print('Outliers are:', round(bank_related[bank_related['duration'] > 644.5]['duration'].count()*100/len(bank_related),2), '%')
# Look, if the call duration is equal to 0 THIS LINES NEED TO BE DELETED LATER 
bank_related[(bank_related['duration'] == 0)]
print("Kind of Contact: \n", bank_related['contact'].unique())
print("\nKind of Months: \n", bank_related['month'].unique())
print("\nKind of Days: \n", bank_related['day_of_week'].unique())
# Contact
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))
sns.countplot(x = 'contact', data = bank_related, ax = ax1, order = bank_related['contact'].value_counts().index)
ax1.set_title('Contact Distribution', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.tick_params(labelsize=10)

# Months
sns.countplot(x = 'month', data = bank_related, ax = ax2, order= ['mar','apr','may','jun','jul','aug' ,'sep','oct','nov','dec'])
ax2.set_title('Month Distribution', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.tick_params(labelsize=10)

# Days
sns.countplot(x = 'day_of_week', data = bank_related, ax = ax3)
ax3.set_title('Days Distribution', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.25)

#Convert into continous number
bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 
#Slicing the duration
def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data
duration(bank_related);
bank_related.head()
bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
bank_o.head()
print("campaign: \n", bank_o['campaign'].unique())
print("\npdays: \n", bank_o['pdays'].unique())
print("\nprevious: \n", bank_o['previous'].unique())
print("\npoutcome: \n", bank_o['poutcome'].unique())
bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
bank_o.head()
bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
bank_se.head()
print("emp.var.rate: \n", bank_se['emp.var.rate'].unique())
print("\ncons.price.idx: \n", bank_se['cons.price.idx'].unique())
print("\ncons.conf.idx: \n", bank_se['cons.conf.idx'].unique())
print("\neuribor3m: \n", bank_se['euribor3m'].unique())
print("\nnr.employed: \n", bank_se['nr.employed'].unique())
bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
bank_final.shape
bank_final_y=pd.concat([bank_final,bank.y],axis=1)

bank_final_y['y'].replace(['yes', 'no'], [1,0], inplace  = True)

bank_final_y.head()
bank_final_y.corr()[['y']].sort_values('y',ascending=False)
plt.figure(figsize=(8,6))
sns.heatmap(bank_final_y[bank_final_y.corr()[['y']].sort_values('y',ascending=True).index].corr(),cmap="Greys")
plt.tick_params(labelsize=16)
y=bank_final_y.y
X=bank_final
X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.2, random_state = 101)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# Logistic Regression
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
logmodel = LogisticRegression() 
logmodel.fit(X_train,y_train)
logpred = logmodel.predict(X_test)

print("Logistic Regression:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, logpred))
print("\nAccuracy:")
print(round(accuracy_score(y_test, logpred),2)*100)
print("\nCross Valuation Score:")
LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(LOGCV)
# K nearest neighbour
#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=123)
    scores = model_selection.cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)
print("KNN:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, knnpred))
print("\nAccuracy:")
print(round(accuracy_score(y_test, knnpred),2)*100)
print("\nCross Valuation Score:")
KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(KNNCV)
# Decision Tree
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)
print("Decision Tree:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, dtreepred))
print("\nAccuracy:")
print(round(accuracy_score(y_test, dtreepred),2)*100)
print("\nCross Valuation Score:")
DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(DTREECV)
#SVM
svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, y_train)
svcpred = svc.predict(X_test)
print("SVM:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, svcpred))
print("\nAccuracy:")
print(round(accuracy_score(y_test, svcpred),2)*100)
print("\nCross Valuation Score:")
SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(SVCCV)
#Random Forest
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)
print("Random Forest:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, rfcpred ))
print("\nAccuracy:")
print(round(accuracy_score(y_test, rfcpred),2)*100)
print("\nCross Valuation Score:")
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(RFCCV)
#XGB
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)
print("XGB:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, xgbprd ))
print("\nAccuracy:")
print(round(accuracy_score(y_test, xgbprd),2)*100)
print("\nCross Valuation Score:")
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
print(XGB)
#GrandientBoosting
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)
print("GrandientBoosting:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, gbkpred ))
print("\nAccuracy:")
print(round(accuracy_score(y_test, gbkpred),2)*100)
print("\nCross Valuation Score:")
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(GBKCV)
models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model',  'XGBoost', 'Gradient Boosting'],
                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, XGB, GBKCV]})

models.sort_values(by='Score', ascending=False)
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 4))
fig, ax_arr = plt.subplots(nrows = 2, ncols = 4, figsize = (20,16))


#LOGMODEL
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('ROC Logistic ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#RANDOM FOREST --------------------
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('ROC Random Forest ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#KNN----------------------
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('ROC KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#DECISION TREE ---------------------
probs = dtree.predict_proba(X_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[0,3].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[0,3].plot([0, 1], [0, 1],'r--')
ax_arr[0,3].set_title('ROC Decision Tree ',fontsize=20)
ax_arr[0,3].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,3].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,3].legend(loc = 'lower right', prop={'size': 16})

#XGB-------------------------------------------
probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax_arr[1,0].plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('ROC XGBOOST ',fontsize=10)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})

#Gradient-------------------------------------
probs = gbk.predict_proba(X_test)
preds = probs[:,1]
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)
roc_aucgbk = metrics.auc(fprgbk, tprgbk)

ax_arr[1,1].plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('ROC GRADIENT BOOST ',fontsize=10)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})


#ALL PLOTS TOGETHER---------------------------

ax_arr[1,3].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='blue')
ax_arr[1,3].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,3].plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
ax_arr[1,3].plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
ax_arr[1,3].plot(fprxgb, tprxgb, 'b', label = 'XGBOOST', color='red')
ax_arr[1,3].plot(fprgbk, tprgbk, 'b', label = 'GRADIENT BOOST', color='yellow')

ax_arr[1,3].set_title('ROC ',fontsize=20)
ax_arr[1,3].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,3].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,3].legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=0.2)
plt.tight_layout() 
#Logistic Regress
print("\nLogisticRegression:")
print(classification_report(y_test,logpred,target_names=['No=0','Yes=1']))
#KNN
print("\nKNN:")
print(classification_report(y_test,knnpred,target_names=['No=0','Yes=1']))
#SVM
print("\nSVM:")
print(classification_report(y_test,svcpred,target_names=['No=0','Yes=1']))
#DecisionTree
print("\nDecisionTree:")
print(classification_report(y_test,dtreepred,target_names=['No=0','Yes=1']))

#RandomForest
print("\nRandomForest:")
print(classification_report(y_test,rfcpred,target_names=['No=0','Yes=1']))

#XGB
print("\nXGB:")
print(classification_report(y_test,xgbprd,target_names=['No=0','Yes=1']))

#GradientBoosting
print("\nGradientBoosting:")
print(classification_report(y_test,gbkpred,target_names=['No=0','Yes=1']))
#Near Miss
nml=NearMiss(version=2) 
#Select the sample with the shortest average distance from the N furthest samples among the positive samples
X_nml,y_nml=nml.fit_sample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_nml, y_nml, test_size = 0.2, random_state = 101)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#XGB
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)
print("XGB:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, xgbprd ))
print("\nAccuracy:")
print(round(accuracy_score(y_test, xgbprd),2)*100)
print("\nCross Valuation Score:")
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
print(XGB)

fig, ax = plt.subplots()
#XGB-------------------------------------------
probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('ROC XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.legend(loc = 'lower right', prop={'size': 16})

#XGB
print("\nXGB:")
print(classification_report(y_test,xgbprd,target_names=['No=0','Yes=1']))
#SMOTEENN
smote_enn=SMOTEENN(random_state=100) 
X_smote_enn,y_smote_enn=smote_enn.fit_sample(X,y)

X_train, X_test, y_train, y_test = train_test_split(X_smote_enn, y_smote_enn, test_size = 0.2, random_state = 101)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#XGB
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)
print("XGB:\n")
print("Confusion matrix:")
print(confusion_matrix(y_test, xgbprd ))
print("\nAccuracy:")
print(round(accuracy_score(y_test, xgbprd),2)*100)
print("\nCross Valuation Score:")
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
print(XGB)

fig, ax = plt.subplots()
#XGB-------------------------------------------
probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('ROC XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.legend(loc = 'lower right', prop={'size': 16})

#XGB
print("\nXGB:")
print(classification_report(y_test,xgbprd,target_names=['No=0','Yes=1']))