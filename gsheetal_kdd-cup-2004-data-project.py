#Import Libraries
import random
import numpy as np
import pandas as pd
import seaborn as ss
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
#Loading Data
Kdd_Cup_Data = pd.read_csv("..../PHY_TRAIN.csv",index_col=0)
#Viewing Data
Kdd_Cup_Data
#Number of Rows and Columns
print ('The KDD Cup 2004 dataset has {0} rows and {1} columns'.format(Kdd_Cup_Data.shape[0],Kdd_Cup_Data.shape[1]))
#Number of Numeric and Categorical Columns
print ("There are {} numeric and {} categorical columns in the KDD Cup 2004 dataset".format(Kdd_Cup_Data.select_dtypes(include=[np.number]).shape[1],Kdd_Cup_Data.select_dtypes(exclude=[np.number]).shape[1]))
#Information of Data
Kdd_Cup_Data.info()
#Summary of Data
display(Kdd_Cup_Data.describe().transpose())
#(Kdd_Cup_Data.describe().transpose()).to_csv("...../Summary_Data.csv")
#Skewness of Data
Kdd_Cup_Data.skew(axis = 0, skipna = True)
#Kdd_Cup_Data.skew(axis = 0, skipna = True).to_csv("..../Skewness.csv")
#Create numeric plots
num_pic = [f for f in Kdd_Cup_Data.columns if Kdd_Cup_Data.dtypes[f] != 'object']
num = pd.melt(Kdd_Cup_Data, value_vars = num_pic)
n_p = ss.FacetGrid (num, col='variable', col_wrap=10, sharex=False, sharey = False)
n_p.map(ss.distplot, 'value');

#As you can see, most of the variables are not skewed. We'll have to transform them in the next stage.
#Missing value indicators for all variables with missing values as 1 else 0
Missing_Indi = Kdd_Cup_Data.isnull().astype(int).add_suffix('_indicator')
#Missing_Indi.to_csv("..../Missing_Value_Indicator.csv")

#Column names with missing values
Missing_Indi.columns[(Missing_Indi == 1).any()]
#Handling Null Values
Nul_C = pd.DataFrame(Kdd_Cup_Data.isnull().sum().sort_values(ascending=False)[:10])
Nul_P = pd.DataFrame(round(Kdd_Cup_Data.isnull().sum().sort_values(ascending = False)/len(Kdd_Cup_Data)*100,2)[round(Kdd_Cup_Data.isnull().sum().sort_values(ascending = False)/len(Kdd_Cup_Data)*100,2) != 0])
Nul_data = pd.concat([Nul_C,Nul_P],axis=1, sort=False)
Nul_data.columns = ['Null Count','Null Percent']
Nul_data.index.name = 'Feature'
Nul_data
#Nul_data.to_csv("..../Null_Data_Values.csv")
#Replacing missing data with the median values
Kdd_Cup_Data.feat20=Kdd_Cup_Data.feat20.fillna(Kdd_Cup_Data.feat20.median())
Kdd_Cup_Data.feat21=Kdd_Cup_Data.feat21.fillna(Kdd_Cup_Data.feat21.median())
Kdd_Cup_Data.feat22=Kdd_Cup_Data.feat22.fillna(Kdd_Cup_Data.feat22.median())
Kdd_Cup_Data.feat29=Kdd_Cup_Data.feat29.fillna(Kdd_Cup_Data.feat29.median())
Kdd_Cup_Data.feat44=Kdd_Cup_Data.feat44.fillna(Kdd_Cup_Data.feat44.median())
Kdd_Cup_Data.feat45=Kdd_Cup_Data.feat45.fillna(Kdd_Cup_Data.feat45.median())
Kdd_Cup_Data.feat46=Kdd_Cup_Data.feat46.fillna(Kdd_Cup_Data.feat46.median())
Kdd_Cup_Data.feat55=Kdd_Cup_Data.feat55.fillna(Kdd_Cup_Data.feat55.median())
#Check remaining missing values if any 
whole_na = (Kdd_Cup_Data.isnull().sum() / len(Kdd_Cup_Data)) * 100
whole_na = whole_na.drop(whole_na[whole_na == 0].index).sort_values(ascending=False)
missing_part = pd.DataFrame({'Missing Ratio' :whole_na})
missing_part.head()
#Correlation score to explain relationship between dependent variable and independent variable
cor = Kdd_Cup_Data.corr()
print("The Top variables highly correlated with Target")
print (cor['target'].sort_values(ascending=False)[:10], '\n') 
#Correlation Analysis among features

# Generate correlation matrix
corr_matrix = Kdd_Cup_Data.corr().abs()

# Choose upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

# Find index of variable columns with correlation greater than 0.60
to_drop = [column for column in upper.columns if any(upper[column] > 0.60)]
print(to_drop)

# Delete Marked Variables
Kdd_Cup_Data.drop(to_drop, axis=1, inplace=True)
#Display Important variables
print("Number of total features are",len(Kdd_Cup_Data.columns))
print()
print("The important variables are:",list(Kdd_Cup_Data.columns))
Kdd_Cup_Data
# Divide dependent and independent variables
X=Kdd_Cup_Data.iloc[:,1:80]
y=Kdd_Cup_Data['target']
X
y
#Standardize the data
num_feat = [f for f in X.columns if X[f].dtype != object]

Std_scale = StandardScaler()
Std_scale.fit(X[num_feat])
Scaled_Num = Std_scale.transform(X[num_feat])

for i, col in enumerate(num_feat):
       X[col] = Scaled_Num[:,i]
X
#Store dataset in new dataframe for further use
X_Added = X
# Splitting the new training dataset into 70:30 ratio for model building
train_dist = int(0.7 * len(X))
X_train, X_test = X[:train_dist], X[train_dist:]
y_train, y_test = y[:train_dist], y[train_dist:]
init_seed = 0
random.seed(init_seed)
np.random.seed(init_seed)
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
logreg_model.fit(X_train,y_train)
logreg_pred = logreg_model.predict(X_test)
logreg_RMSE = np.sqrt(mean_squared_error(y_test,logreg_pred))
logreg_model_features = X_train.columns
print("Accuracy of Logistic Regression Without Interaction Terms is ",(logreg_model.score(X_test, y_test) * 100))
print()
print("RMSE value of the Logistic Regression Without Interaction Terms is ",logreg_RMSE)
print()
print("Predicted values from Logistic Regression Without Interaction Terms are ", logreg_pred)
print("β0 for Logistic Regression without Interaction is ",logreg_model.intercept_)
print()
print()
print("β values for Logistic Regression without Interaction are", list(zip(np.round(logreg_model.coef_[0],4),logreg_model_features)))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Create a no skill prediction 
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
lr_probs = logreg_model.predict_proba(X_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test,lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test,lr_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#Confusion Matrix
print("Confusion Matrix: ")
print(confusion_matrix(y_test, logreg_pred))
print()

value = (confusion_matrix(y_test, logreg_pred))
FP = value.sum(axis=0) - np.diag(value)  
FN = value.sum(axis=1) - np.diag(value)
TP = np.diag(value)
TN = value.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("True Positive Rate is:",TPR)
print()
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("False Positive Rate is:",FPR)
print()
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

#Classification Report
print("Classification Report: ")
print(classification_report(y_test, logreg_pred));
X_Added
#Adding New Features as Interaction Terms
X_Added['feat1*feat4'] = X_Added['feat1']*X_Added['feat4']
X_Added['feat5*feat8'] = X_Added['feat5']*X_Added['feat8']
X_Added['feat13*feat14'] = X_Added['feat13']*X_Added['feat14']
X_Added['feat16*feat31'] = X_Added['feat16']*X_Added['feat31']
X_Added
# Splitting the new training dataset into 70:30 ratio for model building
train_dist = int(0.7 * len(X_Added))
X_Added_train, X_Added_test = X_Added[:train_dist], X_Added[train_dist:]
y_train, y_test = y[:train_dist], y[train_dist:]
from sklearn.linear_model import LogisticRegression

logreg_inter_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
logreg_inter_model = logreg_inter_model.fit(X_Added_train, y_train)
logreg_inter_pred = logreg_inter_model.predict(X_Added_test)
logreg_inter_RMSE = np.sqrt(mean_squared_error(y_test,logreg_inter_pred))
logreg_inter_model_features = X_Added_train.columns
print("Accuracy of Logistic Regression with Interaction Terms is ",(logreg_inter_model.score(X_Added_test, y_test) * 100))
print()
print("RMSE value of the Logistic Regression with Interaction Terms is ",logreg_inter_RMSE)
print()
print("Predicted values from Logistic Regression with Interaction Terms are ", logreg_inter_pred)
print("β0 for Logistic Regression with Interaction is ",logreg_inter_model.intercept_)
print()
print()
print("β values for Logistic Regression with Interaction are", list(zip(np.round(logreg_inter_model.coef_[0] ,4),logreg_inter_model_features)))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
logreg_inter_probs = logreg_inter_model.predict_proba(X_Added_test)

# keep probabilities for the positive outcome only
logreg_inter_probs = logreg_inter_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
logreg_inter_auc = roc_auc_score(y_test,logreg_inter_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Interaction Logistic: ROC AUC=%.3f' % (logreg_inter_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
logreg_inter_fpr, logreg_inter_tpr, _ = roc_curve(y_test,logreg_inter_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(logreg_inter_fpr, logreg_inter_tpr, marker='.', label='Interaction Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#Confusion Matrix
print("Confusion Matrix: ")
print(confusion_matrix(y_test, logreg_inter_pred))
print()

value = (confusion_matrix(y_test, logreg_inter_pred))
FP = value.sum(axis=0) - np.diag(value)  
FN = value.sum(axis=1) - np.diag(value)
TP = np.diag(value)
TN = value.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("True Positive Rate is:",TPR)
print()
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("False Positive Rate is:",FPR)
print()
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

#Classification Report
print("Classification Report: ")
print(classification_report(y_test,logreg_inter_pred));

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=100, max_features="auto", max_depth = 2, bootstrap = True, random_state = 0)
RF_model.fit(X_train,y_train)
RF_pred = RF_model.predict(X_test)
RF_RMSE = np.sqrt(mean_squared_error(y_test,RF_pred))
print("Accuracy of RandomForest Model is ",(RF_model.score(X_test, y_test) * 100))
print()
print("RMSE value of the RandomForest model is ",RF_RMSE)
print()
print("Predicted values from RandomForest Model are ", RF_pred)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
RF_probs = RF_model.predict_proba(X_test)

# keep probabilities for the positive outcome only
RF_probs = RF_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
RF_auc = roc_auc_score(y_test,RF_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('RandomForest: ROC AUC=%.3f' % (RF_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
RF_fpr, RF_tpr, _ = roc_curve(y_test,RF_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(RF_fpr, RF_tpr, marker='.', label='RandomForest')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#Confusion Matrix
print("Confusion Matrix: ")
print(confusion_matrix(y_test, RF_pred))
print()
print()

value = (confusion_matrix(y_test, RF_pred))
FP = value.sum(axis=0) - np.diag(value)  
FN = value.sum(axis=1) - np.diag(value)
TP = np.diag(value)
TN = value.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("True Positive Rate is:",TPR)
print()
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("False Positive Rate is:",FPR)
print()
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

#Classification Report
print("Classification Report: ")
print(classification_report(y_test, RF_pred));

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.1, max_features="auto", max_depth = 2, random_state = 0)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_RMSE = np.sqrt(mean_squared_error(y_test,gb_pred))
print("Accuracy of Gradient Boosting Model is ",(gb_model.score(X_test, y_test) * 100))
print()
print("RMSE value of the Gradient Boosting model is ",gb_RMSE)
print()
print("Predicted values from Gradient Boosting Model are ", gb_pred)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
gb_probs = gb_model.predict_proba(X_test)

# keep probabilities for the positive outcome only
gb_probs = gb_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
gb_auc = roc_auc_score(y_test,gb_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('GradientBoosting: ROC AUC=%.3f' % (gb_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test,gb_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(gb_fpr, gb_tpr, marker='.', label='GradientBoosting')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#Confusion Matrix
print("Confusion Matrix: ")
print(confusion_matrix(y_test, gb_pred))
print()

value = (confusion_matrix(y_test, gb_pred))
FP = value.sum(axis=0) - np.diag(value)  
FN = value.sum(axis=1) - np.diag(value)
TP = np.diag(value)
TN = value.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("True Positive Rate is:",TPR)
print()
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("False Positive Rate is:",FPR)
print()
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

#Classification Report
print("Classification Report: ")
print(classification_report(y_test, gb_pred));

print("Accuracy of Logistic Regression Without Interaction Terms is %.2f"  % (logreg_model.score(X_test, y_test) * 100))
print("Accuracy of Logistic Regression with Interaction Terms is %.2f"  % (logreg_inter_model.score(X_Added_test, y_test) * 100))
print("Accuracy of RandomForest Model is %.2f"  % (RF_model.score(X_test, y_test) * 100))
print("Accuracy of Gradient Boosting Model is %.2f"  % (gb_model.score(X_test, y_test) * 100))
print()
print()
print('Logistic: ROC AUC=%.3f' % (lr_auc * 100))
print('Interaction Logistic: ROC AUC=%.3f' % (logreg_inter_auc * 100))
print('RandomForest: ROC AUC=%.3f' % (RF_auc * 100))
print('GradientBoosting: ROC AUC=%.3f' % (gb_auc * 100))

