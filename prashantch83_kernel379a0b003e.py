# Importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm

import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

# To ignore warnings

import warnings

warnings.filterwarnings("ignore")
pc = pd.read_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/train.csv")
pc_test = pd.read_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/test.csv")
pd.set_option('display.max_row', 1000)

pd.set_option('display.max_columns', 300)

pd.set_option('display.max_colwidth',5000)
pc.head()
pc.Response.value_counts()
# The 'Response' shows that potential customers have been classified into 8 categories

# Most have been classified as level '8', followed by '6' and '7'
pc.info(verbose = True)
# It can be seen that there are 18 float type, 109 int type and 1 object type variables in the dataset. 

#Total columns are 128, where 'Id' and 'Response' will not be part of model learning as they are customer 'id' and 'Target' 

# values respectively 
cols4 = pc.select_dtypes(include=['int64']).columns.values

cols5 = pc.select_dtypes(include = ['float64']).columns.values
round((pc.isnull().sum()/len(pc.index))*100,2)
# As we have lot of columns, will remove columns with above 50% missing values

col = []

for i in pc.columns:

    if round((pc[i].isnull().sum()/len(pc.index))*100,2) >= 50:

        col.append(i)

print(col)

print(len(col))

    
# Removing these columns

pc = pc.drop(col, axis = 1)
round((pc.isnull().sum()/len(pc.index))*100,2)
# Focussing only on columns that have missing values

col = []

for i in pc.columns:

    if round((pc[i].isnull().sum()/len(pc.index))*100,2) != 0:

        col.append(i)

print(col)

print(len(col))

# Analysing 'Employment_Info_1' feature

pc["Employment_Info_1"].describe()
pc["Employment_Info_1"].value_counts()
pc["Employment_Info_1"].max()
# These are normalized values related employment history and since there's insignificant number of missing rows in this case,

# will remove the missing rows rather than imputing them

pc = pc[~pd.isnull(pc["Employment_Info_1"])]
round((pc[col].isnull().sum()/len(pc.index))*100,2)
pc["Employment_Info_4"].describe()
pc["Employment_Info_4"].value_counts()
# We can see that '0' value dominates the distribution of values within this feature. Therefore, imputing value '0' 

# for missing values in this case

pc.loc[pd.isnull(pc["Employment_Info_4"]),"Employment_Info_4"] = 0 
round((pc[col].isnull().sum()/len(pc.index))*100,2)
pc.index = pd.RangeIndex(1, len(pc.index) + 1)
pc["Employment_Info_6"].describe()
pc["Employment_Info_6"].value_counts()
pc["Insurance_History_5"].describe()
pc["Insurance_History_5"].value_counts()
pc["Family_Hist_2"].describe()
pc["Family_Hist_2"].value_counts()
pc["Family_Hist_4"].describe()
pc["Family_Hist_4"].value_counts()
pc["Medical_History_1"].describe()
pc["Medical_History_1"].value_counts()
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
round((pc[col].isnull().sum()/len(pc.index))*100,2)
#Before imputing the values, will first analyze the 'object' feature 

pc["Product_Info_2"].value_counts()
lc = LabelEncoder()
pc["Product_Info_2"] = lc.fit_transform(pc["Product_Info_2"])
pc["Product_Info_2"].describe()
pc["Product_Info_2"].value_counts()
# Imputing values using Iterative Imputer

cols2 = pc.columns
pc_imp = pd.DataFrame(IterativeImputer().fit_transform(pc))
pc_imp.columns = cols2
pc_imp.head()
round((pc_imp[col].isnull().sum()/len(pc_imp.index))*100,2)
pc_imp[cols4] = pc_imp[cols4].astype(int)
pc_imp.info(verbose = True)
pc_imp["Product_Info_2"] = pc_imp["Product_Info_2"].astype(int)
# Removing 'Id' variable as it will not be used for model learning

pc_imp = pc_imp.drop('Id', axis =1)
pc_imp.head()
# Box plots for outlier analysis

col = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI"]

coln = ["Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5"]

col2 = ["Family_Hist_2", "Family_Hist_4"]
sns.boxplot(data = pc_imp[col], orient = 'v')
# We can see that there are few outliers in the case of features: 'Ht', 'Wt' and 'BMI'
plt.figure(figsize = (20,12))

sns.boxplot(data = pc_imp[coln], orient = 'v')
# Outlier removal for 'BMI' and 'Employment_Info_6'

Q1= pc_imp['BMI'].quantile(0.5)

Q3= pc_imp['BMI'].quantile(0.95)

Range=Q3-Q1

print(Range)

pc_imp= pc_imp[(pc_imp['BMI'] >= Q1-1.5*Range) & (pc_imp['BMI'] <= Q3+1.5*Range) ]
Q1= pc_imp['Employment_Info_6'].quantile(0.5)

Q3= pc_imp['Employment_Info_6'].quantile(0.95)

Range=Q3-Q1

pc_imp= pc_imp[(pc_imp['Employment_Info_6'] >= Q1-1.5*Range) & (pc_imp['Employment_Info_6'] <= Q3+1.5*Range) ]
plt.figure(figsize = (10,10))

sns.boxplot(data = pc_imp[col2], orient = 'v')
# pairplot analysis to understand correlation between continous variables

colc = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2","Family_Hist_4"]
sns.pairplot(pc_imp[colc])
colc = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2","Family_Hist_4"]
plt.figure(figsize = (15, 15))

sns.heatmap(pc_imp[colc].corr(), annot = True, square = True, cmap="YlGnBu")

plt.show()
pc_imp = pc_imp.drop(["Ht", "Wt"], axis = 1)
sns.boxplot(x = "Response",y = "Family_Hist_2", data = pc_imp)
sns.boxplot(x = "Response",y = "Family_Hist_4", data = pc_imp)
sns.boxplot(x = "Response",y = "Ins_Age", data = pc_imp)
sns.boxplot(x = "Response",y = "BMI", data = pc_imp)
pc_imp = pc_imp.drop("Family_Hist_4", axis =1)
pc_imp["Product_Info_1"].value_counts()
pc_imp["Medical_History_8"].value_counts()
pc_imp["Medical_History_30"].value_counts()
pc_imp["Medical_History_1"].value_counts()
pc_imp["Medical_History_2"].value_counts()
pc_imp.shape
pc_imp.info(verbose = True)
#Preparing the data in the 'test' dataset as well now
# round((pc_test.isnull().sum()/len(pc_test.index))*100,2)
#First deleting the columns in 'test' dataset that have been deleted in the 'train' dataset

#colt = ['Family_Hist_3', 'Family_Hist_5', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32','Ht','Wt','Family_Hist_4','Id']
#pc_test_rev = pc_test.drop(colt, axis = 1)
#pc_test_rev.shape
# pc_test_rev.info(verbose = True)
#Splitting data between test and train data sets

pc_imp_train, pc_imp_test = train_test_split(pc_imp, train_size = 0.7, test_size = 0.3, random_state = 100)
y_pc_imp_train = pc_imp_train.pop("Response")
X_pc_imp_train = pc_imp_train
y_pc_imp_test = pc_imp_test.pop("Response")
X_pc_imp_test = pc_imp_test
X_pc_imp_train.shape
X_pc_imp_test.shape
# Will now do PCA to reduce dimensionality

pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(X_pc_imp_train)
pca.components_
colnames = list(X_pc_imp_train.columns)
#Dataframe with features and respective first two Prinicple components

pca_df = pd.DataFrame({'PC1':pca.components_[0], 'PC2':pca.components_[1], 'Features': colnames})
pca_df.head()
%matplotlib inline

fig = plt.figure(figsize = (15,15))

plt.scatter(pca_df.PC1, pca_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pca_df.Features):

    plt.annotate(txt, (pca_df.PC1[i],pca_df.PC2[i]))

plt.tight_layout()

plt.show()
pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
# Plotting the cummulative variance and number of PCs graph to identify the correct number of PCs required to explain 95% of variance

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Number of components")

plt.ylabel("Cummulative variance")

plt.show()
# Base Random Forest result

rfc=RandomForestClassifier()
rfc.fit(X_pc_imp_train, y_pc_imp_train)
Tr_predict = rfc.predict(X_pc_imp_train)
# Let's check the report of our default model

print(classification_report(y_pc_imp_train,Tr_predict))
# Printing confusion matrix

print(confusion_matrix(y_pc_imp_train,Tr_predict))
test_predict = rfc.predict(X_pc_imp_test)
# Let's check the report of default model on test dataset

print(classification_report(y_pc_imp_test,test_predict))
# Max depth tuning using CV an

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(10, 80, 5)}



# instantiate the base model

rf_m = RandomForestClassifier()





# fit tree on training data

rf_m = GridSearchCV(rf_m, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                    return_train_score=True)

rf_m.fit(X_pc_imp_train, y_pc_imp_train)
# scores of GridSearch CV

scores = rf_m.cv_results_

# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# n_estimators tuning using CV and keeping tuned 'max_depth' of 15

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(300, 2400, 300)}



# instantiate the base model

rf_e = RandomForestClassifier(max_depth=15)





# fit tree on training data

rf_e = GridSearchCV(rf_e, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                    return_train_score=True)

rf_e.fit(X_pc_imp_train, y_pc_imp_train)
sc = pd.DataFrame(scores)
sc.head()
# scores of GridSearch CV

scores = rf_e.cv_results_

# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# tuning Max_features

# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_features': [10,20,30,40,50]}



# instantiate the model

rf_mx = RandomForestClassifier(max_depth=15, n_estimators = 1250)





# fit tree on training data

rf_mx = GridSearchCV(rf_mx, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                    return_train_score=True)

rf_mx.fit(X_pc_imp_train, y_pc_imp_train)
scores = rf_mx.cv_results_
sc = pd.DataFrame(scores)
sc.head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_features"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_features"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# tuning min_samples_leaf

# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(50, 1000, 50)}



# instantiate the model

rf_sl = RandomForestClassifier(max_depth=15, max_features = 30, n_estimators = 1250)





# fit tree on training data

rf_sl = GridSearchCV(rf_sl, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                    return_train_score=True)

rf_sl.fit(X_pc_imp_train, y_pc_imp_train)
scores = rf_sl.cv_results_
sc = pd.DataFrame(scores)
sc.head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

rf_f = RandomForestClassifier(max_depth=15, n_estimators = 1250, max_features = 30, min_samples_leaf = 800)
rf_f.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred = rf_f.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred))
print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))
rf_f = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 40, min_samples_leaf = 40)
rf_f.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred = rf_f.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred))
print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))
rf_pred = rf_f.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred))
print(metrics.precision_score(y_pc_imp_test, rf_pred, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred, average = 'weighted'))
rf_f_1 = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 50, min_samples_leaf = 30)
rf_f_1.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred = rf_f_1.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred))
print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))
rf_pred_test = rf_f_1.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test, average = 'weighted'))
rf_f_2 = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 50, min_samples_leaf = 20)
rf_f_2.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_2 = rf_f_2.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_2))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_2))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_2, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_2, average = 'weighted'))
rf_pred_test_2 = rf_f_2.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_2))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_2))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_2, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_2, average = 'weighted'))
rf_f_3 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 25)
rf_f_3.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_3 = rf_f_3.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_3))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_3))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_3, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_3, average = 'weighted'))
rf_pred_test_3 = rf_f_3.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_3))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_3))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_3, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_3, average = 'weighted'))
rf_f_4 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20)
rf_f_4.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_4 = rf_f_4.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_4))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_4))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_4, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_4, average = 'weighted'))
rf_pred_test_4 = rf_f_4.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_4))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_4))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_4, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_4, average = 'weighted'))
rf_f_5 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 15)
rf_f_5.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_5 = rf_f_5.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_5))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_5))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_5, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_5, average = 'weighted'))
rf_pred_test_5 = rf_f_5.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_5))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_5))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_5, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_5, average = 'weighted'))
rf_f_6 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 12)
rf_f_6.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_6 = rf_f_6.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_6))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_6))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_6, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_6, average = 'weighted'))
rf_pred_test_6 = rf_f_6.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_6))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_6))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_6, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_6, average = 'weighted'))
rf_f_7 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 8)
rf_f_7.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_7 = rf_f_7.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_7))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_7))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_7, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_7, average = 'weighted'))
rf_pred_test_7 = rf_f_7.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_7))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_7))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_7, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_7, average = 'weighted'))
rf_f_8 = RandomForestClassifier(max_depth=20, n_estimators = 1800, max_features = 30, min_samples_leaf = 15)
rf_f_8.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_8 = rf_f_8.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_8))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_8))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_8, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_8, average = 'weighted'))
rf_pred_test_8 = rf_f_8.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_8))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_8))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_8, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_8, average = 'weighted'))
# Max depth tuning using CV an

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(10, 120, 5)}



# instantiate the base model

rf_m = RandomForestClassifier(n_estimators = 1600, max_features = 30, min_samples_leaf = 15)





# fit tree on training data

rf_m = GridSearchCV(rf_m, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                    return_train_score=True)

rf_m.fit(X_pc_imp_train, y_pc_imp_train)
# scores of GridSearch CV

scores = rf_m.cv_results_

# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

rf_f_9 = RandomForestClassifier(max_depth=18, n_estimators = 1800, max_features = 25, min_samples_leaf = 15)
rf_f_9.fit(X_pc_imp_train, y_pc_imp_train)
rf_pred_train_9 = rf_f_9.predict(X_pc_imp_train)
print(classification_report(y_pc_imp_train,rf_pred_train_9))
print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_9))
print(metrics.precision_score(y_pc_imp_train, rf_pred_train_9, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_train, rf_pred_train_9, average = 'weighted'))
rf_pred_test_9 = rf_f_9.predict(X_pc_imp_test)
print(classification_report(y_pc_imp_test,rf_pred_test_9))
print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_9))
print(metrics.precision_score(y_pc_imp_test, rf_pred_test_9, average = 'weighted'))
print(metrics.recall_score(y_pc_imp_test, rf_pred_test_9, average = 'weighted'))
round((pc_test.isnull().sum()/len(pc_test.index))*100,2)
#First deleting the columns in 'test' dataset that have been deleted in the 'train' dataset

colt = ['Family_Hist_3', 'Family_Hist_5', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32','Ht','Wt','Family_Hist_4','Id']
pc_test_rev = pc_test.drop(colt, axis = 1)
pc_test_rev.shape
round((pc_test_rev.isnull().sum()/len(pc_test_rev.index))*100,2)
# Focussing only on columns that have missing values

col = []

for i in pc_test_rev.columns:

    if round((pc_test_rev[i].isnull().sum()/len(pc_test_rev.index))*100,2) != 0:

        col.append(i)

print(col)

print(len(col))

# Employment_Info_1 and Employment_Info_4 have quite low percentage of missing values so will just remove the rows

pc_test_rev = pc_test_rev[~pd.isnull(pc_test_rev["Employment_Info_1"])]
pc_test_rev = pc_test_rev[~pd.isnull(pc_test_rev["Employment_Info_4"])]
round((pc_test_rev[col].isnull().sum()/len(pc_test_rev[col].index))*100,2)
pc_test_rev.index = pd.RangeIndex(1, len(pc_test_rev.index) + 1)
# Will treat missing values via iterative imputer for rest of columns. Will first encode the column 'Product_Info_2'
lc = LabelEncoder()
pc_test_rev["Product_Info_2"] = lc.fit_transform(pc_test_rev["Product_Info_2"])
# Imputing values using Iterative Imputer

colt2 = pc_test_rev.columns
pc_test_imp = pd.DataFrame(IterativeImputer().fit_transform(pc_test_rev))
pc_test_imp.columns = colt2
pc_test_imp.head()
round((pc_test_imp[col].isnull().sum()/len(pc_test_imp.index))*100,2)
colt4 = np.delete(cols4,0)
print(len(colt4))
colt4 = np.delete(colt4,107)
print(colt4)
pc_test_imp[colt4] = pc_test_imp[colt4].astype(int)
pc_test_imp["Product_Info_2"] = pc_test_imp["Product_Info_2"].astype(int)
pc_test_imp.head()
rf_pred_sub = rf_f_9.predict(pc_test_imp)
rf_pred_sub
rf_pred_sub.reshape(-1)
pred_rk = pd.DataFrame({'Risk': rf_pred_sub})
pred_rk.head()
pred_rk.index = pd.RangeIndex(1, len(pred_rk.index) + 1)
len(pred_rk)
pc_test.index = pd.RangeIndex(1, len(pc_test.index) + 1)
pc_test.head()
fn_dt = pd.DataFrame()
fn_dt['Id'] = pc_test['Id'] 
fn_dt['Response'] = pred_rk['Risk']
# Final dataframe to be submitted is:

fn_dt.head()
# Writing this to csv file for submission

fn_dt.to_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/submission_file.csv")