# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler



#for PCA

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from sklearn.metrics import r2_score



from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



from sklearn import metrics



from sklearn import svm 

from fancyimpute import KNN 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import roc_curve, auc

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
dataset=pd.read_csv('../input/janatahack-crosssell-prediction/train.csv')

test=pd.read_csv('../input/janatahack-crosssell-prediction/test.csv')
dataset.shape
dataset.info()
dataset.head()
dataset.describe()
null_value_table=pd.DataFrame((dataset.isna().sum()/dataset.shape[0])*100).sort_values(0,ascending=False )

null_value_table.columns=['null percentage']

null_value_table[null_value_table['null percentage']>0]
dataset['Vehicle_Age'].value_counts()
dataset['Response'].value_counts()
sns.countplot(x='Response', data=dataset)
dataset['Response'].value_counts()[1]/dataset['Response'].value_counts()[0]
dataset.describe(include='object').columns
cat_columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage','Driving_License','Previously_Insured']
num_columns=[ 'Age', 'Region_Code','Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
dataset.describe().columns
dataset['Driving_License'].value_counts()
dataset['Previously_Insured'].value_counts()
corr=dataset.corr()

corr
# Let's see the correlation matrix 

plt.figure(figsize = (15,10))        # Size of the figure

sns.heatmap(corr,annot = True)

plt.show()
ax = sns.boxplot(x="Response", y="Annual_Premium", data=dataset)
ax = sns.boxplot(x="Response", y="Vintage", data=dataset)
ax = sns.boxplot(x="Response", y="Policy_Sales_Channel", data=dataset)
ax = sns.boxplot(x="Response", y="Age", data=dataset)
import pandas as pd

# dummy encode the categorical columns

dataset_dummies = pd.concat([dataset,pd.get_dummies(dataset[cat_columns], drop_first=True)], axis=1)



# drop the original columns

dataset_dummies.drop(cat_columns, axis=1, inplace=True)
dataset_dummies.head()
import pandas as pd

# dummy encode the categorical columns

dataset_dummies = pd.concat([dataset,pd.get_dummies(dataset[cat_columns], drop_first=True)], axis=1)



# drop the original columns

dataset_dummies.drop(cat_columns, axis=1, inplace=True)
import pandas as pd

# dummy encode the categorical columns

Test_dataset_dummies = pd.concat([test,pd.get_dummies(test[cat_columns], drop_first=True)], axis=1)



# drop the original columns

Test_dataset_dummies.drop(cat_columns, axis=1, inplace=True)
Test_dataset_dummies.drop('id',axis=1)
y = dataset_dummies['Response']

X = dataset_dummies.drop(['Response','id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1,stratify=y)

print("Dimension of X_train:", X_train.shape)

print("Dimension of X_test:", X_test.shape)
#from sklearn.preprocessing import MinMaxScaler



#scaler = MinMaxScaler()



#X_train = scaler.fit_transform(X_train)



#X_train.head()
#X_test = scaler.transform(X_test)



#X_test.head()
def CalculateMetrics(confusion):

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    print("Sesitivity For the Model : ",(TP / float(TP+FN)))

    print("specificity For the Model : ",(TN / float(TN+FP)))

    print("false postive rate For the Model : ",(FP/ float(TN+FP)))

    print("precision/false postive rate For the Model : ",(TP / float(TP+FP)))

    print("Negative predictive value For the Model : ",(TN / float(TN+ FN)))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = roc_curve( actual, probs, drop_intermediate = False )

    auc_score = roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 6))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()
# Logistic Regression

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression(class_weight='balanced',random_state=2)

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))

# Confusion matrix 

confusion = metrics.confusion_matrix(y_test, y_pred)

CalculateMetrics(confusion)


# ROC-AUC curve

draw_roc(y_test, y_pred)


import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
col = X_train.columns
def calculateVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred = y_train_pred * 100

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Conversion':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['id'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Conversion, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.predicted))
calculateVIF(X_train[col])
col=col.drop('Age',1)
col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
calculateVIF(X_train[col])
col=col.drop('Vintage',1)
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
calculateVIF(X_train[col])
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Conversion':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['id'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Conversion, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.predicted))
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Conversion, y_train_pred_final.Conversion_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Conversion, y_train_pred_final.Conversion_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Conversion, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.22 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Conversion, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Conversion, y_train_pred_final.final_predicted)

confusion2
CalculateMetrics(confusion2)
X_test = X_test[col]

X_test.head()
#Adding constant

X_test_sm = sm.add_constant(X_test)

#Making prediction

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Putting CustID to index

y_test_df['id'] = y_test_df.index

# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()

# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})

# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.22 else 0)


print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred_final['final_predicted'])))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred_final['final_predicted'])))

print("AUC_ROC",roc_auc_score(y_test, y_pred_final['Conversion_Prob']))

confusion = metrics.confusion_matrix(y_test, y_pred_final['final_predicted'])

CalculateMetrics(confusion)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4, stratify = y)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function

logistic = LogisticRegression(class_weight='balanced',random_state=42)



# hyperparameter space

params = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'penalty': ['l1', 'l2']}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

model = GridSearchCV(logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
# Fit the grid search to the data

model.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',model.best_score_,'using',model.best_params_)
# Initialize the Logistic regression

model = LogisticRegression(class_weight='balanced',C=0.1, penalty='l2',random_state=42)

# fit the pca training data

model.fit(X_train, y_train)

# predict the testing pca data

y_pred = model.predict(X_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))

confusion = metrics.confusion_matrix(y_test, y_pred)

CalculateMetrics(confusion)
print("ROC_AUC",roc_auc_score(y_test, y_pred))
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler

Minmaxscaler = MinMaxScaler()

#X_scaled = Minmaxscaler.fit_transform(X)



X_train_NB, X_test_NB, y_train_NB, y_test_NB = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=111)



mnb = MultinomialNB()



# fit

mnb.fit(X_train_NB,y_train_NB)



# predict class

predictions = mnb.predict(X_test_NB)



# predict probabilities

y_pred_proba = mnb.predict_proba(X_test_NB)

accuracy = metrics.accuracy_score(y_test_NB, predictions)

print("Classification Report:")

print(classification_report(y_test_NB,predictions))

fpr, tpr, threshold = metrics.roc_curve(y_test_NB, predictions)

roc_auc = metrics.auc(fpr, tpr)

print("Accuracy {}".format(metrics.accuracy_score(y_test_NB, predictions)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test_NB, predictions)))

confusion = metrics.confusion_matrix(y_test_NB, predictions)

CalculateMetrics(confusion)

print("Accuracy for the test dataset",'{:.1%}'.format(accuracy) )

print("ROC for the test dataset",'{:.1%}'.format(roc_auc))

plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))

plt.legend(loc=4)

plt.show()
import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance
import re



regex = re.compile(r"\[|\]|<", re.IGNORECASE)



X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

Test_dataset_dummies.columns= [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in Test_dataset_dummies.columns.values]
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100,stratify=y,shuffle=True)
# fit model on training data with default hyperparameters

model = XGBClassifier(class_weight='balanced',random_state=2)

model.fit(X_train, y_train)
# make predictions for test data

# use predict_proba since we need probabilities to compute auc

y_pred = model.predict_proba(X_test)

y_pred[:10]


# evaluate predictions

import sklearn.metrics as metrics

roc = metrics.roc_auc_score(y_test, y_pred[:, 1])

print("Area under the curve: %.2f%%" % (roc * 100.0))
# hyperparameter tuning with XGBoost

from sklearn.model_selection import GridSearchCV

# creating 3 Fold object 

folds = 3



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_model = XGBClassifier(class_weight='balanced',max_depth=2, n_estimators=200)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'roc_auc', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)
# fit the model

model_cv.fit(X_train, y_train)
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# # plotting

plt.figure(figsize=(16,6))



param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]} 





for n, subsample in enumerate(param_grid['subsample']):

    



    plt.subplot(1,len(param_grid['subsample']), n+1)

    plot_df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(plot_df["param_learning_rate"], plot_df["mean_test_score"])

    plt.plot(plot_df["param_learning_rate"], plot_df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('AUC')

    plt.title("subsample={0}".format(subsample))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='upper left')

    plt.xscale('log')
# chosen hyperparameters

# from the above graph it is evident that learning_rate =0.2 and subsample=0.9 produces a model with higher AUC with overfitting.

#so using those parameters for the final model.

params = {'learning_rate': 0.2,

          'max_depth': 2, 

          'n_estimators':200,

          'subsample':0.9,

         'objective':'binary:logistic'}



# fit model on training data

model = XGBClassifier(params = params)

model.fit(X_train, y_train)


# predict

#y_pred_actual was for the predictions interms of churns

#y_pred is for probabilistic predictions used for AUC.

y_pred_actual = model.predict(X_test)

y_pred = model.predict_proba(X_test)

y_pred[:10]

auc = roc_auc_score(y_test, y_pred[:, 1])

auc
submmission = pd.read_csv('../input/janatahack-crosssell-prediction/sample_submission.csv')
Test_dataset_dummies.drop('id',axis=1,inplace=True)
cat_pred= model.predict_proba(Test_dataset_dummies)[:, 1]

submmission['Response']=cat_pred

submmission.to_csv("final_output.csv", index = False)