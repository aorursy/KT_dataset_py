import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections

import xgboost as xgb





# Other Libraries

from imblearn.datasets import fetch_datasets

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from imblearn.pipeline import make_pipeline as make_pipeline_imb # To do our transformation in a unique time

from imblearn.over_sampling import SMOTE

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, fbeta_score, precision_recall_curve

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcard.csv")
df.head()
pd.options.display.max_columns = 300
print('Shape: ',df.shape)
print('\nColumns: ',df.columns.values)
print('\nData types:\n',df.dtypes.value_counts())
df.describe()
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=df, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
df.isnull().sum().sum()
plt.figure(figsize = (10,10))

plt.title('Credit Card Transactions features correlation plot (Pearson)')

corr = df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens")

plt.show()
s = sns.lmplot(x='V20', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V7', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
s = sns.lmplot(x='V2', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V5', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
target = 'Class'

predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount']
train_df, test_df = train_test_split(df, test_size=0.20, random_state=2018, shuffle=True )

train_df, valid_df = train_test_split(train_df, test_size=0.20, random_state=2018, shuffle=True )
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)

dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)

dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)
#What to monitor (in this case, **train** and **valid**)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Set xgboost parameters

params = {}

params['objective'] = 'binary:logistic'

params['eta'] = 0.039

params['silent'] = True

params['max_depth'] = 2

params['subsample'] = 0.8

params['colsample_bytree'] = 0.9

params['eval_metric'] = 'auc'

params['random_state'] = 2018
model = xgb.train(params, 

                dtrain, 

                1000, 

                watchlist, 

                early_stopping_rounds=50, 

                maximize=True, 

                verbose_eval=50)
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))

xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green") 

plt.show()
preds = model.predict(dtest)
roc_auc_score(test_df[target].values, preds)
classifier = RandomForestClassifier

X = df.drop(["Class"], axis=1).values #Setting the X to do the split

y = df["Class"].values # transforming the values in array

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.20)# splitting data into training and test set
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), \

                                   classifier(random_state=42))



smote_model = smote_pipeline.fit(X_train, y_train)

smote_prediction = smote_model.predict(X_test)
print("normal data distribution: {}".format(Counter(y)))

X_smote, y_smote = SMOTE().fit_sample(X, y)

print("SMOTE data distribution: {}".format(Counter(y_smote)))
print("Confusion Matrix: ")

print(confusion_matrix(y_test, smote_prediction))
print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))
# the function that we will use to better evaluate the model

def print_results(headline, true_value, pred):

    print(headline)

    print("accuracy: {}".format(accuracy_score(true_value, pred)))

    print("precision: {}".format(precision_score(true_value, pred)))

    print("recall: {}".format(recall_score(true_value, pred)))

    print("f2: {}".format(fbeta_score(true_value, pred, beta=2)))
print_results("\nSMOTE + RandomForest classification", y_test, smote_prediction)