import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



from imblearn.over_sampling import SMOTE



from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

sns.set()

plt.rcParams['figure.figsize'] = (12,5)



# Any results you write to the current directory are saved as output.
dpath = '../input/'

telcodf = pd.read_csv(dpath + "WA_Fn-UseC_-Telco-Customer-Churn.csv")
telcodf.head(5)
telcodf.info()
telcodf['Churn'].value_counts() / telcodf['customerID'].nunique() * 100
telcodf.head()
telcodf.dtypes.sort_values()
catvars = telcodf.select_dtypes(include = 'object').columns.tolist()

telcodf[catvars] = telcodf[catvars].replace('',"NA")

telcodf[catvars] = telcodf[catvars].replace(' ',"NA")
telcodf[catvars].apply(lambda x: x.str.contains('NA')).sum()
# Replace the NA with a zero #

telcodf['TotalCharges'] = telcodf['TotalCharges'].str.replace('NA', '0')
# Change the datatypes correctly #

telcodf['TotalCharges'] = pd.to_numeric(telcodf['TotalCharges'])

telcodf['SeniorCitizen'] = telcodf['SeniorCitizen'].astype('object')
fig, axes = plt.subplots(1,2, figsize  = (10,4))

sns.distplot(telcodf[telcodf['Churn'] == 'Yes']['tenure'], label = 'Yes', ax = axes[0])

sns.distplot(telcodf[telcodf['Churn'] == 'No']['tenure'], label = 'No', ax = axes[0])

axes[0].set_title("Tenure Distribution")

plt.legend()



sns.violinplot(x = 'Churn', y = 'tenure', data = telcodf, ax = axes[1])

axes[1].set_title("Number of Lapses by Tenure")



plt.show()
fig, axes = plt.subplots(1,2, figsize  = (10,4))

sns.distplot(telcodf[telcodf['Churn'] == 'Yes']['MonthlyCharges'], label = 'Yes', ax = axes[0])

sns.distplot(telcodf[telcodf['Churn'] == 'No']['MonthlyCharges'], label = 'No', ax = axes[0])

axes[0].set_title("Monthly Charges Distribution")

plt.legend()



sns.violinplot(x = 'Churn', y = 'MonthlyCharges', data = telcodf, ax = axes[1])

axes[1].set_title("Monthly Charges")



plt.show()
fig, axes = plt.subplots(1,2, figsize  = (10,4))

sns.distplot(telcodf[telcodf['Churn'] == 'Yes']['TotalCharges'], label = 'Yes', ax = axes[0])

sns.distplot(telcodf[telcodf['Churn'] == 'No']['TotalCharges'], label = 'No', ax = axes[0])

axes[0].set_title("Total Charges Distribution")

plt.legend()



sns.violinplot(x = 'Churn', y = 'MonthlyCharges', data = telcodf, ax = axes[1])

axes[1].set_title("Total Charges")



plt.show()
plt.figure(figsize = (8,8))

pie = telcodf['Churn'].value_counts()

plt.pie(pie, autopct = '%.f', labels = pie.index)

plt.title("Churn Balance", size = 14)

plt.show()
# Number of categories in each variable

catdf = telcodf.select_dtypes('object')

catdf.drop(['customerID','Churn'], axis = 1, inplace = True)

cts = catdf.nunique().sort_values()

cts.plot(kind = 'barh', width = 0.3, alpha = 0.5, color = 'C0')

plt.title("Number of Categories per Categorical Variable", size = 12)

plt.show()
catvars = catdf.columns.tolist()

def cross_matrix_heatmap(cat_set, cmap):

    fig, axes = plt.subplots(2,2, figsize = (8,8))

    for i, ax in enumerate(axes.flatten()):

        xtab = pd.crosstab(telcodf[cat_set[i]], telcodf['Churn'], normalize='index')

        sns.heatmap(xtab, annot=True, linewidths=.5, cmap = cmap,ax = ax)

        ax.set_title(cat_set[i], size = 14)

    plt.tight_layout()

    plt.show()
cross_matrix_heatmap(catvars[:4],'OrRd')
cross_matrix_heatmap(catvars[4:8],'PuBu')
cross_matrix_heatmap(catvars[8:12],'YlOrRd')
cross_matrix_heatmap(catvars[12:],'PuBu')
model_data = telcodf.copy()
model_data.drop(['gender','PhoneService','MultipleLines'], axis = 1, inplace=True)
model_data.head()
customer_ids = model_data.pop('customerID')
model_data['SeniorCitizen'] = model_data['SeniorCitizen'].map({1: 'Yes', 0:'No'})
model_data.head()
X_full = model_data.drop('Churn', axis = 1)

y_full = model_data['Churn']
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full,y_full, train_size = 0.75, random_state = 12,

                                                    stratify = y_full)
print("Proportion of classes (training):\n{}".format(y_train.value_counts() / len(y_train)))

print("Proportion of classes (testing):\n{}".format(y_test.value_counts() / len(y_test)))
cat_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype == 'object']

num_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = cat_cols + num_cols 
X_train = X_train_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train[cat_cols] = X_train[cat_cols].astype('category')

X_train[num_cols] = X_train[num_cols].astype('float64')
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import FeatureUnion, Pipeline
cat_pipeline = Pipeline(steps = [

    ('OneHot', OneHotEncoder(handle_unknown='ignore', sparse = False)),

    ('StdScaler', StandardScaler())

])
num_pipeline = Pipeline(steps = [

    ('StdScaler', StandardScaler())

])# that's all I wanna do
data_pipeline = ColumnTransformer(

    transformers=[

        ('num', num_pipeline, num_cols),

        ('cat', cat_pipeline, cat_cols)

        ])
# Transform X #

X_train_trans = data_pipeline.fit_transform(X_train)

X_test_trans = data_pipeline.transform(X_test)
# Encode Y #

y_train = y_train.map({"Yes":1, "No":0})

y_test = y_test.map({"Yes":1,"No":0})
os = SMOTE(random_state = 12, k_neighbors=20)

X_train_os,y_train_os = os.fit_sample(X_train_trans, y_train)
# Undersampling #
baselines = [LogisticRegression(),

             XGBClassifier(),

             RandomForestClassifier(),

             DecisionTreeClassifier()]
# Store performance metrics in dataframe #

perfstats = {'Algorithm':[], 'AUC Score':[], 'Accuracy':[], 'Precision':[],'Recall':[]}

model_stats = pd.DataFrame(perfstats)



def model_fitting(algorithms, X_train, X_test, y_train, y_test):

    i = 0

    # Iterate through algorithms, train, test, and evaluate

    for algorithm in algorithms:

        

        # Train #

        algo_name = type(algorithm).__name__

        model_stats.loc[i, 'Algorithm'] = algo_name

        

        algorithm.fit(X_train, y_train)

        yhat = algorithm.predict(X_test)

        

        # Evaluate #

        auc = roc_auc_score(y_test, yhat)

        model_stats.loc[i, 'AUC Score'] = auc

        

        acc = accuracy_score(y_test, yhat)

        model_stats.loc[i, 'Accuracy'] = acc

        

        prec = precision_score(y_test, yhat)

        model_stats.loc[i, 'Precision'] = prec

        

        recall = recall_score(y_test, yhat)

        model_stats.loc[i, 'Recall'] = recall

        

        print("Model: {}".format(algo_name))

        print("AUC Score: {}".format(auc))

        print("Accuracy: {}".format(acc))

        print("Precision: {}".format(prec))

        print("Recall: {}".format(recall))

        print('---------')

        i += 1
# Baseline algorithms #

model_fitting(baselines, X_train_trans, X_test_trans, y_train, y_test)
model_stats
# Logistic Regression #

params = {'penalty':['l1','l2'],

          'C':[1e-100,1e-10,1,10,100]}
log_clf = GridSearchCV(LogisticRegression(), params, cv=5, verbose=0, scoring='roc_auc')
log_clf.fit(X_train_trans, y_train)
print(log_clf.best_params_)

pd.DataFrame(log_clf.cv_results_).sort_values(by = 'rank_test_score').head(5)
# Test;

params = {'penalty':['l1','l2'], 'C':[1e-10,10]} # testing between best auc params 

log_test = GridSearchCV(LogisticRegression(), params, cv=5, verbose=0, scoring='recall')



log_test.fit(X_train_trans, y_train)

log_hat = log_test.predict(X_test_trans)

print("AUC Score: {}".format(roc_auc_score(y_test, log_hat))) 

print("Accuracy: {}".format(accuracy_score(y_test, log_hat)))

print(log_test.best_params_)
log_tuned = LogisticRegression(C = 1e-10, penalty = 'l2')

log_clf_best = log_tuned.fit(X_train_trans, y_train)
yhat_logtuned = log_clf_best.predict(X_test_trans)

print("AUC Score: {}".format(roc_auc_score(y_test, yhat_logtuned)))

print("Accuracy: {}".format(accuracy_score(y_test, yhat_logtuned)))
confusion_matrix(y_test, yhat_logtuned)
# Compute fpr, tpr, thresholds and roc auc

logit_proba = log_clf_best.predict_proba(X_test_trans)[:,1]

fpr, tpr, _ = roc_curve(y_test, logit_proba)



# Plot ROC curve

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, yhat_logtuned))

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC')

plt.legend(loc="lower right")

plt.show()