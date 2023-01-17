from scipy import stats
from scipy.stats import randint
from time import time 
import pandas as pd
import pandas_profiling as pp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
from sklearn.model_selection import cross_validate
from sklearn import metrics  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
import pickle
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 400)
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from xgboost import plot_importance
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/loandata/data_assessments_5k.csv")
df1 = pd.read_csv("../input/loandata/data_loans_5k.csv")
df = pd.merge(df1,df,how='inner',on='master_user_id')
df.shape
def quality_report(df):
 
    dtypes = df.dtypes
    nuniq = df.T.apply(lambda x: x.nunique(), axis=1)
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    quality_df = pd.concat([total, percent, nuniq, dtypes], axis=1, keys=['Total', 'Percent','Nunique', 'Dtype'])
    display(quality_df)
quality_report(df)
Report = pp.ProfileReport(df,minimal=True)
df = df.drop(["assessment_rules",
"bureau",
"cc_kotak_bank_count",
"cibil_salary_date_reported",
"cibil_salary_estimate_type",
"created_at",
"crif_account_purchased_and_restructured_count_52",
"crif_account_purchased_and_restructured_count_52_260",
"crif_account_purchased_and_settled_count_52",
"crif_account_purchased_and_settled_count_52_260",
"crif_account_purchased_count_52",
"crif_account_purchased_count_52_260",
"crif_account_sold_count_52",
"crif_account_sold_count_52_260",
"crif_account_summary",
"crif_accounts_count",
"crif_inquiry_count_9",
"crif_issue_date",
"crif_no_suit_filed_count_52",
"crif_no_suit_filed_count_52_260",
"crif_obligations",
"crif_post_wo_settled_count_52",
"crif_post_wo_settled_count_52_260",
"crif_restructured_due_to_natural_calamity_count_52",
"crif_restructured_due_to_natural_calamity_count_52_260",
"crif_restructured_loan_count_52",
"crif_restructured_loan_count_52_260",
"crif_restructured_loan_govt_mandated_count_52",
"crif_restructured_loan_govt_mandated_count_52_260",
"crif_score",
"crif_settled_count_52",
"crif_settled_count_52_260",
"crif_suit_filed_count_52",
"crif_suit_filed_count_52_260",
"crif_suit_filed_wilful_default_count_52",
"crif_suit_filed_wilful_default_count_52_260",
"crif_sum_overdue_amount_52",
"crif_sum_overdue_amount_52_260",
"crif_sum_overdue_amount_cc_52",
"crif_sum_overdue_amount_cc_52_260",
"crif_sum_overdue_amount_non_cc_52",
"crif_sum_overdue_amount_non_cc_52_260",
"crif_unique_reference_number",
"crif_wilful_default_count_52",
"crif_wilful_default_count_52_260",
"crif_written_off_and_account_sold_count_52",
"crif_written_off_and_account_sold_count_52_260",
"crif_written_off_count_52",
"crif_written_off_count_52_260",
"employer",
"employer_category",
"employer_id",
"is_approved_flexi",
"is_approved_premium",
"network_contacts_in_cibil_score_average",
"network_contacts_in_cibil_score_median",
"network_contacts_in_cibil_score_negative_count",
"network_contacts_in_cibil_wilful_defaults_count",
"network_contacts_in_cibil_write_offs_count",
"network_contacts_in_declined_users_count",
"network_contacts_in_disbursed_users_count",
"network_contacts_in_dpd_30_users_count",
"network_contacts_in_dpd_60_users_count",
"network_contacts_in_dpd_90_users_count",
"network_contacts_in_ps_users_count",
"network_contacts_in_rejected_users_count",
"network_contacts_in_users_count",
"network_contacts_out_cibil_score_average",
"network_contacts_out_cibil_score_median",
"network_contacts_out_cibil_score_negative_count",
"network_contacts_out_cibil_wilful_defaults_count","disbursed_at","perfios_salary_estimate_type","salary_estimate_type"], axis = 1) 
df.sort_values("updated_at", ascending = False,inplace=True) 
df.drop_duplicates(subset ="master_user_id", 
                     keep = "first", inplace = True)
df.drop('updated_at',inplace = True,axis=1)
df.select_dtypes('object')
df.is_non_starter = np.where(df.is_non_starter=='True', 1,df.is_non_starter)
df.is_non_starter = np.where(df.is_non_starter=='False', 0,df.is_non_starter)

df.product_type = np.where(df.product_type=='flexi', 1,df.product_type)
df.product_type = np.where(df.product_type=='premium', 0,df.product_type)
df = df.drop(['city','reason_premium','reason_flexi'],axis=1)
df = df.fillna(0)
df['Target'] = np.where(df.max_dpd >= 30, 1, 0)
plt.figure(figsize=(6,3))
sns.countplot(x='Target',data=df)
plt.show()
 
# Checking the event rate : event is when claim is made
df['Target'].value_counts()
X = df.iloc[:,0:382]
y= df.iloc[:,382:383]
seed=63 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape,y_train.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components = 30)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
param_grid = {'n_estimators': [ 30, 35, 25],
                    'learning_rate': [ 0.1, 0.15,0.2],
                    'gamma':  [0.20,0.10, 0.15],
                    'max_delta_step': [24, 26, 22],
                    'max_depth':[4, 3, 5],
             'min_child_weight': [1, 2, 3, 4]}       

clf = RandomizedSearchCV(xgb, n_iter = 50, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2)
clf
xgb = XGBClassifier(n_estimators=30,
 max_depth= 5,
 gamma = 0.2,
min_child_weight = 5)
model = xgb.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred[0].value_counts()
print('AUPRC = {}'.format(average_precision_score(y_test, y_pred.iloc[:,0])))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(xgb, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb, height = 1, color = colours, grid = False, \
                     show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
df = df.drop([
"network_contacts_out_cibil_write_offs_count",
"network_contacts_out_declined_users_count",
"network_contacts_out_disbursed_users_count",
"network_contacts_out_dpd_30_users_count",
"network_contacts_out_dpd_60_users_count",
"network_contacts_out_dpd_90_users_count",
"network_contacts_out_ps_users_count",
"network_contacts_out_rejected_users_count",
"network_contacts_out_users_count",
"perfios_salary_day_of_month",
"postal_code",
"rule_engine_output",
"salary_day_of_month",
"scheme",
"state",
"version"],axis=1)
X = df.iloc[:,0:366]
y= df.iloc[:,366:367]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
param_grid = {'n_estimators': [ 30, 35, 25],
                    'learning_rate': [ 0.1, 0.15,0.2],
                    'gamma':  [0.20,0.10, 0.15],
                    'max_delta_step': [24, 26, 22],
                    'max_depth':[4, 3, 5],
             'min_child_weight': [1, 2, 3, 4]}       

clf = RandomizedSearchCV(xgb, n_iter = 50, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2)

clf
xgb = XGBClassifier(n_estimators=35,
 max_depth= 3,
 max_delta_step = 26,
 learning_rate = 0.15,
 gamma = 0.1,
min_child_weight = 3)
model = xgb.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred[0].value_counts()
print('AUPRC = {}'.format(average_precision_score(y_test, y_pred.iloc[:,0])))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(xgb, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split as tts
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.datasets import load_spam

# Load the dataset and split into train/test splits
#X, y = load_spam()

#X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

# Create the visualizer, fit, score, and show it
viz = PrecisionRecallCurve(RidgeClassifier())
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()