import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
import math
load_data = pd.read_csv('../input/lending-club-loan-data/loan.csv', low_memory=False)
load_data.head()
load_data.shape
load_data.isnull().sum()[load_data.columns[load_data.isnull().mean()<0.8]]
load_data = load_data[load_data.columns[load_data.isnull().mean()<0.8]]
load_data = load_data[~load_data.earliest_cr_line.isnull()]
load_data = load_data.fillna(0)
# Remove outliers fro annual_income

date_issued = pd.to_datetime(load_data['issue_d'])
date_last_payment = pd.to_datetime(load_data['last_pymnt_d'])
load_data['issue_year']=date_issued.dt.year
load_data['issue_month']=date_issued.dt.month




bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", 
            "Late (16-30 days)", "Late (31-120 days)"]


load_data['loan_condition'] = np.nan

def loan_condition(status):
    if status in bad_loan:
        return 'Bad Loan'
    elif status == "Current":
        return 'Current'
    else:
        return 'Good Loan'
    
    
loan_status = load_data['loan_status'].apply(loan_condition)
load_data['loan_condition']=loan_status
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#0B6623","#E1AD01", "#D72626" ]
labels ="Good Loans","Current", "Bad Loans"

plt.suptitle('Information on Loan Conditions', fontsize=20)

load_data["loan_condition"].value_counts().plot.pie(explode=[0,0.1, 0.2], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)


palette = ["#E1AD01","#0B6623", "#D72626" ]

sns.countplot(x="issue_year", hue="loan_condition", data=load_data, palette=palette)
def countplot_category_against_loan_condition(column):
    palette = ["#E1AD01","#0B6623", "#D72626" ]
    order = sorted(load_data[column].unique())
    sns.countplot(x=column, hue="loan_condition", data=load_data, palette=palette, order=order)
    
countplot_category_against_loan_condition('grade')    
    
color = ["#D72626" ,"#E1AD01","#0B6623" ]
ax = load_data.groupby(['issue_year','loan_condition']).int_rate.mean().unstack().plot(title="Interest Rate by Load Status", color=color)
ax.set_ylabel('Interest Rate (%)', fontsize=12)
ax.set_xlabel('Year', fontsize=12)

house_hold_income = pd.read_csv("../input/purchase-power-index/HouseHold_Income_by_State_by_year.csv")
house_hold_income= house_hold_income.drop(["Unnamed: 0","State"], axis=1)
house_hold_income.head()
house_hold_income = house_hold_income.set_index('State_abbr').stack().reset_index().rename(columns ={'level_1':'year',0:'median_household_income'})
house_hold_income['year'] = house_hold_income['year'].astype(int)
house_hold_income.head()
load_data = pd.merge(house_hold_income,load_data,right_on=['addr_state','issue_year'],left_on=['State_abbr','year'],how='right')
load_data = load_data.drop(["State_abbr","year"], axis=1)
ppi = pd.read_csv("../input/purchase-power-index/ppi.csv")
ppi.head()

ppi = ppi.drop("GeoName",axis=1)
ppi = ppi.set_index('State_abbr').stack().reset_index().rename(columns = {'level_1':'year',0:'ppi'})
ppi['year'] = ppi['year'].astype(int)
ppi.head()

load_data = pd.merge(ppi,load_data,right_on=['addr_state','issue_year'],left_on=['State_abbr','year'],how='right')
load_data = load_data.drop(["State_abbr","year"], axis=1)
load_data['relative_income_index'] = (load_data['annual_inc']/load_data['ppi']*100) - load_data['median_household_income']
load_data = load_data.drop(["ppi","median_household_income"], axis=1)


def sigmoid(x):
    if x < -709:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-x))


months_into_loan = 12*(date_last_payment.dt.year-load_data['issue_year']) + (date_last_payment.dt.month-load_data['issue_month'])
payment_index = (months_into_loan / pd.to_numeric(load_data['term'].str.extract('(\d+)')[0]))/(load_data['out_prncp']/load_data['funded_amnt'])
load_data['payment_index'] = payment_index.fillna(0)
load_data['payment_index'] = load_data['payment_index'].apply(lambda x: sigmoid(x))


load_data['emp_length_integer'] = load_data.emp_length.replace({'< 1 year':0,     
                                                      '1 year':1,
                                                      '2 years':2,
                                                      '3 years':3,
                                                      '4 years':4,
                                                      '5 years':5,
                                                      '6 years':6,
                                                      '7 years':7,
                                                      '8 years':8,
                                                      '9 years':9,
                                                      '10+ years':10,
                                                      None:-1})
palette = ["#E1AD01","#0B6623", "#D72626" ]
countplot_category_against_loan_condition('emp_length_integer')
"""
NUM_CLUSTERS = range(3, 20)
model = [cluster.KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1) for i in NUM_CLUSTERS]
score = [model[i].fit(X).score(X) for i in range(len(model))]
plt.plot(NUM_CLUSTERS,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
"""


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(load_data.emp_title.replace({0:"Missing"}))
NUM_CLUSTERS = 10
model = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
load_data['emp_title_tfidf'] = model.predict(X)
sns.countplot(x="emp_title_tfidf",data=load_data)
load_data['emp_title_tfidf'] = load_data['emp_title_tfidf'].astype('object')
def distplot_numberical_feature_across_loan_condition(column):
    plt.figure()
    remove_outliers = load_data[~((load_data[column]-load_data[column].mean()).abs() > 3*load_data[column].std())]
    palette = ["#0B6623", "#D72626" , "#E1AD01"]
    loan_condition = ["Good Loan","Bad Loan","Current"]
    for i in range(len(loan_condition)):
        # Subset to the airline

        subset = remove_outliers[remove_outliers['loan_condition'] == loan_condition[i]]
        # Draw the density plot
        sns.distplot(subset[column], hist = False, kde = True,
                     kde_kws = {'linewidth': 1},
                     label = loan_condition[i],
                     color = palette[i])

    # Plot formatting
    plt.legend(prop={'size': 16}, title = 'Loan Condition')
    plt.title('Density Plot of {}'.format(column))
    plt.ylabel('Density')

    

for numeric_features in load_data.columns[load_data.dtypes!='object'].tolist():
    distplot_numberical_feature_across_loan_condition(numeric_features)

cat_columns = ["term",
                "grade",
                "home_ownership",
                "verification_status",
                "purpose",
                "application_type",
                "hardship_flag",
                "debt_settlement_flag",
                "pymnt_plan",
                "disbursement_method"]
for categorical_features in cat_columns:
    plt.figure()
    palette = ["#0B6623", "#D72626","#E1AD01" ]
    sns.countplot(x=categorical_features, hue="loan_condition", data=load_data, palette=palette)


date_columns = ["issue_d",
                "issue_year",
                "issue_month",
               "earliest_cr_line",
               "last_pymnt_d",
               "next_pymnt_d",
               "last_credit_pull_d"]
onehot_columns = ["term",
                    "grade",
                    "home_ownership",
                    "verification_status",
                    "purpose",
                    "application_type"]
string_columns =["emp_title",
                    "emp_length",
                    "title",
                    "zip_code",
                    "addr_state",
                    "loan_condition"]

engineered_features =["emp_length_integer",
                     "emp_title_tfidf"]

remove_columns = ["title",
                  "zip_code",
                  "loan_status",
                 "policy_code",
                 "acc_now_delinq"]
remove_columns_time_skewed = [
                        "total_rec_prncp",
                        "total_pymnt",
                        "total_rec_int",
                        "total_pymnt_inv",
                        "max_bal_bc",
                        "last_pymnt_amnt",
                        "out_prncp",
                        "out_prncp_inv",
                        "recoveries",
                        "collection_recovery_fee",
                        "total_rec_late_fee",
                        "inq_last_6mths",
                        "mths_since_rcnt_il",
                        "mths_since_recent_bc",
                        "mths_since_recent_bc_dlq",
                        "mths_since_recent_inq",
                        "mths_since_recent_revol_delinq"
                 ]


remove_one_hot_column = ["hardship_flag",
                        "debt_settlement_flag",
                        "pymnt_plan",
                        "disbursement_method"]




onehot_loan_df = pd.get_dummies(load_data[list(set(onehot_columns)-set(remove_one_hot_column))])
print(onehot_loan_df.shape)
features_df = onehot_loan_df.join(load_data.drop((date_columns+
                                                  onehot_columns+
                                                  string_columns+
                                                  remove_columns+
                                                 remove_columns_time_skewed),axis=1))
features_df = features_df.drop(features_df.columns[features_df.dtypes=='object'].tolist(), axis=1)
features_df['loan_status']=load_data['loan_condition']
features_df.head()
current_loans = features_df[features_df['loan_status']=="Current"]
current_loan_status = features_df[features_df['loan_status']=="Current"]['loan_status']
past_loans = features_df[features_df['loan_status']!="Current"]
past_loan_status = features_df[features_df['loan_status']!="Current"]['loan_status']
past_loan_status = past_loan_status.replace({'Good Loan':1,
                         'Bad Loan':0})
past_loans = past_loans.drop("loan_status", axis=1)
current_loans = current_loans.drop("loan_status", axis=1)
X_train, X_test, y_train, y_test = train_test_split(past_loans, 
                                                    past_loan_status, 
                                                    test_size=0.20, 
                                                    random_state=42,
                                                    stratify=past_loan_status)
"""
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
"""
gb = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1)
gb.fit(X_train,y_train)
y_gb_pred = gb.predict(X_test)
print(classification_report(y_test,y_gb_pred, target_names=["Bad Loan","Good Loan"]))
"""
gb_params = {"n_estimators":np.arange(40,80,10),"learning_rate":np.arange(0.05,0.3,0.05)}
grid_gb = GridSearchCV(GradientBoostingClassifier(),gb_params)
grid_gb.fit(X_train,y_train)
print(classification_report(y_test, grid_gb.predict(X_test), target_names=["Lost Sales","Funded Loans"]))
print(grid_gb.best_params_)

"""

labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(gb.predict(X_test)))
print(cm)
scaler = StandardScaler()
X_train_preprocessed_std = scaler.fit_transform(X_train)
X_test_preprocessed_std = scaler.fit_transform(X_test)
lr_std = LogisticRegression(class_weight="balanced")
lr_std.fit(X_train_preprocessed_std,y_train)
y_lr_pred = lr_std.predict(X_test_preprocessed_std)
print(classification_report(y_test,y_lr_pred, target_names=["Bad Loan","Good Loan"]))
lr = LogisticRegression(max_iter=400, class_weight="balanced")
lr.fit(X_train,y_train)
y_lr_pred = lr.predict(X_test)
print(classification_report(y_test,y_lr_pred, target_names=["Bad Loan","Good Loan"]))
labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(lr.predict(X_test)))
print(cm)
XGB_model = XGBClassifier()
XGB_model.fit(X_train,y_train)


y_xgb_pred = XGB_model.predict(X_test)
print(classification_report(y_test,y_xgb_pred, target_names=["Bad Loan", "Good Loan"]))
plot_importance(XGB_model, max_num_features= 15)

def add_roc_curve(y_pred, model_name, y_test=y_test, plt=plt):
    fpr, tpr, thres = roc_curve(y_test, y_pred)
    auc = round(roc_auc_score(y_test, y_pred),2)
    plt.plot(1-fpr,tpr,label="{model_name}, auc={auc}".format(model_name=model_name,auc=auc))
    plt.legend(loc=0)
    return(plt)
gb_plt = add_roc_curve(pd.DataFrame(gb.predict_proba(X_test))[1], y_test= y_test, model_name="Gradient Boost")
xgb_plt = add_roc_curve(pd.DataFrame(XGB_model.predict_proba(X_test))[1], y_test= y_test, model_name="XGBoost")
lr_plt = add_roc_curve(pd.DataFrame(lr.predict_proba(X_test))[1], y_test= y_test, model_name="Logistic")
lr_std_plt = add_roc_curve(pd.DataFrame(lr_std.predict_proba(X_test))[1], y_test= y_test, model_name="Logistic_std")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(XGB_model.predict(X_test)))
print(cm)
cm = confusion_matrix(y_true = np.array(y_train),
                      y_pred = pd.DataFrame(XGB_model.predict(X_train)))
print(cm)
y_train.value_counts()
pd.Series(XGB_model.predict(current_loans)).value_counts()
palette = ["#0B6623", "#D72626" ]

sns.countplot(x="issue_year", hue="loan_condition", data=load_data[load_data['loan_condition']!="Current"], palette=palette)
palette = ["#D72626" , "#0B6623"]
current_loans_raw_data = load_data[load_data['loan_condition']=="Current"]
#keeping threshold low; as model needs more tweeks
current_loans_raw_data['predicted_loan_condition'] = (pd.DataFrame(XGB_model.predict_proba(current_loans))[1]>0.12)
sns.countplot(x="issue_year", hue="predicted_loan_condition", data=current_loans_raw_data, palette=palette)