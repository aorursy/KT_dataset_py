# Import necessary libraries

import numpy as np

import pandas as pd

import re

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc, f1_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

import joblib



%matplotlib inline
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.shape
print(df.info())
def print_df_unique_vals(df):

    for col in df.columns:

        unique_vals = df[col].unique()

        if len(unique_vals) < 10:

            print("Unique values for column {}: {}".format(col, unique_vals))

        else:

            if is_string_dtype(df[col]):

                print("column {} has values string type".format(col))

            elif is_numeric_dtype(df[col]):

                print("column {} is numerical".format(col))
print_df_unique_vals(df)
dec_reg_exp = r'^[+-]{0,1}((\d*\.)|\d*)\d+$'

abnormal_total_charges = df[~df.TotalCharges.str.contains(dec_reg_exp)]

abnormal_total_charges
df = df[df.TotalCharges.str.contains(dec_reg_exp)]

df['TotalCharges'] = df['TotalCharges'].astype(float)
def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    



display_missing(df)
# Summary statistics:

cont_features = ["tenure", "MonthlyCharges", "TotalCharges"]

df_num = df[cont_features]

df_num.describe()
Q1 =df_num.quantile(0.25)

Q3 = df_num.quantile(0.75)

IQR = Q3 - Q1

IQR

((df_num < (Q1 - 1.5 * IQR)) |(df_num > (Q3 + 1.5 * IQR))).any()
churn = df['Churn'] == 'Yes'
def plot_dist_num_cols_target(df, cont_features, target, target_label):

    fig, axs = plt.subplots(ncols=1, nrows=len(cont_features), figsize=(20, 20))

    plt.subplots_adjust(right=1.5)

    for i, feature in enumerate(cont_features):    

        sns.distplot(df[~target][feature], label='Not {}'.format(target_label), hist=True, color='#e74c3c', ax=axs[i])

        sns.distplot(df[target][feature], label='{}'.format(target_label), hist=True, color='#2ecc71', ax=axs[i])

        

        axs[i].set_xlabel('')

        axs[i].set_xlabel('')

        

        for j in range(len(cont_features)):        

            axs[j].tick_params(axis='x', labelsize=15)

            axs[j].tick_params(axis='y', labelsize=15)



        axs[i].legend(loc='upper right', prop={'size': 10})

        axs[i].legend(loc='upper right', prop={'size': 10})

        axs[i].set_title('Distribution of {} in {}'.format(target_label, feature), size=20, y=1.05)



    plt.tight_layout(pad=5)

    plt.savefig('numerical_attributes.png')

    plt.show()
plot_dist_num_cols_target(df, cont_features, churn, 'Churned')
def plot_dist_num_cols_target(df, cont_features):

    fig, axs = plt.subplots(ncols=1, nrows=len(cont_features), figsize=(20, 20))

    plt.subplots_adjust(right=1.5)

    for i, feature in enumerate(cont_features):    

        sns.distplot(df[feature], label='{}'.format(feature), hist=False, color='#e74c3c', ax=axs[i])

        

        axs[i].set_xlabel('')

        axs[i].set_xlabel('')

        

        for j in range(len(cont_features)):        

            axs[j].tick_params(axis='x', labelsize=15)

            axs[j].tick_params(axis='y', labelsize=15)



        axs[i].legend(loc='upper right', prop={'size': 10})

        axs[i].legend(loc='upper right', prop={'size': 10})

        axs[i].set_title('Distribution of {} feature'.format(feature), size=20, y=1.05)



    plt.tight_layout(pad=5)

    plt.savefig('distr_plots_numerical_attributes.png')

    plt.show()
plot_dist_num_cols_target(df, cont_features)
cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 

                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',

               'Contract', 'PaperlessBilling', 'PaymentMethod']
def plot_bar_cat_cols(df, cat_features, target_label, hue):

    fig, axs = plt.subplots(ncols=2, nrows=8, figsize=(50, 50))

    plt.subplots_adjust(right=1.5, top=1.25)

    for i, feature in enumerate(cat_features, 1):    

        plt.subplot(8, 2, i)

        sns.countplot(x=feature, hue=hue, data=df)

        plt.xlabel('{}'.format(feature), size=30, labelpad=15)

        plt.ylabel('Customer Count', size=30, labelpad=15)    

        plt.tick_params(axis='x', labelsize=30)

        plt.tick_params(axis='y', labelsize=30)



        plt.legend(['Not {}'.format(target_label), '{}'.format(target_label)], loc='upper right', prop={'size': 25})

        plt.title('Count of {} in {} Feature'.format(target_label, feature), size=40, y=1.05)



    plt.tight_layout(h_pad=5)

    plt.savefig('cat_attributes_bar.png')

    plt.show()
plot_bar_cat_cols(df, cat_features, 'Churned', 'Churn')
sns.countplot(x="Churn", data=df)
df['MonthlyCharges'] = pd.qcut(df['MonthlyCharges'], 10)



fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='MonthlyCharges', hue='Churn', data=df)



plt.xlabel('MonthlyCharges', size=15, labelpad=20)

plt.ylabel('Customer count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not churned', 'churned'], loc='upper right', prop={'size': 15})

plt.title('Churn Counts in {} Feature'.format('MonthlyCharges'), size=15, y=1.05)

plt.savefig('churned_v_not_churned.png')

plt.show()
non_numeric_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 

                       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'Churn']



for feature in non_numeric_features:        

    df[feature] = LabelEncoder().fit_transform(df[feature])
df.head()
print(df.info())
cat_features = ['MultipleLines', 'InternetService', 'OnlineSecurity',

                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 

                'PaymentMethod']

encoded_features = []



for feature in cat_features:

    encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

    n = df[feature].nunique()

    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

    encoded_df = pd.DataFrame(encoded_feat, columns=cols)

    encoded_df.index = df.index

    encoded_features.append(encoded_df)
len(encoded_features)
df = pd.concat([df, *encoded_features], axis=1)
df.columns
df2 = df.copy()
drop_cols = ['customerID', 'MultipleLines', 'InternetService', 'OnlineSecurity',

            'OnlineBackup', 'DeviceProtection', 'TechSupport', 

            'StreamingTV', 'StreamingMovies', 'Contract', 

            'PaymentMethod']



df.drop(columns=drop_cols, inplace=True)
X = df.drop(columns=['Churn']).values

y = df["Churn"].values



seed = 42

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y, random_state=seed)

print('X_train shape: {}'.format(x_train.shape))

print('X_test shape: {}'.format(x_test.shape))
skf = StratifiedKFold(n_splits=5)

fprs, tprs, scores, val_auc_scores = [], [], [], []

for train_index, valid_index in skf.split(x_train, y_train):

    x_pseudo_train, x_pseudo_valid = x_train[train_index], x_train[valid_index]

    y_pseudo_train, y_pseudo_valid = y_train[train_index], y_train[valid_index]

    ss = StandardScaler()

    x_pseudo_train_scaled = ss.fit_transform(x_pseudo_train)

    x_pseudo_valid_scaled = ss.transform(x_pseudo_valid)

    lr = LogisticRegression()  # Using default parameters.

    lr.fit(x_pseudo_train_scaled, y_pseudo_train)

    y_pred_train_probs = lr.predict_proba(x_pseudo_train_scaled)[:, 1]

    y_pred_valid_probs = lr.predict_proba(x_pseudo_valid_scaled)[:, 1]

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_pseudo_train, 

                                                 y_pred_train_probs)

    trn_auc_score = auc(trn_fpr, trn_tpr)

    val_fpr, val_tpr, val_thresholds = roc_curve(y_pseudo_valid, 

                                                 y_pred_valid_probs)

    val_auc_score = auc(val_fpr, val_tpr)

    val_auc_scores.append(val_auc_score)

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)
average_val_auc = np.mean(val_auc_scores)

print("Average Validation AUC score: {}".format(average_val_auc))
def plot_roc_curve(fprs, tprs):

    

    tprs_interp = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    f, ax = plt.subplots(figsize=(15, 15))

    

    # Plotting ROC for each fold and computing AUC scores

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):

        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

        tprs_interp[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))

        

    # Plotting ROC for random guessing

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    

    mean_tpr = np.mean(tprs_interp, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    

    # Plotting the mean ROC

    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)

    

    # Plotting the standard deviation around the mean ROC Curve

    std_tpr = np.std(tprs_interp, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

    

    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)

    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)

    ax.tick_params(axis='x', labelsize=15)

    ax.tick_params(axis='y', labelsize=15)

    ax.set_xlim([-0.05, 1.05])

    ax.set_ylim([-0.05, 1.05])



    ax.set_title('ROC Curves of Folds', size=20, y=1.02)

    ax.legend(loc='lower right', prop={'size': 13})

    plt.savefig('roc_curve.png')

    plt.show()



plot_roc_curve(fprs, tprs)
ss = StandardScaler()

x_train_scaled = ss.fit_transform(x_train)

x_test_scaled = ss.transform(x_test)



# Applying logistic regression classifier

lr = LogisticRegression()  # Using default parameters.

lr.fit(x_train_scaled, y_train)  # training the model with X_train, y_train



# Generate Confusion Matrix

y_pred = lr.predict(x_test_scaled)                # Make predictions on test set

y_pred = pd.Series(y_pred)

y_test = pd.Series(y_test)

pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True'], margins=True)
def print_summary_stats(y_pred, y_test):

    TP = sum((y_pred == y_test) & (y_pred == 1))            # No. of True Positives

    FN = sum((y_pred != y_test) & (y_pred == 0))            # No. of False Negatives

    P = TP + FN                                             # Total No. of Positives

    TN = sum((y_pred == y_test) & (y_pred == 0))            # No. of True Negatives

    FP = sum((y_pred != y_test) & (y_pred == 1))

    N = TN + FP



    print("Sensitivity: {}".format(TP / P))                 # True Positive / Positive (Sensitivity)

    print("Specificity: {}".format(TN / N))                 # TN / N (Specificity)

    print("Precision: {}".format(TP / (TP + FP)))           # (TP/ (TP+FP)) Precision

    print("True Negative Rate: {}".format(TN / (TN + FN)))  # (TN / (TN+FN))

    print("Overall Accuracy: {}".format(sum(y_pred == y_test)/len(y_test)))
print_summary_stats(y_pred, y_test)
MODEL_DIR = '/kaggle/working/'

joblib.dump(ss, MODEL_DIR + 'scaler.pkl')

joblib.dump(lr, MODEL_DIR + 'lr_model.pkl')
feats = df.drop(columns=['Churn'])

num_features = len(feats.columns)



coef_info = []

for i in range(num_features):

    coef = lr.coef_[0][i]

    coef_info.append((feats.columns[i], coef))
sorted(coef_info, key=lambda x: abs(x[1]), reverse=True)[:10]
param_grid = { 

    'n_estimators': [200, 500, 1000],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
rfc = RandomForestClassifier(random_state=seed)

cv_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, cv= 5, n_jobs=-1)

cv_rfc.fit(x_train_scaled, y_train)
cv_rfc.best_params_
skf = StratifiedKFold(n_splits=5)

fprs, tprs, scores, val_auc_scores = [], [], [], []

for train_index, valid_index in skf.split(x_train, y_train):

    x_pseudo_train, x_pseudo_valid = x_train[train_index], x_train[valid_index]

    y_pseudo_train, y_pseudo_valid = y_train[train_index], y_train[valid_index]

    ss = StandardScaler()

    x_pseudo_train_scaled = ss.fit_transform(x_pseudo_train)

    x_pseudo_valid_scaled = ss.transform(x_pseudo_valid)

    rf = RandomForestClassifier(**cv_rfc.best_params_)  # Using default parameters.

    rf.fit(x_pseudo_train_scaled, y_pseudo_train)

    y_pred_train_probs = rf.predict_proba(x_pseudo_train_scaled)[:, 1]

    y_pred_valid_probs = rf.predict_proba(x_pseudo_valid_scaled)[:, 1]

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_pseudo_train, 

                                                 y_pred_train_probs)

    trn_auc_score = auc(trn_fpr, trn_tpr)

    val_fpr, val_tpr, val_thresholds = roc_curve(y_pseudo_valid, 

                                                 y_pred_valid_probs)

    val_auc_score = auc(val_fpr, val_tpr)

    val_auc_scores.append(val_auc_score)

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)
average_val_auc = np.mean(val_auc_scores)

print("Average Validation AUC score: {}".format(average_val_auc))
plot_roc_curve(fprs, tprs)
# Applying random forest classifier

rf = RandomForestClassifier(**cv_rfc.best_params_)  # Using CV grid search

rf.fit(x_train_scaled, y_train)  # training the model with X_train, y_train



# Generate Confusion Matrix

y_pred = rf.predict(x_test_scaled)                # Make predictions on test set

y_pred = pd.Series(y_pred)

y_test = pd.Series(y_test)

pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True'], margins=True)
print_summary_stats(y_pred, y_test)
joblib.dump(rf, MODEL_DIR + 'rf_model.pkl')
from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier(**cv_rfc.best_params_)

clf2 = LogisticRegression()

eclf1 = VotingClassifier(estimators=[('rf', clf1), ('lr', clf2)], voting='soft')

eclf1.fit(x_train_scaled, y_train)

predictions = eclf1.predict(x_test_scaled)
print_summary_stats(predictions, y_test)