import pandas as pd

import numpy as np



import random

np.random.seed(123)



# Reading the data

#test = pd.read_csv('../input/blight-ticket/blight_test.csv', engine='python')

train = pd.read_csv('../input/blight-ticket/blight_train.csv', engine='python')



latlons = pd.read_csv('../input/blight-ticket/latlons.csv', engine='python')

addresses = pd.read_csv('../input/blight-ticket/addresses.csv', engine='python')



train.head()
latlons.head()
addresses.head()


# Remove compliances that are NaNs (not 1 or 0)

train = train[(train['compliance'] == 1)|(train['compliance'] == 0)]

train.reset_index(drop=True, inplace=True)


# Extract a new feature: the gap between the date when the ticket was issued and when the court hearing is supposed to happen:

train['day_gap']= (pd.to_datetime(train['hearing_date']) - pd.to_datetime(train['ticket_issued_date'])).dt.days
train.drop(['agency_name', 'inspector_name', 'violator_name',

       'violation_street_number', 'violation_street_name',

       'violation_zip_code', 'mailing_address_str_number',

       'mailing_address_str_name', 'city', 'state', 'zip_code',

       'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',

       'violation_code', 'violation_description',

       'admin_fee', 'state_fee',

       'clean_up_cost', 'payment_amount', 'balance_due',

       'payment_date', 'payment_status', 'collection_status',

       'grafitti_status', 'compliance_detail', 'fine_amount'], axis=1, inplace=True)
train['disposition'].value_counts()
train = train.loc[train['disposition'] != 'Responsible (Fine Waived) by Deter']
train.isnull().sum()
train.dropna(inplace=True)
# Split into training features and target values

X0 = train.loc[:, ['ticket_id', 'disposition', 'late_fee', 'discount_amount',

       'judgment_amount', 'day_gap']]



y0 = train['compliance']
# Merge addresses and latitude and longitute with the training features

X1 = pd.merge(X0, addresses, how='left', on='ticket_id')

X2 = pd.merge(X1, latlons, how='left', on='address')

X2.drop('address', axis=1, inplace=True)
# Convert strings in 'disposition' to dummy values (0 and 1) # I could also try LabelEncoder()

columns_for_dummies = X2[['disposition']]

extra_dummies = pd.get_dummies(columns_for_dummies)



# add the new dummy features 

X3 = pd.merge(X2, extra_dummies, how='left', left_index=True, right_index=True)



# Get the final training set that only includes numeric types

X4 = X3.drop(['ticket_id', 'disposition'], axis=1)



# forward fill any other missing values

X4.ffill(inplace=True)



X4.head()
from sklearn.model_selection import train_test_split



# Split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X4, y0, random_state=0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# then, apply the scaler to the test data:

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

logis = LogisticRegression().fit(X_train_scaled, y_train)
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score

# (1) obtain probabilities of observations belonging to class 1

X_test_sc_probs = logis.predict_proba(X_test_scaled)[:, 1]



# (2) use vals function to convert the probabilities to predictions based on a give threshold

# If threshold is 0.5, then all observations with probability of 0.5 and higher will be assigned to class 1. Otherwise, class 0.

def vals (probabilities, threshold=0.5):

    class_preds = np.where(probabilities >= threshold, 1, 0)

    return class_preds



X_test_sc_class = vals(X_test_sc_probs, 0.5)



# (3) use the predictions and actual values to construct a confustion matrix and output the classification report.

print(confusion_matrix(y_test, X_test_sc_class))

print(classification_report(y_test, X_test_sc_class))
pd.DataFrame(y_test).value_counts()
import matplotlib.pyplot as plt



X_test_sc_probs = logis.predict_proba(X_test_scaled)[:, 1]

recalls = []

precisions = []

def vals_2 (probabilities, threshold=0.5):

    class_preds = np.where(probabilities >= threshold, 1, 0)

    rscore = recall_score(y_test, class_preds)

    pscore = precision_score(y_test, class_preds)

    

    recalls.append(rscore)

    precisions.append(pscore)



for i in np.arange(0.0, 1.0, 0.05):

    vals_2(X_test_sc_probs, i)

    

plt.plot(np.arange(0.0, 1.0, 0.05), recalls, label="Recall")

plt.plot(np.arange(0.0, 1.0, 0.05), precisions, label="Precision")

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.xlabel('Thresholds')

plt.ylabel('Recall / Precisions')

plt.legend()
from sklearn.metrics import precision_recall_curve



precision, recall, thresholds = precision_recall_curve(y_test, X_test_sc_probs)



closest_zero = np.argmin(np.abs(thresholds-0.5))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure()

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.axes().set_aspect('equal')

plt.show()
from sklearn.metrics import roc_curve, auc



fpr_lr, tpr_lr, tts = roc_curve(y_test, X_test_sc_probs)



roc_auc_lr = auc(fpr_lr, tpr_lr)



plt.figure()

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_lr, tpr_lr, lw=3, label=' (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=10)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()
from sklearn.model_selection import GridSearchCV

# list parameters and values we are interested in. 

grid_values = {'C': [1, 10, 100], 'penalty':['l1', 'l2']}



#GridSearch will go over each combination looking for the highest recall score, while using k-fold Cross-Validation on 5 folds

logis_cv = LogisticRegression(max_iter=1000, solver='liblinear' )

grid_lr = GridSearchCV(logis_cv, param_grid = grid_values, cv=5, scoring='recall')

grid_lr.fit(X_train, y_train)
grid_lr.best_params_
nb_lr = LogisticRegression(C=1, penalty='l1', max_iter=1000, solver='liblinear').fit(X_train_scaled, y_train)

nb_lr_probs = nb_lr.predict_proba(X_test_scaled)[:, 1]



def vals (probabilities, threshold=0.5):

    class_preds = np.where(probabilities >= threshold, 1, 0)

    return class_preds



X_best_test_sc_class = vals(nb_lr_probs, 0.5)



# (3) use the predictions and actual values to construct a confustion matrix and output the classification report.

print(confusion_matrix(y_test, X_best_test_sc_class))

print(classification_report(y_test, X_best_test_sc_class))
precision, recall, thresholds = precision_recall_curve(y_test, nb_lr_probs)



closest_zero = np.argmin(np.abs(thresholds-0.5))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure()

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.axes().set_aspect('equal')

plt.show()
fpr_lr, tpr_lr, tts = roc_curve(y_test, nb_lr_probs)



roc_auc_lr = auc(fpr_lr, tpr_lr)



plt.figure()

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_lr, tpr_lr, lw=3, label=' (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=10)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()
pd.DataFrame([X4.columns.values, nb_lr.coef_.T]).T
from sklearn.ensemble import RandomForestClassifier



grid_values = {'n_estimators': [20, 50], 'max_features': [7, 9], 'max_depth': [3, 5, 10]}



temp_tf = RandomForestClassifier(n_jobs=6)

grid_tf = GridSearchCV(temp_tf, param_grid = grid_values, scoring='recall')

grid_tf.fit(X_train, y_train)
grid_tf.best_params_
tforest = RandomForestClassifier(n_estimators=50, n_jobs=6, max_depth=10, max_features=9).fit(X_train, y_train)
test_pred_probs = tforest.predict_proba(X_test)[:, 1]



# (2) use vals function to convert the probabilities to predictions based on a give threshold

# If threshold is 0.5, then all observations with probability of 0.5 and higher will be assigned to class 1. Otherwise, class 0.

def vals (probabilities, threshold=0.5):

    class_preds = np.where(probabilities >= threshold, 1, 0)

    return class_preds



X_test_tf_class = vals(test_pred_probs, 0.5)



# (3) use the predictions and actual values to construct a confustion matrix and output the classification report.

print(confusion_matrix(y_test, X_test_tf_class))

print(classification_report(y_test, X_test_tf_class))
X_test_sc_probs = tforest.predict_proba(X_test)[:, 1]

recalls = []

precisions = []

def vals_2 (probabilities, threshold=0.5):

    class_preds = np.where(probabilities >= threshold, 1, 0)

    rscore = recall_score(y_test, class_preds)

    pscore = precision_score(y_test, class_preds)

    

    recalls.append(rscore)

    precisions.append(pscore)



for i in np.arange(0.0, 1.0, 0.05):

    vals_2(X_test_sc_probs, i)

    

plt.plot(np.arange(0.0, 1.0, 0.05), recalls, label="Recall")

plt.plot(np.arange(0.0, 1.0, 0.05), precisions, label="Precision")

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.xlabel('Thresholds')

plt.ylabel('Recall / Precisions')

plt.legend()
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



precision, recall, thresholds = precision_recall_curve(y_test, test_pred_probs)



closest_zero = np.argmin(np.abs(thresholds-0.5))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure()

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.axes().set_aspect('equal')

plt.show()
from sklearn.metrics import roc_curve, auc



fpr_lr, tpr_lr, tts = roc_curve(y_test, test_pred_probs)



roc_auc_lr = auc(fpr_lr, tpr_lr)



plt.figure()

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_lr, tpr_lr, lw=3, label=' (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=10)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()
pd.DataFrame([X4.columns, tforest.feature_importances_]).T.sort_values(1, ascending=False)
train.groupby(['compliance'])['late_fee'].mean()