# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb



from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

from sklearn.utils import check_X_y

import sklearn.utils

from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss



import tensorflow as tf



import matplotlib.pyplot as plt



# hide warnings

import warnings

warnings.simplefilter("ignore")

# converter for percent values

p2f = lambda s: np.NaN if not s else float(s.strip().strip("%")) / 100



# load data for accepted loans

data = pd.read_csv("../input/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", skipinitialspace=True,

                   converters={"int_rate": p2f, "revol_util": p2f}, nrows=None, usecols=lambda c: False if c=="url" else True)

print(f"Loaded {len(data)} rows, {len(data.columns)} columns.")
dropna_cols = ["funded_amnt", "avg_cur_bal", "bc_util", "loan_status", "dti", "inq_last_6mths"]

len_prev = len(data)

data.dropna(subset=dropna_cols, inplace=True)

n_dropped = len_prev-len(data)

print(f"Dropped {n_dropped} rows ({100*n_dropped/len_prev:.2f}%) with NaN values, {len(data)} rows remaining.")
print(data["loan_status"].unique())
data["defaulted"] = data["loan_status"].map(lambda x: 1 if x in ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]

                                                 else 0 if x in ["Fully Paid", 'Does not meet the credit policy. Status:Fully Paid']

                                                 else -1)

len_prev = len(data)

data.query("defaulted != -1", inplace=True)

n_dropped = len_prev - len(data)

print(f"Dropped {n_dropped} rows ({100*n_dropped/len_prev:.2f}%) with invalid loan_status, {len(data)} rows remaining.")
sub_grades = sorted(data["sub_grade"].unique())

sub_grade_default_prob = {}

sub_grade_n = {}

for sg in sub_grades:

    sg_rows = data[data["sub_grade"] == sg]

    default_frac = sg_rows["defaulted"].sum() / len(sg_rows)

    sub_grade_default_prob[sg] = default_frac



data["sg_default_prob"] = data["sub_grade"].map(lambda x: sub_grade_default_prob[x])
sb.catplot(x="sub_grade", y="defaulted", data=data, kind="bar", aspect=5, order=sub_grades);
# fill NA values where appropriate

fill_na_values = {

    "emp_length": "missing",

}

data.fillna(value=fill_na_values, inplace=True)
# convert binary values to 1/0 (no need to one hot encode these)

data["hardship_flag01"] = data["hardship_flag"].map(lambda x: 1 if x == "Y" else 0)

data["joint_application_flag01"] = data["application_type"].map(lambda x: 0 if x=="Individual" else 1)

data["listed_as_whole_flag01"] = data["initial_list_status"].map(lambda x: 1 if x=="w" else 1)
individual_indices = data["joint_application_flag01"] == 0

data.loc[individual_indices, "annual_inc_joint"] = data[individual_indices]["annual_inc"]

data.loc[individual_indices, "dti_joint"] = data[individual_indices]["dti"]

data.loc[individual_indices, "verification_status_joint"] = data[individual_indices]["verification_status"]

data.loc[individual_indices, "revol_bal_joint"] = data[individual_indices]["revol_bal"]



dropna_cols = ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint"]

len_prev = len(data)

data.dropna(subset=dropna_cols, inplace=True)

n_dropped = len_prev-len(data)

print(f"Dropped {n_dropped} rows ({100*n_dropped/len_prev:.2f}%) with NaN values, {len(data)} rows remaining.")
# Our goal is to predict the loan status, given by the boolean "defaulted" column

y = data["defaulted"].values



# We define the columns used as features to train our model

# continuous valued columns or 

x_columns_cont = ["funded_amnt", "annual_inc", "annual_inc_joint", 

                "dti", "dti_joint","fico_range_low", "fico_range_high", "inq_last_6mths", "mort_acc",

                "open_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal", "revol_bal_joint", "revol_util"]

# binary categorical (0/1) columns

x_columns_bin = ["joint_application_flag01", "listed_as_whole_flag01"]



# columns with categorical values that need to be one hot encoded

x_columns_cat = ["term", "purpose", "hardship_flag", "emp_length", "verification_status_joint",

                 "addr_state"]





ct = ColumnTransformer(transformers=[

    ("identity", FunctionTransformer(func=lambda x: x, validate=False), x_columns_cont + x_columns_bin),

    ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore"), x_columns_cat),

])



X = ct.fit_transform(data)



X_col_labels = x_columns_cont + x_columns_bin + list(ct.named_transformers_["onehot"].get_feature_names())



# check for nans/infs and other stuff

X, y = check_X_y(X, y)
X_train, X_test, y_train, y_test, _, sg_default_prob_test = train_test_split(X, y, data["sg_default_prob"], test_size=0.2, random_state=42)

del X, y, data
# Columns to train on

col_select = [True for c in X_col_labels]

X_train_rest = X_train[:,col_select]

X_test_rest = X_test[:,col_select]

print(f"{len(X_train)} train, {len(X_test)} test samples")

scaler = StandardScaler()

X_train_rest = scaler.fit_transform(X_train_rest)

X_test_rest = scaler.transform (X_test_rest)



del X_train, X_test
ndef_train = np.sum(y_train == 1)

print(f"{ndef_train} ({100*ndef_train/len(y_train):.2f}%) defaulted in training data set")



train_ndefault = np.sum(y_train)

X_train_rest_nondefault = X_train_rest[y_train == 0]

X_train_rest_balanced_nondefault = X_train_rest_nondefault[

    sklearn.utils.random.sample_without_replacement(len(X_train_rest_nondefault),

                                                    train_ndefault, random_state=42)]



X_train_rest_balanced = np.concatenate([X_train_rest[y_train == 1], X_train_rest_balanced_nondefault])



y_train_balanced = np.concatenate([np.ones(train_ndefault),np.zeros(train_ndefault)])

# Now the classes are balanced:

print(f"{np.sum(y_train_balanced == 1)} ({100*np.sum(y_train==1)/len(y_train_balanced):.2f}%) defaulted in balanced training data set")



# define a function to correct probabilities coming from models trained on the balanced dataset

beta = ndef_train / (len(y_train) - ndef_train) # ratio of defaults to non-defaults

# because of numerical errors the probability could be slightly above 1, so we clip the value

correct_balanced_probabilities = lambda probs: np.clip(beta * probs / ((beta - 1) * probs + 1), 0, 1)
# Train a random forest classifier

rf_model = RandomForestClassifier(n_estimators=200, oob_score=True, class_weight="balanced", n_jobs=-1, verbose=True)

rf_model.fit(X_train_rest, y_train)



# print some statistics

print(f"Out of bag score: {rf_model.oob_score_}")

test_pred_proba_rf = rf_model.predict_proba(X_test_rest)[:,1]

test_pred_rf = test_pred_proba_rf > 0.5

print("Test:")

print(classification_report(y_test, test_pred_rf))

print(f"False positives: {np.sum(np.logical_and(test_pred_rf == 1,y_test == 0))}")

print(f"False negatives: {np.sum(np.logical_and(test_pred_rf == 0,y_test == 1))}")

print(f"True positives: {np.sum(np.logical_and(test_pred_rf == 1,y_test == 1))}")

print(f"True negatives: {np.sum(np.logical_and(test_pred_rf == 0,y_test == 0))}")



rf_model_feature_importances = rf_model.feature_importances_

del rf_model
# Train a random forest classifier on the balanced dataset

rf_b_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=True, class_weight=None)

rf_b_model.fit(X_train_rest_balanced, y_train_balanced)



# predict

test_pred_proba_rf_b = correct_balanced_probabilities(rf_b_model.predict_proba(X_test_rest)[:,1])



del rf_b_model
# Train a random forest classifier on the unbalanced dataset without weighting (this should be strictly worse(?))

rf_ub_uw_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=True, class_weight=None)

rf_ub_uw_model.fit(X_train_rest, y_train)



# predict

test_pred_proba_rf_ub_uw = rf_ub_uw_model.predict_proba(X_test_rest)[:,1]



del rf_ub_uw_model
# Train a linear SVM classifier on a subset of the unbalanced dataset using "balanced" class weights

X_train_rest_sample, y_train_sample = sklearn.utils.resample(X_train_rest, y_train, n_samples=20000, random_state=42)



print(f"{np.sum(y_train_sample == 1)} ({100*np.sum(y_train_sample==1)/len(y_train_sample):.2f}%) defaulted in subsampled training data set")



svm_model = svm.SVC(kernel="linear", class_weight="balanced", probability=True)

svm_model.fit(X_train_rest_sample, y_train_sample)



# predict

test_pred_proba_svm = svm_model.predict_proba(X_test_rest)[:,1]



del svm_model
# Train a linear SVM classifier on a subset of the balanced dataset (using the full dataset is infeasible as the algorithm is O(n^2))

X_train_rest_sample, y_train_sample =sklearn.utils.resample(X_train_rest_balanced, y_train_balanced, n_samples=20000, random_state=42)



print(f"{np.sum(y_train_sample == 1)} ({100*np.sum(y_train_sample==1)/len(y_train_sample):.2f}%) defaulted in subsampled balanced training data set")



svm_b_model = svm.SVC(kernel="linear", probability=True)

svm_b_model.fit(X_train_rest_sample, y_train_sample)



# predict and correct probabilities

test_pred_proba_svm_b = correct_balanced_probabilities(svm_b_model.predict_proba(X_test_rest)[:,1])



del svm_b_model, X_train_rest_sample, y_train_sample
# Train an rbf SVM classifier on a subset of the balanced dataset (using the full dataset is infeasible as the algorithm is O(n^2))



X_train_rest_sample, y_train_sample = sklearn.utils.resample(X_train_rest_balanced, y_train_balanced, n_samples=20000, random_state=42)



svm_rbf_b_model = svm.SVC(kernel="rbf", probability=True)

svm_rbf_b_model.fit(X_train_rest_sample, y_train_sample)

test_pred_proba_svm_rbf_b = correct_balanced_probabilities(svm_rbf_b_model.predict_proba(X_test_rest)[:,1])



del svm_rbf_b_model, X_train_rest_sample, y_train_sample
# Train an rbf SVM classifier on a subset of the unbalanced dataset (using the full dataset is infeasible as the algorithm is O(n^2))



X_train_rest_sample, y_train_sample = sklearn.utils.resample(X_train_rest, y_train, n_samples=20000, random_state=42)



svm_rbf_model = svm.SVC(kernel="rbf", class_weight="balanced", probability=True)

svm_rbf_model.fit(X_train_rest_sample, y_train_sample)

test_pred_proba_svm_rbf = svm_rbf_model.predict_proba(X_test_rest)[:,1]



del svm_rbf_model, X_train_rest_sample, y_train_sample
nn_train_b_x, nn_cal_b_x, nn_train_b_y, nn_cal_b_y = train_test_split(X_train_rest_balanced, y_train_balanced, test_size=0.1)

nn_train_ub_x, nn_cal_ub_x, nn_train_ub_y, nn_cal_ub_y = train_test_split(X_train_rest, y_train, test_size=0.025)



# we don't need biases in dense layers as that's covered by batchnorm

# we don't need scaling in batchnorm layers as that's done by the weights in the next layer

nn_model_b = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, input_shape=(nn_train_b_x.shape[1],), use_bias=False),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Dense(64, use_bias=False),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Dense(64, use_bias=False),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Dense(64, use_bias=False),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)

])



nn_model_ub = tf.keras.models.clone_model(nn_model_b)



nn_model_b.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05, decay=0.005),

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])

nn_model_b.fit(nn_train_b_x, nn_train_b_y, epochs=5)



nn_model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05, decay=0.005),

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])



nn_model_ub.fit(nn_train_ub_x, nn_train_ub_y, epochs=5)
from scipy.optimize import minimize



sigmoid = lambda x: 1/(1+np.exp(-x))



# mean cross entropy loss

ce_cost = lambda y_true, proba: -np.sum((y_true * np.log(proba) + (1-y_true) * np.log(1 - proba))) / len(y_true)



def platt_cal_model(y_true, y_logits):

    """Returns a function that Platt scales the logits output by the model, giving (better) calibrated probabilities.

    

    See https://arxiv.org/abs/1706.04599 for more details / theoretical background.

    The inputs y_true and y_logits are the true labels and outputs of the (uncalibrated) model on a dataset that was NOT used for training.

    """

    

    print(f"Cost before: {ce_cost(y_true, sigmoid(y_logits)):.4f}")

    res = minimize(lambda x, y_logits, y_true: ce_cost(y_true, sigmoid(y_logits * x[0] + x[1])),

                   [1.0,0.0], args=(y_logits, y_true), method="Nelder-Mead", options={'maxiter': 500, 'disp': True}, )

    a, b = res.x

    

    cal_function = lambda logits: sigmoid(logits * a + b)

    

    print(f"a:{a}, b:{b}")

    print(f"Cost after : {ce_cost(y_true, cal_function(y_logits)):.4f}")

    

    return cal_function
print("NN model trained on balanced dataset")

# squeezing the output of the model is necessary, otherwise its shape is (n, 1) and multiplying with a shape (n,) array (e.g. in ce_loss) creates a gigantic (n,n) result...

test_pred_logits_nn_b = nn_model_b.predict(X_test_rest).squeeze()

print(f"brier score on test set without balancing correction: {brier_score_loss(y_test, sigmoid(test_pred_logits_nn_b)):.4f}")



test_pred_proba_nn_b = correct_balanced_probabilities(sigmoid(test_pred_logits_nn_b))

print(f"brier score on test set with balancing correction: {brier_score_loss(y_test, test_pred_proba_nn_b):.4f}")



cal_pred_logits_nn_b = nn_model_b.predict(nn_cal_b_x).squeeze()



nn_b_calibration = platt_cal_model(nn_cal_b_y, cal_pred_logits_nn_b)

test_pred_proba_nn_b_platt = correct_balanced_probabilities(nn_b_calibration(test_pred_logits_nn_b))

print(f"brier score on test set with balancing correction and calibration: {brier_score_loss(y_test, test_pred_proba_nn_b_platt):.4f}")



print("\n\nNN model trained on unbalanced dataset")



test_pred_logits_nn_ub = nn_model_ub.predict(X_test_rest).squeeze()

test_pred_proba_nn_ub = sigmoid(test_pred_logits_nn_ub)

print(f"brier score on test set without calibration: {brier_score_loss(y_test, test_pred_proba_nn_ub):.4f}")



cal_pred_logits_nn_ub = nn_model_ub.predict(nn_cal_ub_x).squeeze()



nn_calibration = platt_cal_model(nn_cal_ub_y, cal_pred_logits_nn_ub)

test_pred_proba_nn_ub_platt = nn_calibration(test_pred_logits_nn_ub)

print(f"brier score on test set with platt calibration: {brier_score_loss(y_test, test_pred_proba_nn_ub_platt):.4f}")
# calculate some model statistics

models = [

    (sg_default_prob_test, "subgrade"),

    (test_pred_proba_rf_ub_uw, "random forest raw"),

    (test_pred_proba_rf, "random forest weighted"),

    (test_pred_proba_rf_b, "random forest balanced"),

    (test_pred_proba_svm, "linear SVM weighted"),

    (test_pred_proba_svm_b, "linear SVM balanced"),

    (test_pred_proba_svm_rbf, "RBF SVM weighted"),

    (test_pred_proba_svm_rbf_b, "RBF SVM balanced"),

    (test_pred_proba_nn_b, "NN balanced"),

    (test_pred_proba_nn_b_platt, "NN balanced w. Platt"),

    (test_pred_proba_nn_ub, "NN unbalanced"),

    (test_pred_proba_nn_ub_platt, "NN unbalanced w. Platt")

]

model_stats = {}

for proba, title in models:

    fpr, tpr, thresholds = roc_curve(y_test, proba)

    auc_score = auc(fpr,tpr)

    brier = brier_score_loss(y_test, proba)

    model_stats[title] = (fpr, tpr, auc_score, brier)

plt.figure(figsize=(15,7))

for proba, title in models:

    fpr, tpr, auc_score, brier = model_stats[title]

    plt.plot(fpr, tpr, label=f"{title} (auc: {auc_score:.5f}, Brier: {brier:.5f})")

    

plt.plot([0,1], [0,1], label="guess")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right");
plt.figure(figsize=(15,10))

fpr_sg, tpr_sg, _, _ = model_stats["subgrade"]

for proba, title in models:

    fpr, tpr, auc_score, brier = model_stats[title]

    plt.plot(fpr, tpr-np.interp(fpr, fpr_sg, tpr_sg), label=f"{title} (auc: {auc_score:.5f}, Brier: {brier:.5f})")

    

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate - Grade True Positive Rate')

plt.legend(loc="lower right");
def plot_default_prob_pred_vs_actual(pred_proba, y, prob_min=0, prob_max=0.55, nsteps=15, title=None, ax=None):

    """Generate a calibration plot"""

    prob_cutoffs = np.linspace(prob_min, prob_max, num=nsteps)

    prob_cutoff_actual = []

    prob_cutoff_n = []

    xs = []

    for lower_p, upper_p in zip(prob_cutoffs, prob_cutoffs[1:]):

        cutoff_mask = np.logical_and(pred_proba > lower_p, pred_proba < upper_p) # predicted prob of default > p

        n = np.sum(cutoff_mask)

        defaulted_frac = np.sum(y[cutoff_mask])/n

        xs.append((lower_p + upper_p) / 2)

        prob_cutoff_actual.append(defaulted_frac)

        prob_cutoff_n.append(n)

        

    xs = np.array(xs) * 100

    

    if ax is None:

        fig, ax = plt.subplots();

    color = 'tab:red'

    ax.set_title(title)

    ax.set_xlabel('pred. min default prob [%]')

    ax.set_ylabel('default prob [%]', color=color)

    ax.plot(xs, np.array(prob_cutoff_actual) * 100, 1, color=color)

    ax.plot([prob_min * 100,prob_max * 100], [prob_min * 100, prob_max * 100], color="gray", ls="dashed")

    ax.tick_params(axis='y', labelcolor=color)



    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_yscale("log")



    color = 'tab:blue'

    ax2.set_ylabel('n', color=color)  # we already handled the x-label with ax1

    ax2.plot(xs, prob_cutoff_n, color=color)

    ax2.tick_params(axis='y', labelcolor=color)
fig, axes = plt.subplots(3, 4, figsize=(20,12))

axes = [a for row in axes for a in row] # flatten array

for (proba, title), axis in zip(models, axes):

    plot_default_prob_pred_vs_actual(proba, y_test, ax=axis, title=title)

    

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plot_default_prob_pred_vs_actual(sigmoid(test_pred_logits_nn_b), y_test, title="balanced NN model without balancing correction")
plt.figure(figsize=(17,3))

ax = plt.axes()

sb.barplot(X_col_labels, rf_model_feature_importances, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90);