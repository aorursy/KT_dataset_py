import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
%matplotlib inline
matplotlib.rcParams.update({'font.size': 20})
def read_csv(csv_file, nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows)
    print("File = {}".format(csv_file))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df

data_dir = "../input/"

df = read_csv(data_dir + "application_train.csv")
df_bureau = read_csv(data_dir + "bureau.csv")
df_previous_app = read_csv(data_dir + "previous_application.csv")
df_installments = read_csv(data_dir + "installments_payments.csv")
key = 'SK_ID_CURR'
bureau_cols = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']
bureau_cols_max = ['BUREAU_MAX_' + c for c in bureau_cols]

df = pd.merge(
    left=df,
    right=df_bureau[[key] + bureau_cols].groupby(key).max().rename(
        columns=dict(zip(bureau_cols, bureau_cols_max))),
    left_on=key,
    right_index=True, 
    how='left'
)

# Example: sample of 3 loans
df[[key] + bureau_cols_max].sample(3)
key_prev = 'SK_ID_PREV'
payment_cols = ['AMT_PAYMENT']

# Min payment for all previous loans
df_previous_app = pd.merge(
    left=df_previous_app,
    right=df_installments[[key_prev] + payment_cols].groupby(key_prev).min(),
    left_on=key_prev,
    right_index=True,
    how='left'
)

# Example: SK_ID_CURR #365597
df_previous_app[[key] + [key_prev] + payment_cols][df_previous_app.SK_ID_CURR == 365597]
key = 'SK_ID_CURR'
prev_agg_cols = ['PREV_SUM_MIN_AMT_PAYMENT', 'PREV_MEAN_MIN_AMT_PAYMENT']

# Sum and mean of minimum payments across all previous loans
df_prev_agg = df_previous_app[[key] + payment_cols].groupby(key).agg(['sum', 'mean']);
df_prev_agg.columns = prev_agg_cols

df = pd.merge(
    left=df,
    right=df_prev_agg,
    left_on=key,
    right_index=True,
    how='left'
)

# Example: SK_ID_CURR #365597
df[[key] + prev_agg_cols][df.SK_ID_CURR == 365597]
base_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY',
                'DAYS_EMPLOYED', 'AMT_GOODS_PRICE', 'DAYS_ID_PUBLISH',
                'OWN_CAR_AGE'
               ]
feature_cols = base_cols + bureau_cols_max + prev_agg_cols
y = df.TARGET
X = df[feature_cols]
X = X.fillna(value=X.mean())

# Example: SK_ID_CURR #365597
X[df.SK_ID_CURR == 365597].transpose()
clf = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)
def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score
cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []
    
for (train, test), i in zip(cv.split(X, y), range(5)):
    clf.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)

plot_roc_curve(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])