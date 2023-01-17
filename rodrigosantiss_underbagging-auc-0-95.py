import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
%matplotlib inline
def process(df):
    # Imput missing lines and drop line with problem
    from sklearn.preprocessing import Imputer
    df['lead_time'] = Imputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    df = df.dropna()
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        df[col] = Imputer(missing_values=-99).fit_transform(
                                    df[col].values.reshape(-1, 1))
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        df[col] = (df[col] == 'Yes').astype(int)
    # Normalization    
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    # Scale lead time
    df.lead_time = df.lead_time/df.lead_time.max().astype(np.float64)    
    return df

cols=range(1,23)
df = process(pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv',usecols=cols))
df.head()
def plot_2d(X, y, title=''):
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)

    from sklearn.decomposition import PCA
    dec = PCA(n_components=2)
    X_reduced = dec.fit_transform(X_std)
    
    f, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X_reduced[y==0,0], X_reduced[y==0,1], alpha=0.5, 
               facecolors='none', edgecolors='cornflowerblue', label="Negative")
    ax.scatter(X_reduced[y==1,0], X_reduced[y==1,1], c='darkorange', marker='*', 
               label='Positive')
    plt.title("Explained variance ratio: %.2f%%" % (100*dec.explained_variance_ratio_.sum()))
    ax.legend(loc='lower left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.show()
    
sample = df.sample(5000, random_state=36)

X_sample = sample.drop('went_on_backorder',axis=1).values
y_sample = sample['went_on_backorder'].values

plot_2d(X_sample, y_sample)
X = df.drop('went_on_backorder', axis=1).values
y = df['went_on_backorder'].values
print('Imbalanced ratio in training set: 1:%i' % (Counter(y)[0]/Counter(y)[1]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)
rus = make_pipeline(RandomUnderSampler(),tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5))
forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)

cart.fit(X_train, y_train)
rus.fit(X_train, y_train)
forest.fit(X_train, y_train)
#gboost.fit(X_train, y_train)
n_splits = 10

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
ub = BaggingClassifier(warm_start=True, n_estimators=0)

for split in range(n_splits):
    X_res, y_res = RandomUnderSampler(random_state=split).fit_sample(X_train,y_train) 
    ub.n_estimators += 1
    ub.fit(X_res, y_res)
def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

f, ax = plt.subplots(figsize=(6,6))

roc_auc_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')
roc_auc_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='--')
roc_auc_plot(y_test,cart.predict_proba(X_test),label='CART', l='-.')
roc_auc_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')

ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        label='Random Classifier')    
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic curves')
sns.despine()
def precision_recall_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_test,
                                                  y_proba[:,1])
    average_precision = average_precision_score(y_test, y_proba[:,1],
                                                     average="micro")
    ax.plot(recall, precision, label='%s (average=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)

f, ax = plt.subplots(figsize=(6,6))
precision_recall_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')
precision_recall_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='-')
precision_recall_plot(y_test,cart.predict_proba(X_test),label='CART',l='-.')
precision_recall_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc="upper right")
ax.grid(True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Precision-recall curves')
sns.despine()
