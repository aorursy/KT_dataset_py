!pip install fairlearn
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from IPython.display import display, HTML
df = pd.read_csv("../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv")
df.head()
# Drop irrelevant columns

df.drop(columns=['Loan_ID', 'LoanAmount', 'Loan_Amount_Term'], inplace=True)
df.isnull().sum()
df = df[(~df['Gender'].isnull()) & (~df['Married'].isnull())]

df['Dependents'] = df['Dependents'].map({'0': '0',

                                         '1': '1',

                                         '2': '2',

                                         '3+': '3'})

fill_values = {'Self_Employed': 'NaN', 'Dependents': 'NaN', 'Credit_History': -1}

df.fillna(value=fill_values, inplace=True)
df.isnull().sum()
# Apply LabelEncoder on each of the categorical columns

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 

                    'Credit_History', 'Property_Area', 'Dependents', 'Loan_Status']

le = LabelEncoder()



df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))



# Apply OneHotEncoder on each of the categorical columns

categorical_cols = ['Self_Employed', 'Credit_History', 'Property_Area', 'Dependents']



encoded_features = []

ohe = OneHotEncoder()

for feature in categorical_cols:

    encoded_feat = OneHotEncoder(drop='first').fit_transform(df[feature].values.reshape(-1, 1)).toarray()

    n = df[feature].nunique()

    cols = ['{}_{}'.format(feature, n) for n in range(0, n-1)]

    encoded_df = pd.DataFrame(encoded_feat, columns=cols)

    encoded_df.index = df.index

    encoded_features.append(encoded_df)
df = pd.concat([df.drop(columns=categorical_cols), *encoded_features], axis=1)
df.columns
# Train test split



A = df['Gender']





X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(df.drop(columns="Loan_Status"), 

                                                                     df['Loan_Status'],

                                                                     A,

                                                                     stratify=df['Loan_Status'], 

                                                                     test_size=0.2)



# Combine all training data into a single data frame and glance at a few rows

all_train = pd.concat([X_train, y_train], axis=1)
import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams['figure.figsize'] = (15, 5)

plt.style.use('seaborn-white')



plt.subplot(1, 2, 1)



y_train.value_counts().plot(kind = 'pie',

                            autopct = '%.2f%%',

                            startangle = 90,

                            labels = ['Loan granted', 'Loan not granted'],

                            pctdistance = 0.5)



plt.xlabel('Training dataset', fontsize = 14)



plt.subplot(1, 2, 2)



y_test.value_counts().plot(kind = 'pie',

                           autopct = '%.2f%%',

                           startangle = 90,

                           labels = ['Loan granted', 'Loan not granted'],

                           pctdistance = 0.5)



plt.xlabel('Testing dataset', fontsize = 14)



plt.suptitle('Target Class Balance', fontsize = 16)

plt.show()
gender_grouped = all_train.groupby('Gender')

counts_by_gender = gender_grouped[['Loan_Status']].count().rename(

    columns={'Loan_Status': 'count'})



rates_by_gender = gender_grouped[['Loan_Status']].mean().rename(

    columns={'Loan_Status': 'pass_loan_rate'})



summary_by_gender = pd.concat([counts_by_gender, rates_by_gender], axis=1)

display(summary_by_gender)
from sklearn.linear_model import LogisticRegression



unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)

unmitigated_predictor.fit(X_train, y_train)
from fairlearn.metrics import roc_auc_score_group_summary



def summary_as_df(name, summary):

    a = pd.Series(summary.by_group)

    a['overall'] = summary.overall

    return pd.DataFrame({name: a})



scores_unmitigated = pd.Series(unmitigated_predictor.predict_proba(X_test)[:,1], name="score_unmitigated")

y_pred = (scores_unmitigated >= np.mean(y_test)) * 1

auc_unmitigated = summary_as_df(

    "auc_unmitigated", roc_auc_score_group_summary(y_test, scores_unmitigated, sensitive_features=A_test))
from fairlearn.metrics import (

    group_summary, selection_rate, selection_rate_group_summary,

    demographic_parity_difference, demographic_parity_ratio,

    balanced_accuracy_score_group_summary, roc_auc_score_group_summary,

    equalized_odds_difference, difference_from_summary)

from sklearn.metrics import balanced_accuracy_score, roc_auc_score





# Helper functions

def get_metrics_df(models_dict, y_true, group):

    metrics_dict = {

        "Overall selection rate": (

            lambda x: selection_rate(y_true, x), True),

        "Demographic parity difference": (

            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),

        "Demographic parity ratio": (

            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),

        "-----": (lambda x: "", True),

        "Overall balanced error rate": (

            lambda x: 1-balanced_accuracy_score(y_true, x), True),

        "Balanced error rate difference": (

            lambda x: difference_from_summary(

                balanced_accuracy_score_group_summary(y_true, x, sensitive_features=group)), True),

        "Equalized odds difference": (

            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),

        "------": (lambda x: "", True),

        "Overall AUC": (

            lambda x: roc_auc_score(y_true, x), False),

        "AUC difference": (

            lambda x: difference_from_summary(

                roc_auc_score_group_summary(y_true, x, sensitive_features=group)), False),

    }

    df_dict = {}

    for metric_name, (metric_func, use_preds) in metrics_dict.items():

        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores) 

                                for model_name, (preds, scores) in models_dict.items()]

    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())
models_dict = {"Unmitigated": (y_pred, scores_unmitigated)}

get_metrics_df(models_dict, y_test, A_test)
gs = group_summary(roc_auc_score, y_test, y_pred, sensitive_features=A_test)

gs



plt.figure()

plt.style.use('seaborn-white')

plt.title("AUC per group before mitigating model biases", fontsize = 16)

plt.bar(range(len(gs["by_group"])), list(gs["by_group"].values()), align='center')

plt.xticks(range(len(gs["by_group"])), ['Female', 'Male'])

plt.ylim(0, 1)

plt.show()
srg = selection_rate_group_summary(y_test, y_pred, sensitive_features=A_test)



plt.figure()

plt.style.use('seaborn-white')

plt.title("Selection rate per group before mitigating model biases", fontsize = 16)

plt.bar(range(len(srg["by_group"])), list(srg["by_group"].values()), align='center')

plt.xticks(range(len(srg["by_group"])), ['Female', 'Male'])

plt.ylim(0, 1)

plt.show()
from fairlearn.widget import FairlearnDashboard



FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=['Gender'],

                   y_true=y_test,

                   y_pred={"Unmitigated": y_pred})
from fairlearn.reductions import GridSearch, DemographicParity

from sklearn.calibration import CalibratedClassifierCV



sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),

                   constraints=DemographicParity(),

                   grid_size=200,

                   grid_limit=10)



sweep.fit(X_train, y_train, sensitive_features=A_train)



calibrated_predictors = []

for predictor in sweep._predictors:

    calibrated = CalibratedClassifierCV(base_estimator=predictor, cv='prefit', method='sigmoid')

    calibrated.fit(X_train, y_train)

    calibrated_predictors.append(calibrated)
from scipy.stats import cumfreq



def compare_cdfs(data, A, num_bins=100):

    cdfs = {}

    assert len(np.unique(A)) == 2

    

    limits = ( min(data), max(data) )

    s = 0.5 * (limits[1] - limits[0]) / (num_bins - 1)

    limits = ( limits[0]-s, limits[1] + s)

    

    for a in np.unique(A):

        subset = data[A==a]

        

        cdfs[a] = cumfreq(subset, numbins=num_bins, defaultreallimits=limits)

        

    lower_limits = [v.lowerlimit for _, v in cdfs.items()]

    bin_sizes = [v.binsize for _,v in cdfs.items()]

    actual_num_bins = [v.cumcount.size for _,v in cdfs.items()]

    

    assert len(np.unique(lower_limits)) == 1

    assert len(np.unique(bin_sizes)) == 1

    assert np.all([num_bins==v.cumcount.size for _,v in cdfs.items()])

    

    xs = lower_limits[0] + np.linspace(0, bin_sizes[0]*num_bins, num_bins)

    

    disparities = np.zeros(num_bins)

    for i in range(num_bins):

        cdf_values = np.clip([v.cumcount[i]/len(data[A==k]) for k,v in cdfs.items()],0,1)

        disparities[i] = max(cdf_values)-min(cdf_values)  

    

    return xs, cdfs, disparities





def plot_and_compare_cdfs(data, A, num_bins=100, loc='best'):

    xs, cdfs, disparities = compare_cdfs(data, A, num_bins)

    

    for k, v in cdfs.items():

        plt.plot(xs, v.cumcount/len(data[A==k]), label=k)

    

    assert disparities.argmax().size == 1

    d_idx = disparities.argmax()

    

    xs_line = [xs[d_idx],xs[d_idx]]

    counts = [v.cumcount[d_idx]/len(data[A==k]) for k, v in cdfs.items()]

    ys_line = [min(counts), max(counts)]

    

    plt.plot(xs_line, ys_line, 'o--')

    disparity_label = "max disparity = {0:.3f}\nat {1:0.3f}".format(disparities[d_idx], xs[d_idx])

    plt.text(xs[d_idx], 1, disparity_label, ha="right", va="top")

    

    plt.xlabel(data.name)

    plt.ylabel("cumulative frequency")

    plt.legend(loc=loc)

    plt.show()
from fairlearn.metrics import roc_auc_score_group_min



def auc_disparity_sweep_plot(predictors, names, marker='o', scale_size=1, zorder=-1):

    roc_auc = np.zeros(len(predictors))

    disparity = np.zeros(len(predictors))

    

    for i in range(len(predictors)):

        preds = predictors[i].predict_proba(X_test)[:,1]

        roc_auc[i] = roc_auc_score_group_min(y_test, preds, sensitive_features=A_test)

        _, _, dis = compare_cdfs(preds, A_test)

        disparity[i] = dis.max()

        

    plt.scatter(roc_auc, disparity,

                s=scale_size * plt.rcParams['lines.markersize'] ** 2, marker=marker, zorder=zorder)

    for i in range(len(roc_auc)):

        plt.annotate(names[i], (roc_auc[i], disparity[i]), xytext=(3,2), textcoords="offset points", zorder=zorder+1)

    plt.xlabel("worst-case AUC")

    plt.ylabel("demographic disparity")

    

auc_disparity_sweep_plot(calibrated_predictors, names=range(len(calibrated_predictors)))

auc_disparity_sweep_plot([unmitigated_predictor], names=[''], marker='*', zorder=1, scale_size=5)

plt.show()
# a convenience function that transforms the result of a group metric call into a data frame



scores_model101 = pd.Series(calibrated_predictors[101].predict_proba(X_test)[:,1], name="score_model101")



auc_model101 = summary_as_df(

        "auc_model101", roc_auc_score_group_summary(y_test, scores_model101, sensitive_features=A_test))



display(HTML('<span id="grid_search_comparison">'),

        pd.concat([auc_model101, auc_unmitigated], axis=1),

        HTML('</span>'))

plot_and_compare_cdfs(scores_model101, A_test.reset_index(drop=True))

plot_and_compare_cdfs(scores_unmitigated, A_test.reset_index(drop=True))
y_pred_mitigated = (scores_model101 >= np.mean(y_test)) * 1



gs = group_summary(roc_auc_score, y_test, y_pred_mitigated, sensitive_features=A_test)

gs



plt.figure()

plt.style.use('seaborn-white')

plt.title("AUC per group after mitigating model biases", fontsize = 16)

plt.bar(range(len(gs["by_group"])), list(gs["by_group"].values()), align='center')

plt.xticks(range(len(gs["by_group"])), ['Female', 'Male'])

plt.ylim(0, 1)

plt.show()
srg = selection_rate_group_summary(y_test, y_pred_mitigated, sensitive_features=A_test)



plt.figure()

plt.style.use('seaborn-white')

plt.title("Selection rate per group after mitigating model biases", fontsize = 16)

plt.bar(range(len(srg["by_group"])), list(srg["by_group"].values()), align='center')

plt.xticks(range(len(srg["by_group"])), ['Female', 'Male'])

plt.ylim(0, 1)

plt.show()
models_dict = {"Unmitigated": (y_pred, scores_unmitigated)}

get_metrics_df(models_dict, y_test, A_test)
models_dict = {"Mitigated": (y_pred_mitigated, scores_model101)}

get_metrics_df(models_dict, y_test, A_test)