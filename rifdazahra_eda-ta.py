import numpy as np

import scipy as sp

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling
df = pd.read_csv('../input/pre-eda-ta/pre_eda.csv', index_col=None)

df.head()
df=df.drop('Unnamed: 0',axis=1)
print(list(df.columns))
#splitting categorical and continuous features

continuous_features=[value for value in list(df._get_numeric_data().columns) if value not in ["policy_code","is_charged_off"]]

categorical_features=[value for value in df.columns if value not in [*continuous_features,"is_charged_off","loan_status"]]

target_label="is_charged_off"
def hist(col_name, full_name, continuous):

    """

    Visualize a variable with and without faceting on the loan status.

    - col_name is the variable name in the dataframe

    - full_name is the full variable name

    - continuous is True if the variable is continuous, False otherwise

    """

    f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6,5), dpi=90)

    

    # Plot without loan status

    if continuous:

        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)

    else:

        sns.countplot(df[col_name], order=df[col_name].value_counts().index, color='#5975A4', saturation=1, ax=ax1)

    ax1.set_xlabel(full_name)

    ax1.set_ylabel('Count')

    ax1.set_title("Bar Plot {}".format(full_name))

    plt.xticks(rotation='vertical')

    

    plt.savefig("{}_bar_plot_all.png".format(col_name))

#     plt.show() 

    

    
def box(col_name, full_name, continuous):

    """

    Visualize a variable with and without faceting on the loan status.

    - col_name is the variable name in the dataframe

    - full_name is the full variable name

    - continuous is True if the variable is continuous, False otherwise

    """

    f, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(9,5), dpi=90)

    

    # Plot with loan status

    if continuous:

        sns.boxplot(x=col_name, y='loan_status', data=df, ax=ax2)

        ax2.set_ylabel('')

        ax2.set_title(full_name + ' by Loan Status')

    else:

        charge_off_rates = df.groupby(col_name)['loan_status'].value_counts(normalize=True,sort=True).loc[:,'Charged Off']

        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1, ax=ax2)

        ax2.set_ylabel('Fraction of df Charged-off')

        ax2.set_title('Charge-off Rate by ' + full_name)

    ax2.set_xlabel(full_name)

    plt.xticks(rotation='vertical')



#     plt.savefig("{}_box_plot.png".format(col_name))

#     plt.show()



# for feature in categorical_features:

# # feature = 'zip_code'

#     count = df[feature].describe().loc['count']

# #     unique = df[feature].describe().loc['unique']

#     top = df[feature].describe().loc['top']

#     freq = df[feature].describe().loc['freq']

#     dtype = df[feature].dtypes

#     table = """% Please add the following required packages to your document preamble:

#     % \\usepackage{{longtable}}

#     % Note: It may be necessary to compile the document several times to get a multi-page table to line up properly

#     \\begin{{longtable}}[c]{{|llll|}}

#     \\caption{{Deskripsi Fitur {}}}

#     \\label{{tab:desc-{}}}\\\\

#     \\hline

#     \\textbf{{Nama Fitur}} & \\textit{{{}}} & \\textbf{{Type Data}} & {} \\\\

#     \\endfirsthead

#     %

#     \\multicolumn{{4}}{{c}}%

#     {{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\

#     \\hline

#     \\textbf{{Nama Fitur}} & \\textit{{{}}} & \\textbf{{Type Data}} & {} \\\\

#     \\endhead

#     %

#     \\textbf{{Count}} & ${}$ & \\textbf{{Top}} & {} \\\\

#     \\textbf{{Unique}} & ${}$ & \\textbf{{Freq}} & ${}$ \\ \\hline

#     \\end{{longtable}}

#     """.format(feature,feature,feature,dtype,feature,dtype,count,top,freq,freq)

#     text_file = open("desc_{}.tex".format(feature), "wt")

#     n = text_file.write(table)

#     text_file.close()
for feature in continuous_features:

    count = df[feature].describe().loc['count']

    mean = df[feature].describe().loc['mean']

    std = df[feature].describe().loc['std']

    minn = df[feature].describe().loc['min']

    dualima = df[feature].describe().loc['25%']

    limapuluh = df[feature].describe().loc['50%']

    tujuhlima = df[feature].describe().loc['75%']

    maxx = df[feature].describe().loc['max']

    dtype = df[feature].dtypes

    table = """% Please add the following required packages to your document preamble:

    % \\usepackage{{longtable}}

    % Note: It may be necessary to compile the document several times to get a multi-page table to line up properly

    \\begin{{longtable}}[c]{{|llll|}}

    \\caption{{Deskripsi Fitur {}}}

    \\label{{tab:desc-{}}}\\\\

    \\hline

    \\textbf{{Nama Fitur}} & \\textit{{{}}} & \\textbf{{Min}} & ${:.2f}$ \\\\

    \\endfirsthead

    %

    \\multicolumn{{4}}{{c}}%

    {{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\

    \\hline

    \\textbf{{Nama Fitur}} & \\textit{{{}}} & \\textbf{{Min}} & ${:.2f}$ \\\\

    \\endhead

    %

    \\textbf{{Type Data}} & {} & \\textbf{{25\\%}} & ${:.2f}$ \\\\

    \\textbf{{Count}} & ${}$ & \\textbf{{50\\%}} & ${:.2f}$ \\\\

    \\textbf{{Mean}} & ${:.2f}$ & \\textbf{{75\\%}} & ${:.2f}$ \\\\

    \\textbf{{Std}} & ${:.2f}$ & \\textbf{{Max}} & ${:.2f}$ \\ \\hline

    \\end{{longtable}}

    """.format(feature,feature,feature,minn,feature,minn,dtype,dualima,count,limapuluh,mean,tujuhlima,std,maxx)

    text_file = open("desc_{}.tex".format(feature), "wt")

    n = text_file.write(table)

    text_file.close()
# for feature in continuous_features:

#     box(feature, feature, continuous=True)

#     hist(feature, feature, continuous=True)
# for feature in continuous_features:

#     box(feature, feature, continuous=True)

#     hist(feature, feature, continuous=True)

    

for feature in categorical_features:

#     box(feature, feature, continuous=False)

    hist(feature, feature, continuous=False)
df.emp_title.value_counts().nlargest(10)
# # Check Class variables that has 0 value for Genuine transactions and 1 for Fraud

# print("Class as pie chart:")

# fig, ax = plt.subplots(1, 1)

# ax.pie(df.loan_status.value_counts(),autopct='%1.1f%%', labels=['Fully Paid','Charged off'], colors=['plum','thistle'])

# plt.axis('equal')

# plt.ylabel('')
# #splitting categorical and continuous features

# continuous_features=[value for value in list(df._get_numeric_data().columns) if value not in ["policy_code","is_charged_off"]]

# # categorical_features=[value for value in df.columns if value not in [*continuous_features,"is_charged_off"]]

# target_label="is_charged_off"
# #let us check correlations and shapes of those 25 principal components.

# # Features V1, V2, ... V28 are the principal components obtained with PCA.

# import seaborn as sns

# import matplotlib.gridspec as gridspec

# gs = gridspec.GridSpec(43, 1)

# plt.figure(figsize=(6,43*4))

# for i, col in enumerate(df[df.iloc[:,0:43].columns]):

#     if str(col) in continuous_features:

#         ax5 = plt.subplot(gs[i])

#         sns.distplot(df[col][df[target_label] == 1], bins=50, color='r', kde_kws={'bw':0.1})

#         sns.distplot(df[col][df[target_label] == 0], bins=50, color='g', kde_kws={'bw':0.1})

#         ax5.set_xlabel('')

#         ax5.set_title('feature: ' + str(col))

# plt.show()
df['earliest_cr_line'] = df['earliest_cr_line'].astype('datetime64[ns]')
df['ecl_year'] = pd.DatetimeIndex(df['earliest_cr_line']).year

df['ecl_month'] = pd.DatetimeIndex(df['earliest_cr_line']).month
hist('ecl_year', 'ecl_year', continuous=False)
from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
def split_data(df, drop_list):

    df = df.drop(drop_list,axis=1)

    print(df.columns)

    #test train split time

    from sklearn.model_selection import train_test_split

    y = df[target_label].values #target

    X = df.drop([target_label],axis=1).values #features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                    random_state=42, stratify=y)



    print("train-set size: ", len(y_train),

      "\ntest-set size: ", len(y_test))

    print("charged-off cases in test-set: ", sum(y_test))

    return X_train, X_test, y_train, y_test
def get_predictions(clf, X_train, y_train, X_test):

    # create classifier

    clf = clf

    # fit it to training data

    clf.fit(X_train,y_train)

    # predict using test data

    y_pred = clf.predict(X_test)

    # Compute predicted probabilities: y_pred_prob

    y_pred_prob = clf.predict_proba(X_test)

    #for fun: train-set predictions

    train_pred = clf.predict(X_train)

    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 

    return y_pred, y_pred_prob

def print_scores(y_test,y_pred,y_pred_prob):

    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 

    print("recall score: ", recall_score(y_test,y_pred))

    print("precision score: ", precision_score(y_test,y_pred))

    print("f1 score: ", f1_score(y_test,y_pred))

    print("accuracy score: ", accuracy_score(y_test,y_pred))

    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

import time
# Case-NB-1 : do not drop anything

drop_list = ["loan_status",*categorical_features]

X_train, X_test, y_train, y_test = split_data(df, drop_list)

start = time.time()

y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

stop = time.time()

print(f"Training time: {stop - start} s")

print_scores(y_test,y_pred,y_pred_prob)