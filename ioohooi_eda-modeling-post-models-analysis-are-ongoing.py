import numpy as np



import pandas as pd



import os



import matplotlib_venn



import matplotlib.pyplot as plt



import seaborn as sns



import sklearn



from sklearn.naive_bayes import BernoulliNB



from sklearn.model_selection import cross_validate



from sklearn.model_selection import StratifiedKFold
pd.options.display.max_rows = 999
sns.set(font_scale=2)
! ls -alh ../input
! ls -alh ../input/ieee-fraud-detection/
folder_path = '../input/ieee-fraud-detection/'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

submission = pd.read_csv(f'{folder_path}sample_submission.csv')
print("train_identity shape: ", train_identity.shape)

print("train_transaction shape: ", train_transaction.shape)

print("test_identity shape: ", test_identity.shape)

print("test_transaction shape: ", test_transaction.shape)

print("submission shape: ", submission.shape)
train_identity.head().T
train_transaction.head().T
test_identity.head().T
test_transaction.head().T
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))

matplotlib_venn.venn2(

    subsets=[

        set(train_transaction.TransactionID),

        set(train_identity.TransactionID)

    ],

    set_labels=[

        "Transaction",

        "Identity"

    ],

    ax=ax1

)

matplotlib_venn.venn2(

    subsets=[

        set(test_transaction.TransactionID),

        set(test_identity.TransactionID)

    ],

    set_labels=[

        "Transaction",

        "Identity"

    ],

    ax=ax2

)

ax1.set_title("Train")

ax2.set_title("Test")

plt.suptitle("Intersection of transaction and identity TransactionIDs")

plt.tight_layout()

plt.show()
train_transaction["has_identity_info"] = train_transaction.TransactionID.isin(

    train_identity.TransactionID

)



test_transaction["has_identity_info"] = test_transaction.TransactionID.isin(

    test_identity.TransactionID

)
train_transaction_na_flags = train_transaction[[

        column for column in train_transaction.columns.values if column not in [

            "isFraud", "has_identity_info", "TransactionID", "TransactionDT"

        ]

    ]

].isna()



test_transaction_na_flags = test_transaction[[

        column for column in test_transaction.columns.values if column not in [

            "has_identity_info", "TransactionID", "TransactionDT"

        ]

    ]

].isna()



train_transaction_na_flags["isFraud"] = train_transaction["isFraud"]

train_transaction_na_flags_gr = train_transaction_na_flags.groupby("isFraud").mean() * 100



train_transaction_na_flags["has_identity_info"] = train_transaction["has_identity_info"]

train_transaction_na_flags["TransactionDT"] = train_transaction["TransactionDT"]



test_transaction_na_flags["has_identity_info"] = test_transaction["has_identity_info"]

test_transaction_na_flags["TransactionDT"] = test_transaction["TransactionDT"]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))



sns.countplot(

    y="isFraud",

    hue="has_identity_info",

    data=train_transaction,

    ax=ax1

)



sns.boxplot(

    y="isFraud",

    x="value",

    data=pd.melt(train_transaction_na_flags_gr.T),

    ax=ax2,

    orient="h",

    showmeans=True,

    meanline=True,

    meanprops=dict(linestyle='-.', linewidth=2.5, color='red')

)



# https://matplotlib.org/3.1.0/gallery/statistics/bxp.html



ax1.set_title("The number of transactions without identity info")



ax2.set_title("The distribution of null-values percentages for transactions features")



plt.suptitle("An overview of data completeness for train part of the data")



ax1.grid(True)



ax2.grid(True)



plt.show()
print("Descriptive statistics about null-values in features of transactions:")



print()



print(train_transaction_na_flags_gr.T.describe())



train_transaction_na_flags_gr_t = train_transaction_na_flags_gr.T



print()



print("So what we have is:")



print("\t- 50% of non-fraud train transactions have > 28% of features with null-values;")



print("\t- 50% of fraud train transactions have > 29% of features with null-values;")



print("\t- 25% of non-fraud train transactions have > 78% of features with null-values;")



print("\t- 25% of fraud train transactions have > 50% of features with null-values;")



print(

    "\t- there are {} non-fraud train transactions where > 90% features have null-values;".format(

        train_transaction_na_flags_gr_t[train_transaction_na_flags_gr_t[0] > 90][0].count()

    )

)



print(

    "\t- there are {} fraud train transactions where > 80% features have null-values.".format(

        train_transaction_na_flags_gr_t[train_transaction_na_flags_gr_t[1] > 80][1].count()

    )

)



print("\t- there are some non-fraud train transactions where > 94% features have null-values;")



print("\t- there are some fraud train transactions where > 82% features have null-values.")



print()



print(

    "Percent of non-fraud train transactions with identity info:",

    train_transaction[train_transaction.isFraud == 0].has_identity_info.mean() * 100

)



print()



print(

    "Percent of fraud train transactions with identity info:",

    train_transaction[train_transaction.isFraud == 1].has_identity_info.mean() * 100

)
train_transaction_na_flags_gr_t_m = pd.melt(

    train_transaction_na_flags_gr.T.reset_index(),

    id_vars=['index'],

    value_vars=[0, 1]

).rename(

    {"index": "features", "value": "percentage"},

    axis=1

)



train_transaction_na_flags_gr_t_m.isFraud = train_transaction_na_flags_gr_t_m.isFraud.astype(np.bool)



plt.figure(figsize=(20, 150))



sns.barplot(

    y="features",

    x="percentage",

    hue="isFraud",

    data=train_transaction_na_flags_gr_t_m,

    order=train_transaction_na_flags_gr.T.sort_values(by=0, ascending=False).index,

    orient='h'

)



plt.title("Percentage of null-values in features for train part of the data")



plt.show()
train_transaction_na_flags.sort_values(by="TransactionDT", ascending=True, inplace=True)
sorted(sklearn.metrics.SCORERS.keys())
X, y = train_transaction_na_flags[

    [column for column in train_transaction_na_flags.columns.values if column not in ["isFraud", "TransactionDT"]]

], train_transaction_na_flags["isFraud"]
bernoulli_nb = BernoulliNB()



scores = pd.DataFrame(

    cross_validate(

        bernoulli_nb,

        X,

        y,

        cv=StratifiedKFold(30),

        scoring=["roc_auc", "accuracy", "recall", "precision", "f1"],

        n_jobs=-1,

        verbose=10

    )

)



scores.index = ["Split {}".format(i) for i in range(1, 31)]
plt.figure(figsize=(30, 15))

scores[

    [

        "test_roc_auc",

        "test_accuracy",

        "test_recall",

        "test_precision",

        "test_f1"

    ]

].boxplot()

plt.show()
plt.figure(figsize=(30, 15))

scores[["fit_time", "score_time"]].boxplot()

plt.show()