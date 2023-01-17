import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.linear_model import LogisticRegression

%matplotlib inline
data2 = pd.read_csv('../input/culture_data_processed.csv')
data2.info()
for item in data2.columns:

    print(item)
cult_comp_set = ["HIERARCHY","CLAN_CULTURE", "COMP_PARTNER", "ADHOC", "HOLOC", "INNOV_CULTURE"]
def get_var_correlation(data, culture_metrics: set):

    col_set = set(data2.columns)

    in_set = list(col_set.difference(culture_metrics))

    in_set=list(in_set)

    culture_metrics=list(culture_metrics)

    fig = plt.figure(figsize=(6,9))

    sns.heatmap(data2.corr()[culture_metrics].drop(culture_metrics))

    plt.tight_layout()

#     plt.savefig('CORRMAT.JPG')

    plt.show()
get_var_correlation(data2, set(cult_comp_set))
def get_decision_tree(data, target_col):

    cult_comp_set = ["HIERARCHY","CLAN_CULTURE", "COMP_PARTNER", "ADHOC", "HOLOC", "INNOV_CULTURE"]

    data_Y = data[[target_col]]

    data_X = data.drop(cult_comp_set, axis=1)

    clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)

    clf.fit(data_X, data_Y)

    plt.figure(figsize=(9,9))

    plot_tree(clf, feature_names=data_X.columns, class_names=["NO", "YES"], filled=True)

    plt.title(target_col)

    plt.tight_layout()

#     plt.savefig(target_col+"_TREE.JPG")

    plt.show()
for col in cult_comp_set:

    get_decision_tree(data2, col)
def get_weight_graph(data, target_col):

    cult_comp_set = ["HIERARCHY","CLAN_CULTURE", "COMP_PARTNER", "ADHOC", "HOLOC", "INNOV_CULTURE"]

    data_Y = data[[target_col]]

    data_X = data.drop(cult_comp_set, axis=1)

    clf = LogisticRegression(random_state=0)

    clf.fit(data_X, data_Y)

    weight_df = pd.DataFrame({"COL": data_X.columns, "WEIGHT": clf.coef_[0]}) 

    weight_df["RANK"]=weight_df["WEIGHT"].apply(lambda x: np.absolute(x))

    weight_df = weight_df.sort_values(by=["RANK"], ascending=False)

    plt.figure(figsize=(10,6))

    sns.barplot(data = weight_df, x = "COL", y = "WEIGHT")

    plt.xticks(rotation=90)

    plt.title(target_col)

    plt.tight_layout()

#     plt.savefig(target_col+"_LR_WEIGHTS.JPG")

    plt.show()    
for col in cult_comp_set:

    get_weight_graph(data2, col)
sns.heatmap(data2[cult_comp_set].corr())

plt.tight_layout()

plt.savefig('CORR_CULT.JPG')

plt.show()