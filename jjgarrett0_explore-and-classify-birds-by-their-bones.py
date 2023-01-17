# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("ggplot")



# Any results you write to the current directory are saved as output.
bird = pd.read_csv(

    "../input/bird.csv", 

    dtype={"id": "str"}

)



bird.dropna(axis=0, how="any", inplace=True)



bird.shape
bird.describe()
size_of_each_group = bird.groupby("type").size()



ax = size_of_each_group.plot(

    kind="bar", 

    title="Number of birds in each ecological group", 

    color="#00304e",

    figsize=((6, 4))

)



ax.set_title("Number of birds in each ecological group", fontsize=10)

_ = ax.set_xlabel("Ecological Group")



for x, y in zip(np.arange(0, len(size_of_each_group)), size_of_each_group):

    _ = ax.annotate("{:d}".format(y), xy=(x-(0.13 if len(str(y)) == 3 else 0.1), y-6), fontsize=10, color="#eeeeee")
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler(with_mean=False) # Do not centralize the features, keep them positive.



bird_raw = bird.copy() # Make a copy of original data.



feature_columns = ['huml', 'humw', 'ulnal', 'ulnaw', 'feml', 'femw', 'tibl', 'tibw', 'tarl', 'tarw'] # numeric feature columns.



bird[feature_columns] = scaler.fit_transform(bird_raw[feature_columns]) # standardlize the numeric features.
corr = bird_raw[feature_columns].corr()



_, ax = plt.subplots(figsize=(8, 7))



# cmap = sns.diverging_palette(220, 10, as_cmap=True)



_ = sns.heatmap(

    corr, 

    cmap=sns.light_palette("#00304e", as_cmap=True), 

    square=True, 

    cbar_kws={"shrink": .9}, 

    ax=ax, 

    annot = True, 

    annot_kws = {"fontsize": 8}

)
_ = sns.pairplot(

    data=bird_raw, 

    kind="scatter", 

    vars=feature_columns, 

    hue="type", 

    diag_kind="hist", 

    palette=sns.color_palette("Set1", n_colors=6, desat=.5)

)
g = sns.FacetGrid(bird_raw, col="type")



for x, y, c in zip(["huml", "ulnal", "feml", "tibl", "tarl"], ["humw", "ulnaw", "femw", "tibw", "tarw"], sns.color_palette("Set1", n_colors=6, desat=.5)):

    _ = g.map(sns.regplot, x, y, scatter_kws={"s": 12, "alpha": 0.4}, line_kws={"lw": 1}, color=c)



_ = g.add_legend()
_, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))



for f, ax in zip(feature_columns, axes.ravel()):

    _ = sns.boxplot(

        data=bird_raw, 

        y=f, 

        x='type', 

        ax=ax, 

        palette=sns.color_palette("Set1", n_colors=6, desat=.5)

    )



_ = axes[3, 1].annotate("No Data", xy=(.4, .48))

_ = axes[3, 2].annotate("No Data", xy=(.4, .48))
limb_hind_ratio = pd.DataFrame(

    {"ratio": (bird_raw.huml + bird_raw.ulnal) / (bird_raw.feml + bird_raw.tibl + bird_raw.tarl), 

     "type": bird_raw.type})



_, axes = plt.subplots(nrows=2, ncols=3,  figsize=(12, 8))



for t, ax, c in zip(limb_hind_ratio.type.unique(), axes.ravel(), sns.color_palette("Set1", n_colors=6, desat=.5)):

    _ = sns.distplot(

        limb_hind_ratio.loc[limb_hind_ratio.type == t, 'ratio'], 

        rug=True, 

        axlabel=t, 

        bins=20, 

        ax=ax, 

        color=c

    )
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(bird[feature_columns])



explained_variance = pd.DataFrame({"evr": pca.explained_variance_ratio_, "evrc": pca.explained_variance_ratio_.cumsum()}, 

                                  index=pd.Index(["pc{:d}".format(i) for i in np.arange(1, len(feature_columns) + 1)], name="principle components"))



_, ax = plt.subplots(figsize=(8, 4))

_ = explained_variance.evrc.plot(kind="line", color="#ee7621", ax=ax, linestyle="-", marker="h")

_ = explained_variance.evr.plot(kind="bar", ax=ax, color="#00304e", alpha=0.8)

_ = ax.set_title("Explained Variance Ratio of Principle Components", fontsize=10)

_ = ax.set_ylim([0.0, 1.1])



for x, y in zip(np.arange(0, len(explained_variance.evrc)), explained_variance.evrc):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.2, y+0.03), fontsize=7)



for x, y in zip(np.arange(1, len(explained_variance.evr)), explained_variance.evr[1:]):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.15, y+0.02), fontsize=7)
pc_df = pd.DataFrame(

    pca.transform(bird[feature_columns]), 

    columns=["pc{:d}".format(i) for i in np.arange(1, len(feature_columns) + 1)]

)



pc_df["type"] = bird.type



_ = sns.pairplot(

    data=pc_df, 

    vars=["pc{:d}".format(i) for i in np.arange(1, 6)], 

    hue="type", 

    diag_kind="kde", 

    palette=sns.color_palette("Set1", n_colors=6, desat=.5)

)
robust = pd.DataFrame({

        "humr": bird_raw.humw / bird_raw.huml, 

        "ulnar": bird_raw.ulnaw / bird_raw.ulnal,

        "femr": bird_raw.femw / bird_raw.feml,

        "tibr": bird_raw.tibw / bird_raw.tibl,

        "tarr": bird_raw.tarw / bird_raw.tarl,

        "type": bird_raw.type}

)



_, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))



for f, ax in zip(["humr", "ulnar", "femr", "tibr", "tarr"], axes.ravel()):

    _ = sns.boxplot(

        data=robust, 

        y=f, 

        x='type', 

        ax=ax, 

        palette=sns.color_palette("Set1", n_colors=6, desat=.5)

    )

    

    if f == "tibr":

        ax.set_ylim((0.0, 0.1))



_ = axes[2, 1].annotate("No Data", xy=(.42, .5), fontsize=8)
bird_extended = pd.concat([bird_raw, robust[["humr", "ulnar", "femr", "tibr", "tarr"]], limb_hind_ratio["ratio"]], axis=1)



feature_columns_extended = ["huml", "humw", "ulnal", "ulnaw", "feml", "femw", "tibl", "tibw", "tarl", "tarw", "humr", "ulnar", "femr", "tibr", "tarr", "ratio"]



bird_extended[feature_columns_extended] = scaler.fit_transform(bird_extended[feature_columns_extended])
from sklearn.feature_selection import chi2



chi2_result = chi2(bird_extended[feature_columns_extended], bird_extended.type)

chi2_result = pd.DataFrame({"feature": feature_columns_extended, "chi2_statics": chi2_result[0], "p_values": chi2_result[1]})

chi2_result.sort_values(by="p_values", ascending=False, inplace=True)

chi2_result.set_index(keys="feature", inplace=True)



ax = chi2_result["p_values"].plot(kind="barh", logx=True, color="#00304e")



_ = ax.annotate("{:3.2f}".format(chi2_result.chi2_statics[chi2_result.shape[0] - 1]), xy=(chi2_result.p_values[chi2_result.shape[0] - 1], len(feature_columns_extended) - 1), xytext=(0, -3), textcoords="offset pixels", fontsize=8, color="#00304e")

for y, x, c in zip(np.arange(0, len(feature_columns_extended) - 1), chi2_result.p_values[:-1], chi2_result.chi2_statics[:-1]):

    _ = ax.annotate("{:3.2f}".format(c), xy=(x, y), xytext=(-35, -3), textcoords="offset pixels", fontsize=8, color="#eeeeee")



_ = ax.set_xlabel("p-value (chi2 value)")

_ = ax.set_title("chi2 values and p-values of features", fontsize=10)
from sklearn.model_selection import train_test_split



train_f, test_f, train_l, test_l = train_test_split(bird_extended[feature_columns_extended], bird_extended.type, train_size=0.6)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV





lr = LogisticRegression()

params = {

    "penalty": ["l1", "l2"],

    "C": [0.1, 1.0, 5.0, 10.0],

    "class_weight": [None, "balanced"],

    #"multi_class": ["ovr", "multinomial"]

}



gs = GridSearchCV(estimator=lr, param_grid=params, scoring="accuracy", cv=5, refit=True)

_ = gs.fit(train_f, train_l)
print(gs.best_params_)
from sklearn.metrics import confusion_matrix, classification_report

predict_l = gs.predict(test_f)



print(classification_report(test_l, predict_l))
cm = confusion_matrix(test_l, predict_l)

_ = sns.heatmap(cm, square = True, xticklabels = ["P", "R", "SO", "SW", "T", "W"], annot = True, annot_kws = {"fontsize": 10}, yticklabels = ["P", "R", "SO", "SW", "T", "W"], cbar = True, cbar_kws={"shrink": 1.0}, cmap=sns.light_palette("#00304e", as_cmap=True)).set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")
from sklearn.metrics import accuracy_score



print("Accuracy: {:.3f}".format(accuracy_score(y_true=test_l, y_pred=predict_l)))
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

_ = sns.heatmap(abs(gs.best_estimator_.coef_), ax=ax, square = True, xticklabels = feature_columns_extended, annot = True, annot_kws = {"fontsize": 10}, yticklabels = gs.best_estimator_.classes_, cbar = True, cbar_kws = {"shrink": .6}, cmap=sns.light_palette("#00304e", as_cmap=True)).set(xlabel = "Features", ylabel = "Ecological Group", title = "Absolute Feature Weights")
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



params = {

    "n_estimators": [5, 10, 20, 50],

    "criterion": ["gini", "entropy"],

    "max_depth": [5, 10, 15],

    "class_weight": [None, "balanced"]

}



rfc_gs = GridSearchCV(estimator=rfc, param_grid=params, scoring="accuracy", cv=5, refit=True)

_ = rfc_gs.fit(train_f, train_l)
predict_l = rfc_gs.predict(test_f)



print(classification_report(test_l, predict_l))
cm = confusion_matrix(test_l, predict_l)

_ = sns.heatmap(cm, square = True, xticklabels = ["P", "R", "SO", "SW", "T", "W"], annot = True, annot_kws = {"fontsize": 8}, yticklabels = ["P", "R", "SO", "SW", "T", "W"], cbar = True, cbar_kws={"shrink": 1.0}, cmap=sns.light_palette("#00304e", as_cmap=True)).set(xlabel = "Predicted Ecological Group", ylabel = "Real Ecological Group", title = "Confusion Matrix")
print("Accuracy: {:.3f}".format(accuracy_score(y_true=test_l, y_pred=predict_l)))
feature_importances = pd.DataFrame(

    {

        "importance": rfc_gs.best_estimator_.feature_importances_

    }, 

    index=pd.Index(feature_columns_extended, name="feature")

).sort_values(by="importance")



ax = feature_importances.plot(kind="barh", color="#00304e", legend=False)



for y, x in zip(np.arange(0, feature_importances.shape[0]), feature_importances.importance):

    _ = ax.annotate("{:.3f}".format(x), xy=(x-0.007, y-0.1), fontsize=8, color="#eeeeee")





_ = ax.set_xlabel("importance")