# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotnine import *

import seaborn as sns



from sklearn.metrics import roc_auc_score



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
data = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
data.shape
data.head()
data = data.drop(columns=["Patient ID"])
old_names = ['Patient addmited to intensive care unit (1=yes, 0=no)',

             'Patient addmited to semi-intensive unit (1=yes, 0=no)',

             'Patient addmited to regular ward (1=yes, 0=no)',

             'SARS-Cov-2 exam result']

new_names = ["int_care", "semiint_care", "reg_ward", "infected"]

colname_mapper = dict(zip(old_names, new_names))

data = data.rename(columns=colname_mapper)



data["infected"] = 1*(data["infected"] == "positive")



data[new_names].melt().groupby('variable').value.mean()
df_plot = data.isna().sum().sort_values(ascending=False).reset_index(name="count")#.tail(10)

# df_plot.loc[int(0.1*len(df_plot)):int(0.9*len(df_plot)),'index'] = '.'

df_plot["percent_missing"] = df_plot["count"]/len(data)

# df_plot["label"] = df_plot["percent"]

# df_plot.loc[df_plot.index.values % 10 == 0, "label"] = df_plot["percent"]

# df_plot.loc["index"] = df_plot["index"].apply(lambda x: str(x)[:12] + "...")

df_plot["index"] = pd.Categorical(df_plot["index"], categories=df_plot["index"])
(ggplot(df_plot, aes("index", "percent_missing")) +

    geom_bar(stat="identity", fill="#1B9E77") +

#     geom_text(aes(label="np.round(label,2)", y="percent-0.01"), size=6, color="yellow", ) +

    theme_minimal() +

    coord_flip() +

    theme(axis_text_y=element_text(size=5)) +

    labs(title="Missing value distribution", y="Feature", x="Missing value percentage"))
pd.concat([df_plot.head(10), pd.DataFrame([["..."]*3], columns=df_plot.columns), df_plot.tail(10)])
data = data[df_plot.loc[len(data) - df_plot["count"] > 10, "index"]]
df_plot = data.isna()

percent_na = df_plot.sum()/len(data)

df_plot = df_plot.loc[:,(percent_na < 0.97) & (percent_na > 0.01)]
from scipy.cluster import hierarchy

from scipy.spatial import distance

#increase recursion limit to run clustermap

import sys

sys.setrecursionlimit(10000)



def nans_cluster():

    row_linkage = hierarchy.linkage(

        distance.pdist(df_plot.T), method='complete', metric="hamming")



    col_linkage = hierarchy.linkage(

        distance.pdist(df_plot), method='complete', metric="hamming")



    d = hierarchy.distance.pdist(df_plot)

    col_clusters = pd.Series(hierarchy.fcluster(col_linkage, 5, 'distance'), name="cluster")

    row_clusters = pd.Series(hierarchy.fcluster(row_linkage, 25, 'distance'), name="cluster")

    # colormap = pd.Series(sns.color_palette("Set2", col_clusters.max()+1).as_hex())

    colormap = pd.Series(["yellow", "red", "green", "blue", "black"])

    col_colors = col_clusters.map(colormap)

    col_colors.index = df_plot.index



    row_colors = row_clusters.map(colormap)

    row_colors.index = df_plot.columns



    data["cluster"] = col_clusters



    return sns.clustermap(df_plot.T, row_linkage=row_linkage, col_linkage=col_linkage, 

                   row_colors=row_colors, col_colors=col_colors,

                   method="complete", metric="hamming",

                   figsize=(13, 13))



nans_cluster()
data.groupby("cluster")["infected"].mean()
obj_columns = data.columns[data.dtypes == "object"]

data[obj_columns].head()
obj_description = data[obj_columns].describe().T.sort_values("unique", ascending=False)

obj_description
data["Urine - Leukocytes"].dropna().sort_values()
data["Urine - Leukocytes"] = data["Urine - Leukocytes"].str.replace("<", "1").astype(float)
data["Urine - pH"].dropna()#.astype("float")
data.loc[data["Urine - pH"] == "Não Realizado", "Urine - pH"] = np.nan

data["Urine - pH"] = data["Urine - pH"].astype(float)
data["Strepto A"].drop_duplicates()
data.loc[data["Strepto A"] == "not_done", "Strepto A"] = np.nan
data["Urine - Hemoglobin"].drop_duplicates()
data.loc[data["Urine - Hemoglobin"] == "not_done", "Urine - Hemoglobin"] = np.nan
obj_columns = data.columns[data.dtypes == "object"]

obj_description = data[obj_columns].describe().T.sort_values("unique", ascending=False)

obj_description
cols = obj_description.query("top=='not_detected'").index.values

for c in cols:

    data.loc[data[c]=="not_detected", c] = 0

    data.loc[data[c]=="detected", c] = 1

    data[c] = data[c].astype("float")
cols = obj_description.query("top=='absent'").index.values

for c in cols:

    data.loc[data[c]=="absent", c] = 0

    data.loc[data[c]=="present", c] = 1

    data.loc[data[c]=="not_done", c] = np.nan

    data[c] = data[c].astype("float")
cols = obj_description.query("top=='normal'").index.values

for c in cols:

    data.loc[data[c]=="normal", c] = 1

    data.loc[data[c]=="not_done", c] = np.nan

    data[c] = data[c].astype("float")
cols = obj_description.query("top=='negative'").index.values

for c in cols:

    data.loc[data[c]=="negative", c] = 0

    data.loc[data[c]=="positive", c] = 1

    data[c] = data[c].astype("float")
# data = data.drop(columns=["Urine - Nitrite"])
obj_columns = data.columns[data.dtypes == "object"]

obj_description = data[obj_columns].describe().T.sort_values("unique", ascending=False)

obj_description
categorical_features = obj_description.index.tolist()
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
def fix_non_ascii_space(x):

    """ Fixes non-ascii space characters"""

    return ''.join([i if ord(i) < 128 else ' ' for i in x])



        

data.columns = [fix_non_ascii_space(x) for x in data.columns]

data.columns = data.columns.str.replace(",", "")
X_train, X_test, y_train, y_test = train_test_split(

    data.drop(columns=['int_care', 'semiint_care', 'reg_ward', 'infected']),

    data['infected'],

    test_size=0.02,

    random_state=42)
for c in categorical_features:

    X_train[c] = X_train[c].astype("category")
params = {

    "max_depth": [2, 3, 4, 5, 10],

    "num_leaves": [31, 100],

    "colsample_bytree": [0.6, 0.8, 1.]

}



lgb = LGBMClassifier()



gs = GridSearchCV(lgb, params, scoring=["neg_log_loss", "roc_auc"], cv=5, refit=False)

gs.fit(X_train, y_train)
(pd.DataFrame(gs.cv_results_)

 .sort_values('mean_test_roc_auc')

 [["params", "mean_test_neg_log_loss", "mean_test_roc_auc", "std_test_neg_log_loss"]])
chosen_params = {'colsample_bytree': 0.8, 'max_depth': 3, 'num_leaves': 31}

lgb = LGBMClassifier(**chosen_params)
fitted_columns = X_train.drop(columns=['cluster']).columns.tolist()

lgb.fit(X_train[fitted_columns], y_train) # fitted model not used now, but will be used later
preds = cross_val_predict(lgb, X_train[fitted_columns],

                          y_train, cv=5, method="predict_proba")
X_train['predictions'] = preds[:,1]

X_train['target'] = y_train.astype("str").values
nans_cluster()
df_temp = pd.concat([

    X_train.groupby('cluster').target.apply(lambda x: x.astype(int).mean()),

    X_train.groupby('cluster').apply(lambda x: roc_auc_score(x.target, x.predictions))],

    1)

df_temp.columns = ["precentage_of_infected", "ROC_AUC"]

df_temp
print(ggplot(X_train, aes("predictions")) +

    geom_histogram(bins=10, fill="#008ABC") + 

    scale_y_log10() +

    facet_wrap("~cluster") +

    labs(x="Predicted probability", title="Output probabilities per cluster") +

    theme_538())
(ggplot(X_train, aes("predictions", fill="target")) +

    geom_histogram(position="fill", bins=10) +

    facet_wrap("~cluster") +

    scale_fill_brewer(type="qual", palette="Set1") +

    labs(x="Predicted probability", title="Class separation per cluster") +

    theme_538())
subset_tests = ["Patient age quantile", "Mean platelet volume ", "Hematocrit",

                "Hemoglobin", "Monocytes", "Red blood cell distribution width (RDW)",

                "Platelets", "Mean corpuscular volume (MCV)", "Eosinophils",

                "Mean corpuscular hemoglobin (MCH)", "Basophils", "Leukocytes", 

                "Mean corpuscular hemoglobin concentration (MCHC)", 

                "Red blood Cells", "Lymphocytes"]



chosen_params = {'colsample_bytree': 0.6, 'max_depth': 5, 'num_leaves': 31}

lgb_subset = LGBMClassifier(**chosen_params)



X_subset = X_train[subset_tests].dropna()

y_subset = y_train[X_subset.index]



lgb_subset.fit(X_subset, y_subset) #fit this for later use with shap

X_subset["predictions"] = cross_val_predict(lgb_subset, X_subset,

                                            y_subset, cv=5, method="predict_proba")[:,1]



X_subset["target"] = y_subset.astype(str)



auc = roc_auc_score(X_subset.target, X_subset.predictions)



print(f"====== ROC AUC {np.round(auc, 3)} =======")



print(ggplot(X_subset, aes("predictions")) +

    geom_histogram(bins=10, fill="#008ABC") + 

    scale_y_log10() +

    labs(x="Predicted probability", title="Output probabilities for subset model") +

    theme_538())



(ggplot(X_subset, aes("predictions", fill="target")) +

    geom_histogram(position="fill", bins=10) +

    scale_fill_brewer(type="qual", palette="Set1") +

    labs(x="Predicted probability", title="Class separation for subset model") +

    theme_538())
X_train['non_missing'] = X_train.notna().sum(1)
(ggplot(X_train, aes('non_missing', 'predictions')) +

    geom_point(alpha=0.2) +

    geom_smooth(method='lm') +

    facet_wrap('~cluster'))
def feature_importance(lgb, importance_type="gain", return_df=False):



    importances = lgb.booster_.feature_importance(importance_type='gain')

    names = lgb.booster_.feature_name()

    df = pd.DataFrame({"names": names, importance_type: importances})



    df = df.sort_values(importance_type, ascending=True)

    df = df.tail(20)

    df["names"] = pd.Categorical(df["names"], categories=df["names"])



    if return_df:

        return df

    else: 

        return (ggplot(df, aes("names", importance_type)) +

        geom_bar(stat="identity") +

        coord_flip())
feature_importance(lgb)
def histogram_target(df, variable, target, labels={}):

    df = df.copy()

    

#     if df[variable].nunique() < 25:

#         gg = (ggplot(df, aes(variable, fill=target)) +

#               geom_bar(position="fill"))

#     else:

    df = df[[variable, target]].dropna()

    gg = (ggplot(df, aes(variable, fill=target)) +

          geom_histogram(position="fill"))



    return (

        gg +

        scale_fill_brewer(type="qual", palette="Set1") +

        theme_minimal() +

        labs(title=variable)

    )





features_to_plot = (feature_importance(lgb, return_df=True)

                    .sort_values("gain", ascending=False)

                    .names

                    .str

                    .replace("_", " ")

                    .head(10)

                    .tolist()

                   )



for f in features_to_plot:

    print(histogram_target(X_train, f, "target"))
import shap



shap.initjs()
explainer = shap.TreeExplainer(lgb_subset)

shap_values = explainer.shap_values(X_subset[subset_tests])[1]



for f in subset_tests:

    shap.dependence_plot(f, shap_values, X_subset[subset_tests])
def plot_2feats(df, x, y):

    return(ggplot(df, aes(x, y, color="target")) + 

          geom_point(alpha=0.8) +

          scale_color_brewer(type="qual", palette="Set1") +

          labs(title=f"{x} vs {y}") +

          theme_minimal())



print(plot_2feats(X_train, "Hematocrit", "Leukocytes"))

print(plot_2feats(X_train, "Basophils", "Leukocytes"))

print(plot_2feats(X_train, "Hematocrit", "Platelets"))
X_subset_test = X_test[subset_tests].dropna()

y_subset_test = y_test[X_subset_test.index]
(pd.DataFrame({"target": y_subset_test,

              "predictions": lgb_subset.predict_proba(X_subset_test)[:,1]})

 .reset_index(drop=True)

 .sort_values("predictions"))
i=1
shap_values = explainer.shap_values(X_subset_test)[1]



shap.force_plot(explainer.expected_value[1], shap_values[1,:], X_subset_test.iloc[1,:])
(plot_2feats(X_train, "Eosinophils", "Leukocytes") +

    geom_point(data=X_subset_test.iloc[i:(i+1),:], color="orange", alpha=0.7, size=10) +

    labs(title="Zoom in Subject 1"))
i=0
shap.force_plot(explainer.expected_value[1], shap_values[i,:], X_subset_test.iloc[i,:])
(plot_2feats(X_train, "Mean corpuscular hemoglobin (MCH)", "Mean corpuscular hemoglobin concentration (MCHC)") +

    geom_point(data=X_subset_test.iloc[i:(i+1),:], color="orange", alpha=0.7, size=10) +

    labs(title="Zoom in Subject 2"))
i=4
shap.force_plot(explainer.expected_value[1], shap_values[i,:], X_subset_test.iloc[i,:])
(plot_2feats(X_train, "Eosinophils", "Leukocytes") +

    geom_point(data=X_subset_test.iloc[i:(i+1),:], color="orange", alpha=0.7, size=10) +

    labs(title="Zoom in Subject 3"))
i=7
shap.force_plot(explainer.expected_value[1], shap_values[i,:], X_subset_test.iloc[i,:])
(plot_2feats(X_train, "Eosinophils", "Leukocytes") +

    geom_point(data=X_subset_test.iloc[i:(i+1),:], color="orange", alpha=0.7, size=10) +

    labs(title="Zoom in Subject 4"))