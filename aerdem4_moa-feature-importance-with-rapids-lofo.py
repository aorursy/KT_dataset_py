import sys



!pip install lofo-importance

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf

import cuml





df = cudf.read_csv("/kaggle/input/lish-moa/train_features.csv")



features = ["cp_time", "cp_dose"]



for f in features:

    df[f] = cuml.LabelEncoder().fit_transform(df[f])

    

df = df[df["cp_type"] == "trt_cp"]



target_df = cudf.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

targets = [col for col in target_df.columns if col != "sig_id"]



df = df.merge(target_df, on="sig_id")

df.shape
# working on cudf support for LOFO. temporary solution for now:

df = df.to_pandas()
from lofo import LOFOImportance, Dataset, plot_importance

import cuml





importances = dict()



feature_groups = {"gene": df[[col for col in df.columns if col.startswith("g-")]].values,

                  "cell": df[[col for col in df.columns if col.startswith("c-")]].values}



for target in targets[:10]:

    dataset = Dataset(df=df, target=target, features=features, feature_groups=feature_groups)



    lofo_imp = LOFOImportance(dataset, cv=4, scoring="neg_log_loss", model=cuml.LogisticRegression(C=0.01))



    importances[target] = lofo_imp.get_importance()
plot_importance(importances["5-alpha_reductase_inhibitor"], figsize=(8, 6), kind="box")
plot_importance(importances["adrenergic_receptor_agonist"], figsize=(8, 6), kind="box")
plot_importance(importances["11-beta-hsd1_inhibitor"], figsize=(8, 6), kind="default")