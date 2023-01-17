import numpy as np
import pandas as pd
import os
data = pd.read_csv("../input/data.csv")
data = data.drop(["Unnamed: 0", "sort_time", "sort_time_watching", "sort_time", "code", "code_watching"], axis=1)#, errors="ignore"
data.shape
# data["action_info_watching"].value_counts().plot("bar")
# data = data[data["action_info_watching"] != 5].append(data[data["action_info_watching"] == 5].sample(1500))
def make_bin(x):
    if x == 5:
        return 1
    return 0

data["action_info_watching"] = data["action_info_watching"].apply(make_bin)
data.head()
# data[data.isna()].sum()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

label_columns = ["material_type", "year", "subject", "education_level", "creation_year", "material_type_watching", "year_watching", "subject_watching", "education_level_watching", "creation_year_watching"]
onehot_columns = ["material_type", "year", "education_level", "creation_year", "material_type_watching", "year_watching", "education_level_watching", "creation_year_watching"]

le = LabelEncoder()

for column in label_columns:
    data[column] = le.fit_transform(data[column])

for column in onehot_columns:
    one_hot = pd.get_dummies(data[column], prefix=column)
    data = pd.concat([data, one_hot], axis=1)
data = data.drop(onehot_columns, axis=1)
# data = data[sorted([str(column) for column in data.columns.tolist()])]
data = data[sorted(data.columns.tolist())]
data.head(10)
data["action_info_watching"].value_counts().plot("bar")
# data["action_info_watching"] = data[data["action_info_watching"] == 0].append(data[data["action_info_watching"] == 1]#.sample(1500))["action_info_watching"]
data = data[sorted(data.columns.tolist())]
import catboost as cb
from sklearn.model_selection import train_test_split

X_train, X_test_val, Y_train, Y_test_val = train_test_split(data.drop(["action_info_watching"], axis=1), data["action_info_watching"], 
                                                            test_size=0.4, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val,
                                                test_size=0.5, random_state=42)

train_pool = cb.Pool(X_train, Y_train)
val_pool = cb.Pool(X_val, Y_val)
model = cb.CatBoostClassifier(
    custom_metric=["F1", "Precision", "Recall"],
#     classes_count=5
)

model.fit(
    train_pool,
    eval_set=val_pool,
    logging_level="Verbose",  # 'Silent', 'Verbose', 'Info', 'Debug'
    early_stopping_rounds=1,
    use_best_model=True,
    plot=True
)
proba = model.predict_proba(X_test)[:, 1]
print(proba)
prediction = model.predict(X_test).reshape(1, -1)[0]
results = pd.DataFrame({"prediction": prediction, "real": Y_test.values, "proba": proba})
results["prediction"] = results["prediction"].astype(int)
results.head(100)
# results[results["real"] != 5].head(100)
from sklearn.metrics import classification_report, f1_score
print(classification_report(Y_test.values, prediction))
from matplotlib import pyplot as plt
plt.xlabel("Features")
plt.ylabel("The score of importance")
info = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
info.plot("bar", title="Features importance", figsize=(20, 20))
print(info.head(15).keys())
model.save_model("model_v01.cbm", format="cbm") 
print(X_train.columns.tolist())
