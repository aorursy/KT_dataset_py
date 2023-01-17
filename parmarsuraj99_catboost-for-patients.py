import gc, os

import numpy as np

import pandas as pd

from sklearn import preprocessing



from catboost import CatBoostClassifier, Pool
train_df = pd.read_csv("/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv")

test_df = pd.read_csv("/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv")

sample_submission = pd.read_csv("/kaggle/input/av-healthcare-analytics-ii/healthcare/sample_sub.csv")
train_df.info()
train_df.isna().sum()
train_df
for col in train_df.columns:

    print(f"{col}: {train_df[col].unique()}")
label_features = ["Hospital_type_code", "Hospital_region_code", "Department", "Ward_Type", "Ward_Facility_Code",

                 "Type of Admission", "Severity of Illness", "Age"]
label_encoders = dict()

for label_feature in label_features+["Stay"]:

    le = preprocessing.LabelEncoder()

    le.fit(train_df[label_feature].unique())

    label_encoders[label_feature] = le

    

    del le

    gc.collect()
def label_encode(df:pd.DataFrame, label_features:list, encoders:dict):

    

    for i, feature_ in enumerate(label_features):

        print(feature_)

        df[label_features[i]+"_encoded"]=label_encoders[label_features[i]].transform(df[label_features[i]].values)
label_encode(train_df, label_features+["Stay"], label_encoders)

label_encode(test_df, label_features, label_encoders)
train_df.isna().sum()
features = [f for f in train_df.columns if f not in["case_id", "Stay", "Stay_encoded", "patientid", "City_Code_Patient", "Bed Grade"]+label_features]

target = ["Stay_encoded"]
eval_split = train_df.groupby(["Stay_encoded"]).sample(frac=.2)

train_split = train_df.drop(eval_split.index, axis=0)
params = {

    "iterations":1000,

    "task_type":"GPU",

}

clf = CatBoostClassifier(**params)
train_set = Pool(

    data=train_split[features],

    label=train_split[target],

)



eval_set = Pool(

    data=eval_split[features],

    label=eval_split[target]

)
clf.fit(train_set,

       eval_set=eval_set)
preds = clf.predict(test_df[features]).reshape(-1, )
test_stay = label_encoders["Stay"].inverse_transform(preds)
submission_df = pd.DataFrame({"case_id":test_df["case_id"], "Stay":test_stay})
submission_df
submission_df.to_csv("submission.csv")