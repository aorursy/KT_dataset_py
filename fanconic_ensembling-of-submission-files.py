import numpy as np

import pandas as pd

import os
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    

    return df

    

test_df = read_testset()
test_df.shape
# EfficientNetB0

y_pred1 = np.load('../input/ensembling-of-models/y_pred1.npy')



# InceptionV3

y_pred2 = np.load('../input/ensembling-of-models/y_pred2.npy')



# EfficientNetB3

y_pred3 = np.load('../input/ensembling-of-models/y_pred3.npy')

y_pred1.shape, y_pred2.shape, y_pred3.shape
y_test = np.mean([y_pred1,y_pred2,y_pred3], axis = 0)



print(y_test.shape)
test_df.iloc[:, :] = y_test



test_df = test_df.stack().reset_index()



test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])



test_df = test_df.drop(["Image", "Diagnosis"], axis=1)



test_df.to_csv('submission.csv', index=False)
test_df.head()