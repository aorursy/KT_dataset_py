import pandas as pd

import numpy as np

import re



DIR_DATA_A = "../input/ukara-test-phase"

DIR_DATA_B = "../input/ukara-test-phase"

DIR_DATA_FINAL_A = "../input/ukara-training-bi-lstm-with-word2vec-for-data-a"

DIR_DATA_FINAL_B = "../input/ukara-training-bi-lstm-with-word2vec-for-data-b"
!ls "../input/ukara-test-phase"
data_A_train = pd.read_csv("{}/data_train_A.csv".format(DIR_DATA_A))

data_A_dev = pd.read_csv("{}/data_dev_A.csv".format(DIR_DATA_A))

data_A_test = pd.read_csv("{}/data_test_A.csv".format(DIR_DATA_A))



data_B_train = pd.read_csv("{}/data_train_B.csv".format(DIR_DATA_B))

data_B_dev = pd.read_csv("{}/data_dev_B.csv".format(DIR_DATA_B))

data_B_test = pd.read_csv("{}/data_test_B.csv".format(DIR_DATA_B))





df_final_A = pd.read_csv("{}/df_final_A_test.csv".format(DIR_DATA_FINAL_A))

df_dev_A = pd.read_csv("{}/df_final_A_dev.csv".format(DIR_DATA_FINAL_A))

df_cv_A = pd.read_csv("{}/df_final_A_cv.csv".format(DIR_DATA_FINAL_A))



df_final_B = pd.read_csv("{}/df_final_B_test.csv".format(DIR_DATA_FINAL_B))

df_dev_B = pd.read_csv("{}/df_final_B_dev.csv".format(DIR_DATA_FINAL_B))

df_cv_B = pd.read_csv("{}/df_final_B_cv.csv".format(DIR_DATA_FINAL_B))
data_A_train.head()
data_A_dev.head()
data_A_test.head()
df_final_A.head()
df_cv_A.head()
from sklearn.metrics import precision_score, recall_score, f1_score

def lr_cv_threshold(data_train, data_cv, threshold=None):

    if threshold==None:

        print("Threshold is not set. Searching..")

        for thresholdt in range(400, 700, 1):

            cur_bin_pred_cv = [1 if x>=thresholdt/1000 else 0 for x in data_cv]

            f1 = f1_score(data_train, cur_bin_pred_cv)

            precision = precision_score(data_train, cur_bin_pred_cv)

            recall = recall_score(data_train, cur_bin_pred_cv)



            print("score %.3f: %.5f, %.5f, %.5f" % (thresholdt/1000, f1, precision, recall))

        

    else:

        print("Threshold is set. Using {} as threshold.".format(threshold))

        cur_bin_pred_cv = [1 if x>=threshold else 0 for x in data_cv]

        f1 = f1_score(data_train, cur_bin_pred_cv)

        precision = precision_score(data_train, cur_bin_pred_cv)

        recall = recall_score(data_train, cur_bin_pred_cv)

        print("score %.3f: %.5f, %.5f, %.5f" % (threshold, f1, precision, recall))

    
# lr_cv_threshold(data_A_train["LABEL"].values, df_cv_A["LABEL"].values)
print("Cross-validation: ")

lr_cv_threshold(data_A_train["LABEL"].values, df_cv_A["LABEL"].values, threshold=0.5)

print("\nDev Data: ")

lr_cv_threshold(data_A_dev["LABEL"].values, df_dev_A["LABEL"].values, threshold=0.5)

print("\nTest Data: ")

lr_cv_threshold(data_A_test["LABEL"].values, df_final_A["LABEL"].values, threshold=0.5)
# lr_cv_threshold(data_B_train["LABEL"].values, df_cv_B["LABEL"].values)
print("Cross-validation: ")

lr_cv_threshold(data_B_train["LABEL"].values, df_cv_B["LABEL"].values, threshold=0.5)

print("\nDev Data: ")

lr_cv_threshold(data_B_dev["LABEL"].values, df_dev_B["LABEL"].values, threshold=0.5)

print("\nTest Data: ")

lr_cv_threshold(data_B_test["LABEL"].values, df_final_B["LABEL"].values, threshold=0.5)
print("Dev Data Result")

prediction = np.array([])

label = np.array([])



prediction = np.append(prediction, [1 if x>=0.5 else 0 for x in df_dev_A["LABEL"].values])

label = np.append(label, data_A_dev["LABEL"].values)



prediction = np.append(prediction, [1 if x>=0.5 else 0 for x in df_dev_B["LABEL"].values])

label = np.append(label, data_B_dev["LABEL"].values)



f1 = f1_score(label, prediction)

precision = precision_score(label, prediction)

recall = recall_score(label, prediction)



print("F1:\t\t%.5f \nPrecision:\t%.5f \nRecall:\t\t%.5f" % (f1, precision, recall))
print("Test Data Result")

prediction = np.array([])

label = np.array([])



prediction = np.append(prediction, [1 if x>=0.5 else 0 for x in df_final_A["LABEL"].values])

label = np.append(label, data_A_test["LABEL"].values)



prediction = np.append(prediction, [1 if x>=0.5 else 0 for x in df_final_B["LABEL"].values])

label = np.append(label, data_B_test["LABEL"].values)





f1 = f1_score(label, prediction)

precision = precision_score(label, prediction)

recall = recall_score(label, prediction)



print("F1:\t\t%.5f \nPrecision:\t%.5f \nRecall:\t\t%.5f" % (f1, precision, recall))
df_final_A["LABEL"] = df_final_A["LABEL"].apply(lambda x: 1 if x>=500/1000 else 0)

df_final_A.head()
df_final_B["LABEL"] = df_final_B["LABEL"].apply(lambda x: 1 if x>=500/1000 else 0)

df_final_B.head()
df_final = pd.concat([df_final_A, df_final_B])

df_final= df_final.reset_index(drop=True)

print(df_final.shape)

df_final.head()
df_final["LABEL"].value_counts()
df_final.to_json('{}/predictions_test.json'.format("."), orient='records')