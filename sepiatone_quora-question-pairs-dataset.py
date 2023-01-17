import numpy as np

import pandas as pd



# read in the quora question pairs dataset

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data_train.head(2)
data_train.loc[ (data_train["is_duplicate"] == 1).array].head(2)
print("The total number of question pairs in the dataset: {}".format(len(data_train)))

print("Duplicate question pairs: {:.2f} %".format(data_train["is_duplicate"].sum() / len(data_train) * 100))



s_qid = pd.Series(data_train["qid1"].tolist() + data_train["qid2"].tolist())

print("Total number of questions in the training data: {}".format(len(np.unique(s_qid))))

print("Number of questions that appear multiple times: {}".format(np.sum(s_qid.value_counts() > 1)))



# s_questions = pd.Series(data_train["question1"].tolist() + data_train["question2"].tolist())



# # max_len = 0

# # max_len = max(max_len, len(s_questions))

# # print("Max sentence length:", max_len)