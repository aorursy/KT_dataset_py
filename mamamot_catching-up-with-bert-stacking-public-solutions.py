import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path



import os

print(os.listdir("../input"))
input_path = Path("../input")

submission_file_name = "submission.csv"

right_answers = pd.read_csv(input_path/"gendered-pronoun-resolution"/"test_stage_1.tsv", sep="\t", index_col="ID")



results = [

    pd.read_csv(input_path/"end2end-coref-resolution-by-attention-rnn"/submission_file_name, index_col="ID"),

    pd.read_csv(input_path/"fastai-awd-lstm-solution-0-71-lb"/submission_file_name, index_col="ID"),

    pd.read_csv(input_path/"coref-by-mlp-cnn-coattention"/submission_file_name, index_col="ID"),

]
stacked = sum(results) / len(results)
stacked.to_csv("submission.csv")