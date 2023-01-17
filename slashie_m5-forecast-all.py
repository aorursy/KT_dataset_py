import os

import numpy as np 

import pandas as pd
scores_df = []

submit_df = []

main = "/kaggle/input/"

for d in os.listdir(main):

    if d.startswith("m5-forecast-"):

        for f in os.listdir(os.path.join(main,d)):

            if f.startswith("Scores_"):

                scores_df.append(pd.read_csv(os.path.join(main,d,f)))

            elif f.startswith("Submit_"):

                submit_df.append(pd.read_csv(os.path.join(main,d,f)))

            else:

                pass

    

scores_df = pd.concat(scores_df, ignore_index=True)

submit_df = pd.concat(submit_df, ignore_index=True)

submit_df.loc[:,"F1":"F28"] = submit_df.loc[:,"F1":"F28"].applymap(lambda x: np.where(x<0, 0, x))

sub_df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

sub_df = sub_df[['id']].merge(submit_df, on=['id'])

sub_df.to_csv("submission.csv", index=False)