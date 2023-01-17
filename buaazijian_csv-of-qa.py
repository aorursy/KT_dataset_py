# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from pathlib import Path

import pandas as pd

import numpy as np

from tqdm import tqdm

pred_df = pd.read_csv("../input/csv-tijiao/submission_1_10_4.csv")



sample_csv = "../input/google-quest-challenge/sample_submission.csv"

sample_df = pd.read_csv(sample_csv)

n=0

for line in tqdm(pred_df.values):

    for i in range(len(sample_df)):

        if(sample_df.loc[i]['qa_id']==int(line[0])):

            for j in range(1,31):

                sample_df.iloc[i,j] = line[j]

            break

sample_df.head()

sample_df.to_csv("submission.csv", index=False)

print("done!")

print(sample_df.head())