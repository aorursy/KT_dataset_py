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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/hmda_2017_il_all-records_labels.csv")
df.columns
dataf = df[['loan_type_name','loan_purpose_name','applicant_ethnicity_name',"applicant_race_name_1",'applicant_income_000s','action_taken_name']]
dataf.columns
dataf.applicant_race_name_1.value_counts()
ax = sns.countplot(y=dataf.applicant_race_name_1)
new_df = dataf[dataf["applicant_race_name_1"] != "Not applicable"]

new_df.head()
new_df.columns
new_df["action_taken_name"].unique()
new_df["Decision"] = new_df["action_taken_name"].apply(lambda x: "Approved" if x == 'Loan originated' else "Not Approved")
new_df[["Decision","action_taken_name"]].head()
sns.countplot(data = new_df, y = "applicant_race_name_1",hue = "Decision")