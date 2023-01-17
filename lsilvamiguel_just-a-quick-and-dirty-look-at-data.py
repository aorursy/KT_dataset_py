# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



print("numpy version: {0}, pandas version: {1}".format(np.__version__,pd.__version__))
data_df = pd.read_csv("../input/database.csv")

data_df.head()
data_df.shape
data_df.isnull().values.any()
# Rename some columns



data_df.rename(columns={'Victim Ethnicity': 'Victim_Ethnicity', 'Perpetrator Race': 'Perpetrator_Race'}, inplace=True)
data_df["Victim_Ethnicity"].unique()
data_df["Perpetrator_Race"].unique()
data_df.Victim_Ethnicity.value_counts().plot(kind='bar')
data_df.Perpetrator_Race.value_counts().plot(kind='bar')