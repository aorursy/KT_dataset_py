# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def process_line(line):

    fixed_line = line.replace("[msec],size", "[msec]-size").replace("\n","")

    data = fixed_line

    df = data.split(",")

    return df





def add_row(df, row):

    df.loc[-1] = row  # adding a row

    df.index = df.index + 1  # shifting index

    df = df.sort_index()  # sorting by index

#     df = df.sort_values(by='UUID')

    return df





def read_moriarty_fixed(path):

    main_df = pd.read_csv(path, error_bad_lines=False)  # read the good lines

    new_rows = []

    with open(path, "r") as file:  # read again but only bad lines

        line = file.readline()

        while line != "":

            if line.count(",") == 8:

                new_rows.append(process_line(line))

            line = file.readline()



    # Form a new dataframe

    for row in new_rows:

        main_df = add_row(main_df, row)



    # Re-index the dataframe based on the UUID column

    main_df["UUID"] = pd.to_numeric(main_df["UUID"])

    return main_df.sort_values("UUID").reset_index().drop("index", axis=1)



#     return main_df  # return the fixed one



moriarty_df = read_moriarty_fixed("../input/Moriarty.csv")



moriarty_df
# Check out the balance in the dataset (malicious vs benign)

moriarty_df.groupby("ActionType")["UUID"].nunique()  # the UUID isn't relevant
# Check out the description and the action of the benign and the malicious activity

benign = moriarty_df[moriarty_df["ActionType"] == "benign"][["Details", "Action"]]      # Benign



benign
# Check out the description and the action of the benign and the malicious activity

malicious = moriarty_df[moriarty_df["ActionType"] == "malicious"][["Details", "Action"]]      # malicious



malicious