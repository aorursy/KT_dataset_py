# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train_csv_path = os.path.join("../input","train.csv")

# Any results you write to the current directory are saved as output.
train_table = pd.read_csv(train_csv_path)

cols = list(train_table.head())

id_list = list(train_table["id"])

category_columns = []

binary_columns = []

print("Total columns: {}".format(len(cols)))

print("Total data length: {} ".format(len(id_list)))
print("Target List:")

print("Target : {:.5f} {:.5f} {:.5f} ".format(train_table['target'].mean(),train_table['target'].var(),train_table['target'].min()))
total_category = 0

for col in cols:

    if col=="id" or col=="target":

        continue

    if "cat" in col:

        total_category += 1

        column_data = train_table[col]

        print("{:12} Mean: {:.5f} | Variance: {:.5f}| Max: {:.5f} | Min: {:.5f}".format(col,column_data.mean(),column_data.var(),column_data.max(),column_data.min()))

print("total :{}".format(total_category))
for col in cols:

    if col=="id" or col=="target":

        continue

    if "bin" not in col and "cat" not in col:

        column_data = train_table[col]

        print("{:12} Mean: {:.5f} | Variance: {:.5f}| Max: {:.5f} | Min: {:.5f}".format(col,column_data.mean(),column_data.var(),column_data.max(),column_data.min()))
# run time too long, just look at the result below

# for col in cols:

#     for col_2 in cols:

#         if col != col_2:

#             correlations = train_table[col].corr(train_table[col_2])

#             print("{} X {} : {:.5f}".format(col,col_2,correlations))
train_col = train_table[cols].values

print(train_col)