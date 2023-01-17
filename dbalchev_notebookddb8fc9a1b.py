# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
hr_data = pd.read_csv("../input/HR_comma_sep.csv")

hr_data.head()
hr_data.sales = hr_data.sales.astype("category")

hr_data.salary = hr_data.salary.astype("category")
left_data = hr_data[hr_data.left == 1]

not_left_data = hr_data[hr_data.left == 0]
plot_columns = hr_data.columns[:-2]



fig, axes = plt.subplots(nrows=len(plot_columns), ncols=3, figsize=(15,15))

for col_index, col in enumerate(plot_columns):

    other_args = {

        "label": col

    }

    hr_data[col].plot.hist(ax=axes[col_index, 0], **other_args)

    left_data[col].plot.hist(ax=axes[col_index, 1], **other_args)

    axes[col_index, 1].set_label(col)

    not_left_data[col].plot.hist(ax=axes[col_index, 2], **other_args)

    
hr_data.sales.cat.codes