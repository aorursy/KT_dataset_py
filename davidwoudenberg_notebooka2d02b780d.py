# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

complete_train_data = pd.read_csv('../input/train.csv', index_col = 0)

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(complete_train_data.head(2))

print(complete_train_data.columns)
complete_train_data.plot(x = "LotArea", y = "SalePrice", kind = 'scatter')
complete_train_data.SalePrice.isnull().sum()

complete_train_data.Utilities.value_counts()
def plot_bar(togroup):

    val_counts = complete_train_data[togroup].value_counts()

    if len(val_counts) > 10:

        return

    groups = complete_train_data.groupby(togroup)

    means = groups.SalePrice.mean()

    error = groups.SalePrice.std()

    print(togroup)

    print(pd.concat([val_counts, means, error], axis = 1))



for col in complete_train_data.columns:

    plot_bar(col)