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

from matplotlib import pyplot as plt

path = '../input/Kannada-MNIST/'

file_name_dict= {'dig':'Dig-MNIST','test':'test','train':'train','sample':'sample_submission'}

df_dict = {key:pd.read_csv(path + value + '.csv') for key, value in file_name_dict.items()}
def visualisation_pipe(df):

    df = df.copy()

    try:

        df = df.drop('label', axis=1)

    except:

        df = df.drop('id', axis=1)

        return df

    return df



def preview_MNIST(df, n_rows=2, n_columns=5):

    df = df.copy()

    df_train_viz = df.pipe(visualisation_pipe)

    fig, ax = plt.subplots(n_rows,n_columns,figsize = (n_columns*3,n_rows*2))

    for row_index in range(0, n_rows*n_columns):

        ax[row_index%n_rows,row_index//n_rows].imshow(df_train_viz.iloc[row_index].values.reshape(28,28),cmap='binary')

    fig.suptitle('Preview of Kanada MNIST')

for key in ['test', 'train', 'dig']:

    preview_MNIST(df_dict[key], 5, 5)