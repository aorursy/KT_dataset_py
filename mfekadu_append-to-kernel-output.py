# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import secrets

import string
IN_CSV_FILENAME = "/kaggle/input/append-to-kernel-output/data.csv"

OUT_CSV_FILENAME = "data.csv"
os.listdir()
def _get_secret_char():

    return secrets.choice(

        string.ascii_letters 

        + string.digits 

        + string.punctuation

    )
def _get_data():

    return [_get_secret_char()]
def _get_df(fname=None):

    if fname == None:

        return pd.DataFrame(_get_data())

    else:

        return pd.read_csv(fname)
def _do_write(in_fname, out_fname):

    df = _get_df(in_fname)

    df.to_csv(out_fname, index=False)
# def _do_read_then_append(fname):

#     df = pd.read_csv(fname)

#     last_val = df.loc[len(df) - 1][0]

#     df.loc[len(df)] = [last_val + 1]

#     df.to_csv(fname, index=False)



def _do_append_on_write_csv(fname):

    df = _get_df()

    df.to_csv(fname, mode='a', header=False, index=False)

    

def _do_append(fname):

    # _do_read_then_append(fname)

    _do_append_on_write_csv(fname)
if OUT_CSV_FILENAME not in os.listdir():

    # copies IN_CSV_FILENAME to OUT_CSV_FILENAME

    _do_write(IN_CSV_FILENAME, OUT_CSV_FILENAME)

    # does the append

    _do_append(OUT_CSV_FILENAME)

else:

    _do_append(OUT_CSV_FILENAME)
df = pd.read_csv(OUT_CSV_FILENAME)

df