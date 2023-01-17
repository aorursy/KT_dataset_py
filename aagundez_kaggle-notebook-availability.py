# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
files = glob.glob('../input/*.csv')

frame = pd.DataFrame()

frame_list = []



for csv in files:

    df = pd.read_csv(csv, index_col=None, header=0)

    frame_list.append(df)

    

frame = pd.concat(frame_list)
frame.head()