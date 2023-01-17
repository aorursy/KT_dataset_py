# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data_read=pd.read_csv('../input/youtube-new/CAvideos.csv')

# Any results you write to the current directory are saved as output.
print(data_read.head())
