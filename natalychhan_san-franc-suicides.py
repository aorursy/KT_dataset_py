# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

crimes_train = pd.read_csv('../input/train.csv', encoding='utf8', usecols=['Category', 'Dates'], parse_dates=['Dates'],dayfirst=True, index_col='Dates')

crimes_train_suic = crimes_train[crimes_train['Category'] == "SUICIDE"]
crimes_train_suic['Weekday'] = crimes_train_suic.index.weekday 
crimes_train_suic.index= crimes_train_suic.index.to_period('Y').to_timestamp('Y')
gr_crimes_train_suic= crimes_train_suic.groupby( [ crimes_train_suic.index, "Weekday"] ).count()
gr_crimes_train_suic.reset_index().groupby('Weekday').describe()










