# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import codecs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filename = '../input/globalterrorismdb_0616dist.csv'
#with codecs.decode(filename, encoding='utf-8', errors='ignore') as fdata:
db = pd.read_csv(filename, encoding='ISO-8859-1', usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])
terrorism_in_russia = db[(db.country_txt=='Russia')]
terrorism_in_russia.describe()