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
data = pd.read_csv('../input/data.csv')
data.describe()
data[:89010]['1960'].value_counts()
data[:1]['1960']
data = data[np.isfinite(data['1960'])]
Arabdata = data[data['Country Name']=='Arab World']
Arabdata
data['Country Name'].value_counts()
Ghanadata = data[data['Country Name']=='Ghana']
Ghanadata