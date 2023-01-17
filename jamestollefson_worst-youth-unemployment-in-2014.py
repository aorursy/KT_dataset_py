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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/API_ILO_country_YU.csv', index_col='Country Name')

data = data.drop('Country Code', 1)

print(data.info())
data.head()

data.columns = data.columns.astype(str)
for column in data:

    info = data[column]

    _ = plt.xlabel('Youth Unemployment (%)')

    _ = plt.ylabel('Number of Countries')

    _ = plt.title('World Youth Unemployment')

    sns.swarmplot(info)

    plt.tight_layout()

    plt.show()
for column in data:

    info = data[column]

    std = np.std(info)

    mean = np.mean(info)

    median = np.median(info)

    print(mean, median, std)
two_std_above = 17.94353 + 2 * 11.52826

print(two_std_above)
high_unemployment = data[data['2014'] >= 41.00005]

print(high_unemployment)
egypt = high_unemployment.loc['Egypt, Arab Rep.']

rate = high_unemployment[['2010', '2011', '2012', '2013', '2014']]

_ = egypt.plot.bar()

_ = plt.xlabel('Year')

_ = plt.ylabel('Youth Percent Unemployment')

_ = plt.title('Egyptian Youth Unemployment 2010 - 2014')

plt.show()