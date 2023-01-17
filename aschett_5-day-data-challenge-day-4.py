# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sets = pd.read_csv('../input/sets.csv')

themes = pd.read_csv('../input/themes.csv')

# Practice checking references in other tables

for i in range(20):

    print(sets['name'][i], '->', themes['name'][sets['theme_id'][i]])
# This is perhaps not the most useful visualization...

sns.countplot(sets['theme_id']).set_title('Theme frequencies')
# This either...I'll work on it...what is this?

sns.swarmplot(sets['theme_id']).set_title('Theme frequencies')