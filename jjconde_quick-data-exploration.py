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
# Read the data and look at some examples

edu = pd.read_csv('../input/xAPI-Edu-Data.csv')

edu.head()
# Check for the type of each features

edu.info()
# Check for null values

edu.isnull().sum(axis=0)
features = edu.columns.tolist()



# Print the unique values

for feature in features:

    unique_values = map(lambda x: str(x), edu[feature].unique())

    unique_values = ', '.join(unique_values)

    print('{0}: {1}\n'.format(feature, unique_values))
# Look at the frequency for each feature

for feature in features:

    print('{0}:\n {1}\n'.format(feature, dict(edu[feature].value_counts())))