import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
# data visualization imports
import seaborn as sns
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
sf_features_train = pd.read_csv("../input/train.csv")
sf_features_train.head()
sf_features_train.groupby('Category').size().plot(kind='bar')
theft_data = sf_features_train[sf_features_train['Category'] == 'LARCENY/THEFT']
sns.countplot(x='DayOfWeek', data=theft_data)
sf_features_train.groupby('Resolution').size().plot(kind='bar')
juvenile_resolutions = [
    'CLEARED-CONTACT JUVENILE FOR MORE INFO',
    'JUVENILE ADMONISHED',
    'JUVENILE BOOKED',
    'JUVENILE CITED',
    'JUVENILE DIVERTED'
]
criterion = sf_features_train['Resolution'].map(lambda x: x in juvenile_resolutions)
juv_cases = sf_features_train[criterion]
juv_cases.groupby('Category').size().plot(kind='bar')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
