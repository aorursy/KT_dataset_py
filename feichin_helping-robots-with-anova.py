import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from scipy.stats import f_oneway



X_train = pd.read_csv('../input/X_train.csv')

X_test = pd.read_csv('../input/X_test.csv')

y_train = pd.read_csv('../input/y_train.csv')
X_train.head()
# any null values?

X_train.isnull().sum()
y_train.head()
# what are our surface materials i.e. targets?

np.unique(y_train['surface'])
# encode surface targets

encoder = LabelEncoder()

surfaces = np.unique(y_train['surface'])

y_train['surface'] = encoder.fit_transform(y_train['surface'])



# do we have a strong variation in means across groups?

# let's find out with pairwise t-Tests for groups with same surface

joined = X_train.set_index('series_id').join(

    y_train.set_index('series_id'))



def anova_across_surface(surface, X):

    # helper function to calculate anovas for group samples of surface levels

    records = X[X.loc[:, 'surface']==surface]

    group_nos = np.unique(records.loc[:, 'group_id'])

    anovas = []

    for col in records.columns:

        samples = [list(X[X.loc[:, 'group_id'] == i][col]) for i in group_nos]

        aov = f_oneway(*samples)

        anovas.append(aov[0])

        anovas.append(aov[1])

    return anovas



# calculate all the anovas first

anovas = dict()

for i in range(0, 9):

    # for each surface level

    anovas[i] = anova_across_surface(i, joined)

    

# make nice tables

no_of_columns = 3

new_cols = ['row_id\t', 'measr_no', 'orien_X', 'orient_Y',

       'orient_Z', 'orient_W', 'velocity_X',

       'velocity_Y', 'velocity_Z', 'accel_X',

       'accel_Y', 'accel_Z', 'group_id',

       'surface']

joined.columns = new_cols

line_1 = '\n' + '\t\t|%s' * no_of_columns

line_2 = 'surface\t\t' + '|F\t      p-value\t' * no_of_columns

line_3 = '%12.12s' + '\t|%9.3e   %8.3f' * no_of_columns



for i in range(no_of_columns, len(joined.columns), no_of_columns):

    print(line_1 % tuple(joined.columns[i-no_of_columns:i]))

    print(line_2)

    print('=' * 22 * (no_of_columns + 1))

    for j, surface in zip(range(0,9), encoder.inverse_transform(list(range(0, 9)))):

        row = anovas[j][2*(i-no_of_columns):2*i]

        print(line_3 % tuple([surface] + anovas[j][2*(i-no_of_columns):2*i]))