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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

print("train data: {} samples and {} features".format(*train_data.shape))

print("test data: {} samples and {} features".format(*test_data.shape))
train_data.head()
test_data.head()
if 'id' in train_data.keys():

    train_data.drop('id',1, inplace=True)

print("train_data now has {} columns".format(train_data.shape[1]))
target = train_data[['loss']]

features = train_data[[f for f in train_data.keys() if f != 'loss']]
target.head()
features.head()
class DataQualityReport(object):    

    '''

    Report about data basic statistics

    '''

    

    CONT_TABLE_COLS = ['count', 'missing', 'cardinality', 'min', '1st_quartile', 

                       'mean', 'median', '3rd_quartile', 'max', 'std_dev']

    CAT_TABLE_COLS  = ['count', 'missing', 'cardinality', 'mode', 'mode_freq', 

                       'mode_percent', '2nd_mode', '2nd_mode_freq', '2nd_mode_percent']

        

    def __init__(self, df, continuous_features, categorical_features):

        '''

        Constructor

        Parameters:

            df - pandas DataFrame object containing the dataset

            categorical_features - the list of feature names of categorical type

            continuous_features - the list of feature names of continuous type

        '''

        self._continuous_features   = continuous_features

        self._categorical_features  = categorical_features



        self.cont_table = pd.DataFrame(columns=DataQualityReport.CONT_TABLE_COLS, index=continuous_features)

        self.cat_table  = pd.DataFrame(columns=DataQualityReport.CAT_TABLE_COLS, index=categorical_features)

        self.cont_table.index.name = 'feature'

        self.cat_table.index.name = 'feature'

        stats = df.describe()

        self._populate_cont_table(df, stats)

        self._populate_cat_table(df, stats)

  

    def _populate_cont_table(self, df, stats):

        for feature in self._continuous_features:

            self.cont_table['count'][feature]           = df[feature].value_counts().sum()

            self.cont_table['missing'][feature]         = df[feature].isnull().sum()

            self.cont_table['cardinality'][feature]     = df[feature].unique().shape[0]

            self.cont_table['min'][feature]             = stats[feature]['min']

            self.cont_table['1st_quartile'][feature]    = stats[feature]['25%']

            self.cont_table['mean'][feature]            = stats[feature]['mean']

            self.cont_table['median'][feature]          = stats[feature]['50%']

            self.cont_table['3rd_quartile'][feature]    = stats[feature]['75%']

            self.cont_table['max'][feature]             = stats[feature]['max']

            self.cont_table['std_dev'][feature]         = stats[feature]['std']

            

            

    def _populate_cat_table(self, df, stats):

        for feature in self._categorical_features:

            self.cat_table['count'][feature]           = df[feature].value_counts().sum()

            self.cat_table['missing'][feature]         = df[feature].isnull().sum()

            self.cat_table['cardinality'][feature]     = df[feature].unique().shape[0]

            vc = df[feature].value_counts()

            self.cat_table['mode'][feature]            = vc.index[0]

            self.cat_table['mode_freq'][feature]       = vc.values[0]

            self.cat_table['mode_percent'][feature]    = float(vc.values[0])/vc.sum()*100

            if vc.shape[0] > 1:

                self.cat_table['2nd_mode'][feature]            = vc.index[1]

                self.cat_table['2nd_mode_freq'][feature]       = vc.values[1]

                self.cat_table['2nd_mode_percent'][feature]    = float(vc.values[1])/vc.sum()*100
target_dqr = DataQualityReport(target, ['loss'], [])

target_dqr.cont_table
cat_feature_names = [name for name in features.keys() if name[0:3] == "cat"]

print(",".join(cat_feature_names))
cont_feature_names = [name for name in features.keys() if name[0:4] == "cont"]

print(",".join(cont_feature_names))
features_dqr = DataQualityReport(features, cont_feature_names, cat_feature_names)
features_dqr.cont_table
features_dqr.cat_table