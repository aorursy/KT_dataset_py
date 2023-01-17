import pandas as pd

import seaborn as sns



train = pd.read_csv('../input/data_set_ALL_AML_train.csv')



train
train.describe()
def plot_histogram(feature_no):

    if (type(feature_no) is int) and (feature_no >= 1) and (feature_no <= 33):

        s = str(feature_no)

        numeric_var = train[s]

        sns.distplot(numeric_var,kde = False).set_title('Feature '+s)

    else:

        print('No such feature.')
plot_histogram(15)