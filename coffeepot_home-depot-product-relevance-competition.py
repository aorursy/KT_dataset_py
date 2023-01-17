import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# read in data

df_attbs = pd.read_csv('../input/attributes.csv')

df_prod_desc = pd.read_csv('../input/product_descriptions.csv')

df_sample = pd.read_csv('../input/sample_submission.csv')

test = pd.read_csv('../input/test.csv', encoding='latin1')

train = pd.read_csv('../input/train.csv', encoding='latin1')



# data heads

df_attbs.head(10)

# df_prod_desc.head(5)



# df_sample.head(5)



# test.head(5)

# train.head(5)
print('df_attbs size: ' + str(df_attbs.shape))

print('df_prod_desc size: ' + str(df_prod_desc.shape))

print('df_sample size: ' + str(df_sample.shape))

print('test size: ' + str(test.shape))

print('train size: ' + str(train.shape))
# look at product_uid dtypes for cleaning

print(df_attbs.dtypes)