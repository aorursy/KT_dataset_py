import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# pandas settings

pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format

application_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv", nrows=None)

application_test = pd.read_csv("../input/home-credit-default-risk/application_test.csv", nrows=None)
len(application_train), len(application_test)
use_cols = ['SK_ID_CURR',

 'NAME_CONTRACT_TYPE',

 'CODE_GENDER',

 'FLAG_OWN_CAR',

 'FLAG_OWN_REALTY',

 'CNT_CHILDREN',

 'AMT_INCOME_TOTAL',

 'AMT_CREDIT',

 'NAME_INCOME_TYPE',

 'NAME_EDUCATION_TYPE',

 'NAME_HOUSING_TYPE',

 'EXT_SOURCE_1',

 'EXT_SOURCE_2',

 'EXT_SOURCE_3',

]
application_train.head()[use_cols +  ['TARGET']]
application_test.head()[use_cols]
application_train.TARGET.value_counts()