# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





%%time



gh_raw_prefix = 'https://raw.githubusercontent.com/quinn-dougherty/well/master/'



csv_local = ['train_features.csv', 'test_features.csv', 'train_labels.csv', 'sample_submission.csv']

csv_github = {x: gh_raw_prefix + x for x in csv_local}



import pandas as pd

import numpy as np





def cleanski(df): 

    START_ = time()



    df = df.fillna('NOT_KNOWN')    

    boolski = ['public_meeting', 'permit']

    for feat in boolski: 

        df[feat] = df[feat].map({True: 1, False: 0, 'NOT_KNOWN': 0.4})

    to_drop = ['wpt_name', 'region', 'ward', 'scheme_name', 'district_code']

    cats = ['region_district'] + list(df.select_dtypes(include='object').drop(to_drop[:-1], axis=1).columns)

    nums = list(df.select_dtypes(exclude='object').drop(['id', 'district_code'], axis=1).columns)



    insigs = ['funder', 'installer', 'subvillage', 'ward']

    

    def insignificant(features, k=3): 

        cumula = 0

        for feat in features: 

            j = time()

            cumula += j - START_

            valcts = df[feat].str.lower().value_counts()



            df[feat] = [val if valcts[val] > k else "OTHER" for val in df[feat].str.lower()]

            continue

        #print(f'{cumula:.3}')

        pass



    df['date_recorded'] = pd.to_datetime(df['date_recorded']).apply(lambda x: x.toordinal())

    

    assert df.region.isna().sum() == df.district_code.isna().sum() == 0

    df['region_district'] = df.apply(lambda row: f'{row.region}_{row.district_code}', axis=1)

    

    insignificant(insigs)

    

    assert all([x==0 for x in df.isna().sum()])

    return ((df.drop(to_drop, axis=1)

              ), cats, nums)



df_train, cats, nums = cleanski(pd.read_csv(csv_local[0]))

df_test = cleanski(pd.read_csv(csv_local[1]))[0]

test_indices = df_test.id.to_numpy()



target_train = pd.read_csv(csv_local[2])





assert all([x==0 for x in df_test.isna().sum()])



print(df_train.shape)



N = df_train.shape[0]

N_test = df_test.shape[0]



sample_submission = pd.read_csv(csv_local[3])

submit_rows = sample_submission.id.to_numpy()

assert (submit_rows == test_indices).all()
