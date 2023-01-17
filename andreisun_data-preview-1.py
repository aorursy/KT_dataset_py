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



path = "../input/"

files = ['documents_categories.csv', 'documents_meta.csv', 'documents_entities.csv', 'promoted_content.csv', 'documents_topics.csv', 'events.csv', 'page_views_sample.csv', 'clicks_train.csv']



for fname in files:

    print('_'*50)

    try:        

        print(fname, ' reading...')

        dtype = None

        if fname == 'promoted_content.csv':

            df = pd.read_csv(path + fname, dtype = {'advertiser_id': np.str})

        elif fname == 'documents_meta.csv':

            df = pd.read_csv(path + fname, parse_dates = ['publish_time'])

        elif fname == 'documents_topics.csv':

            df = pd.read_csv(path + fname, dtype = {'geo_location': np.str})            

        elif fname == 'events.csv':

            df = pd.read_csv(path + fname, dtype = {'platform': np.str})

        else:

            df = pd.read_csv(path + fname)

        print(df.columns)

        print(df.head())

        print(df.describe())

    except:

        print('error in ', fname)