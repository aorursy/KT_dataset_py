# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

base_path = '../input'

click_train_path = os.path.join(base_path, 'clicks_train.csv')

click_test_path = os.path.join(base_path, 'clicks_test.csv')

documents_categories_path = os.path.join(base_path, 'documents_categories.csv')

documents_entities_path = os.path.join(base_path, 'documents_entities.csv')

documents_meta_path = os.path.join(base_path, 'documents_meta.csv')

events_path = os.path.join(base_path, 'events.csv')

page_views_sample_path = os.path.join(base_path, 'page_views_sample.csv')

promoted_content_path = os.path.join(base_path, 'promoted_content.csv')

sample_submission_path = os.path.join(base_path, 'sample_submission.csv')



print (click_train_path, os.path.exists(click_train_path))

print (click_test_path, os.path.exists(click_test_path))

print (documents_categories_path, os.path.exists(documents_categories_path))

print (documents_entities_path, os.path.exists(documents_entities_path))

print (documents_meta_path, os.path.exists(documents_meta_path))

print (events_path, os.path.exists(events_path))

print (page_views_sample_path, os.path.exists(page_views_sample_path))

print (promoted_content_path, os.path.exists(promoted_content_path))

print (sample_submission_path, os.path.exists(sample_submission_path))
click_train_df = pd.read_csv(click_train_path)

#### clicked per each user.

unique_display_id = len(click_train_df.display_id.unique())

unique_ad_id = len(click_train_df.ad_id.unique())

ctr = click_train_df.clicked.sum()/click_train_df.shape[0]



print ('unique display id: ', unique_display_id)

print ('unique ad id: ', unique_ad_id)

print ('click sum: ', click_train_df.clicked.sum())

print('ctr:', ctr)
ctr = click_train_df.clicked.sum()/click_train_df.shape[0]

print('ctr:', ctr)