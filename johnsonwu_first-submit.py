# -*- coding: utf-8 -*-

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



app_events = pd.read_csv('../input/app_events.csv', encoding='utf-8')

app_labels = pd.read_csv('../input/app_labels.csv', encoding='utf-8')

events = pd.read_csv('../input/events.csv', encoding='utf-8')

gender_age_train = pd.read_csv('../input/gender_age_train.csv', encoding='utf-8')

gender_age_test = pd.read_csv('../input/gender_age_test.csv', encoding='utf-8')

phone_brand_model = pd.read_csv('../input/phone_brand_device_model.csv', encoding='utf-8')

label_categories = pd.read_csv('../input/label_categories.csv', encoding='utf-8')



print('---------------------app events--------------------')

print(app_events.head(2))

print('---------------------app labels--------------------')

print(app_labels.head(2))

print('---------------------events------------------------')

print(events.head(2))

print('---------------------gender age--------------------')

print(gender_age_train.head(2))

print('---------------------label cate--------------------')

print(label_categories.head(2))

print('--------------------phone brand--------------------')

print(phone_brand_model.head(10))

phone_brand_model = pd.read_csv('../input/phone_brand_device_model.csv', encoding='utf8')

