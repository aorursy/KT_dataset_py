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
df_train = pd.read_csv('../input/train.csv')
df_train_photo_ids = pd.read_csv('../input/train_photo_to_biz_ids.csv')
df_train.head()
df_train_photo_ids.head()
label_notation = {0: 'good_for_lunch',
                  1: 'good_for_dinner',
                  2: 'takes_reservations',
                  3: 'outdoor_seating',
                  4: 'restaurant_is_expensive',
                  5: 'has_alcohol',
                  6: 'has_table_service',
                  7: 'ambience_is_classy',
                  8: 'good_for_kids'}

