# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U pip-licenses
!pip-licenses --format=csv --with-system > submission.csv
!pip-licenses --with-system
df = pd.read_csv("submission.csv", delimiter=',')

df.head(3)
total_list = df['License'].sort_values().unique().tolist()

remove_list = ['Artistic License','BSD-like','CC-BY License','CC0',

               'Copyright (C) 2013, Martin Journois',

               'Copyright (c) 2013 Yhat, Inc.',

               'Copyright (c) 2014, Alireza Savand, Contributors',

               'Dual License','Expat License', 'Expat license',

               'GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.',

 'GNU GPL',

 'GNU GPL 2.0',

 'GNU General Public License (GPL)',

 'GNU LGPL',

 'GNU Lesser General Public License (LGPL), Version 3',

 'GNU Lesser General Public License, version 2.1',

 'GNU Lesser General Public License, version 3 or later',

 'GNU/LGPLv3',

 'GPL',

 'GPL-2',

 'GPL-3.0',

 'GPLv2',

 'GPLv3',

 'GPLv3+',

                'LGPL',

 'LGPL 2.1',

 'LGPL+BSD',

 'LGPL-3.0+',

 'LGPL/MIT',

 'LGPLv2.1',

 'LGPLv2.1+',

 'LGPLv3',

 'LGPLv3+',

 'LICENSE.txt',

 'License',

 'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',

 'License :: OSI Approved :: Apache Software License',

   'MPL 2.0',

 'MPL v2',

 'MPL-2.0',

 'MPLv2.0, MIT Licences',

 'Modified BSD',        'NCSA',

 'New BSD',   'OSI Approved',

 'OSI Approved :: BSD License',   'Standard PIL License',  'UNKNOWN', 'WTFPL',

 'ZPL 2.1', 'public domain, Python, 2-Clause BSD, GPL 3 (see COPYING.txt)']

keep_list = list(set(total_list )- set(remove_list))

keep_list

              
df_filtered = df[df['License'].isin(keep_list)]

print(df.shape)

print(df_filtered.shape)

df_filtered.head(3)
df_filtered.to_csv("filtered.csv", index=False)