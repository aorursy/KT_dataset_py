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
#dummy database

ndf = pd.DataFrame({'col3': [0, 1, 2, 3, 4],
  'col4': ['2017-02-04',
       '2017-02-04',
       '17-02-2004',
       '2017-02-04',
       '2017-02-03'],
  'col5': ['2017-02-04',
       '2017-02-04',
       '17-02-2004',
       '17-02-2014',
       '2017-02-03'],
  'col6': ['Mar-20-2009',
        'Mar 20, 2009',
        'March 20, 2009', 
        'Mar. 20, 2009', 
        'Mar 20 2009']})
ndf

#to convert in common datetime format

mask = ndf.astype(str).apply(lambda x : x.str.match('(((0?[1-9]|1[0-2])((\/)|(-)))?(((0?[1-9]|[1-2][0-9]|3[0-1])((\/)|(-))))((19[0-9][0-9])|(20[0-1]{1}[0-9])|([0-9][0-9]))|((19[0-9][0-9])|(20[0-1]{1}[0-9])))|((0[1-9])|(1[0-9])|(2[0-9])|(3[0-1]))?(\D)?(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)((\s|\.|-)((19[0-9][0-9])|(20[0-9][0-9])))'
).all())

ndf.loc[:,mask] = ndf.loc[:,mask].apply(pd.to_datetime)
         
    
mask1 = ndf.astype(str).apply(lambda x : x.str.match('\d+/\d+(?:/\d+)?|(?:\d+ )?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?(?:-\d+-\d+| \d+(?:th|rd|st|nd)?,? \d+| \d+)|\d{4}'
).all())

ndf.loc[:,mask1] = ndf.loc[:,mask1].apply(pd.to_datetime)    


ndf
