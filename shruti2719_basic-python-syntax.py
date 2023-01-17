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
import numpy as np

import pandas as pd
# how to give multiple assignment
mike,sarah,bob=21,16,13
mike
age,name=22,"shruti"

age
#create 2 variable

age1=12

age2=18
age1+age2
age1-age2
age2%age1
firstName="shruti"

lastName="Nagpurkar"
firstName+ " "+ lastName
"Hi" * 10
sentence="shruti was playing basketball"
sentence[0]
sentence[0:6]

#index starts with 0 and after : end + 1
sentence[:1]
sentence[:-8]
#placeholders in strings

name="jake"

sent="%s is 15 year old"

sent%name

sent%("shruti")

sent="%s %s is the principal"

sent%("prakash","nitin")

sent="%s is %d year old"

sent%("radha",12)