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
import apache_beam
!pip install apache_beam
import os

import pandas as pd
df = pd.DataFrame(columns=['file_name','Date','observation'])



path = "../input/cityofla/CityofLA/Job Bulletins"










for filename in os.listdir(path):

    with open(os.path.join(path, filename),errors='ignore') as f:

        observation = f.read()



        if "Open Date:" in observation:

            job_bulletin_date = observation.split("Open Date:")[1].split("(")[0].strip()

            current_df = pd.DataFrame({'file_name':filename,'Date':job_bulletin_date,'observation': [observation]})

        df = df.append(current_df, ignore_index=True)
df