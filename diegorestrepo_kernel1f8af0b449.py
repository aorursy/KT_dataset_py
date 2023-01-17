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

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cv=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv',low_memory=False)
cv.columns
kk=cv[~cv.publish_time.isna()].reset_index(drop=True)
kk[kk.publish_time.str.split('-').str[0]=='2020'].shape
kk.isna()
i=10

#i=cv[cv.sha=='3cdc48bb9e40afd30a59463b7872761a726998c8'].index[0]

sha=cv.loc[i,'sha']

url='/kaggle/input/CORD-19-research-challenge/{}/{}/pdf_json/{}.json'.format(

        cv.loc[i,'full_text_file'],cv.loc[i,'full_text_file'],cv.loc[i,'sha'])
a=pd.read_json(url, orient='index').T
pd.DataFrame( a.body_text.values[0] )[:3]
n=3

txt=''.join( [p+'\n\n' for p in (pd.DataFrame( a.body_text.values[0] )).text ][:n] )

print(txt)
pd.DataFrame.from_dict(a.bib_entries.loc[0]).T