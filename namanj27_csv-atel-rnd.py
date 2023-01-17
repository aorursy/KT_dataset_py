# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import re
import pandas as pd

data = pd.read_csv("../input/data1.csv")
data.loc[0]

# dfx = pd.DataFrame({'Text':[1,2,3,4,5]})

# dfx
data
data['Text'][0]
# for i in range(data.shape[0]):

#     d = data['Text'][i]

#     str = ""

#     for line in d:

        

#         line = line.replace('\n','')

#         line = line.replace('\xa0','')



#         str+=line

#     data["Text"].loc[i] = str

#     data["Text"].loc[i] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", data["Text"].loc[i])
import nltk.data

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# print ('\n--\n'.join(tokenizer.tokenize(d)))
filepath = data['number'].tolist()
filepath = [i+".csv" for i in filepath]

    
filepath[9]
for i in range(data.shape[0]):

    d = data['Text'][i]

    new_df = pd.DataFrame(tokenizer.tokenize(d)).stack()

    new_df.to_csv(filepath[i],index = False,header=True)

#     new_df = new_df.reset_index([0, 'number'])

#     new_df.columns = ['EmployeeId', 'City']