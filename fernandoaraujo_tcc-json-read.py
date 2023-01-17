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
import json

from pandas.io.json import json_normalize
with open('/kaggle/input/zzzfile/fk_pregao.json') as json_file:

    fk_pregao = json.load(json_file)
fk_pregao_df = pd.DataFrame.from_dict(json_normalize(fk_pregao), orient='columns')
fk_pregao_df.T