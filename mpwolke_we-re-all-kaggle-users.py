#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://storage.googleapis.com/kaggle-avatars/thumbnails/2080166-kg.png', width=400,height=400)
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
print('Thats all Kagglers. You dont have to upvote because Im going to upvote you all, since you deserved it')