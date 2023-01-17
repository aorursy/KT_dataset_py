# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



n=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if n<5:

            print(os.path.join(dirname, filename))

        n=n+1



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt



img_array = np.array(Image.open('/kaggle/input/simpsons-challenge-gft/test/108136.jpg'))

plt.imshow(img_array)

df_solution=pd.read_csv('/kaggle/input/simpsons-challenge-gft/challenge_solution_sandbox.txt')

df_solution.columns

header = ["Id", "Category"]

df_solution=df_solution.sort_values(by=['Id'])

df_solution=df_solution.reset_index()

df_solution.to_csv('solution.csv',columns = header,index=False)



# At this point you can see in the right under output your new file