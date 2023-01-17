import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
testsConducted = pd.read_excel('/kaggle/input/covid19-tests-conducted-by-country/Tests_Conducted.xlsx')

print("No. of Line items: ", testsConducted.shape)

testsConducted.head()
testsConducted.dtypes