# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import altair as alt

hr_data = pd.read_csv("/kaggle/input/hr-churn-competition/HR_train.csv")
alt .\
  Chart(
    hr_data
  ) .\
  encode(
    alt.Y('count(value):Q', title = 'Frequency'), 
    color = 'Attrition'
  ) .\
  mark_bar() .\
  facet(
    columns = 5
  )
