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
def age (date):

    date = date.split(".")

    age = 2020-int(date[2])

    age_list.append(age)

age_list = []

team = pd.read_excel("/kaggle/input/sper-lig-20192020-players/Futbolcular.xlsx")

team_separete  = team["Doğum Tarihi"].apply(age)



team["age"] = age_list



team_group = team.groupby("Kulüp Adı")["age"].mean()



print(team_group)