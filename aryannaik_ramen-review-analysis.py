# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import scipy as stats
import seaborn as sns
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
#Convert Ramen CSV file to Panda Dataframe
ramen = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
ramen
#sample of 30 ratngs 
ramen.sample(50)
ramen.sample(50).describe()

#Correlation between style and Stars 
ramen[["Style","Stars"]]
cup_ratings = ramen.loc[ramen["Style"] == "Cup"]
cup_ratings

sns.countplot(np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64)))

