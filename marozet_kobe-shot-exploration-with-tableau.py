# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#I've decided to take a quick look at Kobe shots in Tableau.
#Right now it's just an exploratory part. I will add no leakage models later.
#I've made a few modifications to the tableau part so the year by year of selected features
#can be explored easier
#It might look complex at first but I feel the tradeoff for being able 
#to dig into even a single game can be beneficial
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/KobeBryantShotExplorationforKaggleCompetition/KobeBryantsshotexploration?:embed=true&:display_count=yes&:showVizHome=no', width=750, height=925)
import tensorflow
from tensorflow.contrib import skflow