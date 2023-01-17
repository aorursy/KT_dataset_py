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
survey_results_df = pd.read_csv("/kaggle/input/stack-overflow-developer-survey-2020/developer_survey_2020/survey_results_public.csv")

survey_results_df.info()
from collections import Counter



lang_counts = Counter()

for langs in survey_results_df["LanguageWorkedWith"].dropna():

    for lang in langs.split(";"):

        lang_counts[lang] += 1
lang_counts["JavaScript"] / lang_counts["Haskell"]