# pip install pandas-profiling
# from pandas_profiling import ProfileReport
# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')
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
from pandas_profiling import ProfileReport
df = pd.read_csv("/kaggle/input/youtube-trending-video-dataset/IN_youtube_trending_data.csv")
df
profile = ProfileReport(df, title="Profiling Report: IN_youtube_trending_data.csv")
profile.to_file(output_file='profile.html')
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if ".csv" in filename:  # Get all .csv files only.
            path = os.path.join(dirname, filename)
            print(os.path.join(dirname, filename))
            df = pd.read_csv(path)   # Read csv file in pandas.
            profile_title = path.split("/")[-1]   # extracting file names only from the file path.
            profile = ProfileReport(df, title=f"Profiling Report: {profile_title}", progress_bar=False)
            profile.to_file(output_file=f"{profile_title.split('.')[0]}.html")
print("Profiles created.")