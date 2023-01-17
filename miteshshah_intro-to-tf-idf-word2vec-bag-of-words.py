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
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')
# Filtering the rows to ignore rating = 3 (Neutral Rating)
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""",conn)
filtered_data
def partition(x):
    if x < 3:
        return "Negative"
    return "Positive"
actual_score = filtered_data["Score"]
positive_negative = actual_score.apply(partition)
filtered_data["Score"] = positive_negative
filtered_data
display = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 AND UserId='AR5J8UI46CURR' ORDER BY ProductId""",conn)
display
# Sorting the data according to the product id
# Parameters for sort_values are self-explanatory. Or can look at the documentation in pandas as well
sorted_data = filtered_data.sort_values("ProductId",axis=0,ascending=True,inplace=False,kind="quicksort",na_position="last")
# Deduplication of the Entries
final = sorted_data.drop_duplicates(subset=["UserId","ProfileName","Time","Text"],keep="first",inplace=False)
final
# Example of the Range Constraints Issue
display = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3 AND Id = 44737 OR Id = 64422 ORDER BY ProductId""",conn)
display
final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]
final
final['Score'].value_counts()

