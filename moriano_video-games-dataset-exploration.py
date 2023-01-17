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
import pandas as pd
raw = pd.read_csv("../input/vgsales.csv")
raw.tail()
set(raw["Publisher"])
set(raw["Platform"])
nes_only = raw[raw["Platform"] == "NES"]
nes_only
nes_only.sort_values(by="Global_Sales", ascending=False)
nes_sales = nes_only[["Name", "Year", "Global_Sales"]]
nes_sales.head()

nes_sales = nes_sales.sort_values(by="Year")
nes_sales.head()
nes_to_plot = nes_sales.groupby(["Year"]).sum()
nes_to_plot
nes_to_plot.plot(kind="bar",
                 figsize=(15, 8), 
                 title="Total of top selling games for NES")
nes_genres_year = nes_only.groupby(["Year", "Genre"]).sum()
nes_genres_year.describe()
nes_genres_year.info()
nes_genres_year = nes_genres_year[["Global_Sales"]]
nes_genres_year.head()
#nes_genres_year.unstack(level=0).plot(kind='bar', subplots=True)
import matplotlib.pyplot as plt
column_limit = 2
fig, axes = plt.subplots(nrows=4, ncols=column_limit+1)
fig.tight_layout()
genres = set(nes_genres_year.index.get_level_values(1))


row = 0
column = 0
for genre in genres:
    
    to_plot = nes_genres_year.xs(genre, level="Genre")
    to_plot.plot(kind="bar", title="Sales per year for genre " + genre, ax=axes[row,column], figsize=(20, 15))
    #print("Row", row, "Column", column)
    if column == column_limit:
        column = -1
        row += 1
    column += 1
    

