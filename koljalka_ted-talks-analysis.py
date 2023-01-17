# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import pandas as pd

df = pd.read_csv("../input/ted_main.csv")
populaarsed = df[['title', 'main_speaker',"speaker_occupation", 'views']].sort_values('views', ascending=False)[:10]

populaarsed
df.languages.plot.hist(bins=70, grid=False, rwidth=0.7)

plt.xlabel("Keelte arv")

plt.ylabel("Episoodide arv")

plt.show()
df.plot.scatter("views", "languages", alpha=0.2);

plt.xlabel("Vaatamiste arv miljonites")

plt.ylabel("Keelte arv")

plt.show()
release_date = pd.to_datetime(df["published_date"],unit='s').dt.year

lf=pd.DataFrame(df.groupby(release_date)["views", "languages", "comments"].mean())

lf