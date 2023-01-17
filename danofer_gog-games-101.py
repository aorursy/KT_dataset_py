# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/games.json')

print(df.shape)
df.head()
df.languages.head(1).values
# normalize MB, GB...  requires func to multiply (not just replac with zeroes , e.g. "2.5 GB")
# df.head().size.str.replace("MB",)
df = df.loc[df.player_rating != -1]  # drop games without (enough) reviews
df.drop("reviews",axis=1).to_csv("gog_game_reviews_112018.csv.gz",index=False,compression="gzip")
