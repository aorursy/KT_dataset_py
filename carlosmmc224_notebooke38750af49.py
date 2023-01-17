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
yee=pd.read_csv("../input/lyrics.csv")

yee.head()
yee.groupby("genre").count().sort_values("song",ascending=False).song
yee_1=yee[yee.song=="ego-remix"]

yee_1.head()





yee_1.lyrics.str.split()
yee[yee.genre=="R&B"].groupby("artist").count().sort_values("song", ascending=False).song.head()