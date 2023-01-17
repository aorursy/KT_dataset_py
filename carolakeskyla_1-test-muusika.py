# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline 

pd.set_option('display.max_rows', 20)



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/lyrics.csv")



#Kui palju ja millist infot mul on?

df.info()
z = df.genre.unique()

print(z.size, ", Žanrid: ", z)
#Kui palju esineb igas žanris laule?

df["genre"].value_counts()
#Millistel Rock žanri artistitel on kõige rohkem laule?

artistid = df[df['genre'] == 'Rock']

print(artistid["artist"].value_counts())
#Top 10 aastat laulusõnade rohkuse/ vähesuse järgi_

df["year"].value_counts().sort_values(ascending=False)
#Kui palju laule vahemikus 1980-1995?

#okei = df.year.plot.hist(range=(1980,1995));

lisadega = df.year.plot.hist(range=(1980,1995), rwidth=0.95, color='black');
#Kui paljud laulud sisaldavad/ ei sisalda antud sõnesi?

print((df['lyrics'].str.contains("love")==True).value_counts())

print((df['lyrics'].str.contains("hate")==True).value_counts())



#Sisaldavad mõlemat?

print(((df['lyrics'].str.contains("love")==True)&(df['lyrics'].str.contains("hate")==True)).value_counts())