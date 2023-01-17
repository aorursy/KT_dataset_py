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
df = pd.read_csv("../input/movies.csv")
dff = df[df.runtime != 0]
print("Keskmine filmide kestvus:", dff["runtime"].mean())

print("Kõike pikem film kestis:", dff["runtime"].max())

print("Kõige vähem kestvam film on:", dff["runtime"].min())
dff.plot.scatter("runtime", "vote_average", alpha = 0.2)
df.runtime.plot.hist();
aa = dff.assign(release_year=pd.to_datetime(dff.release_date).dt.year)

aa.groupby(["release_year", "runtime"])["vote_average", "vote_count"].mean()
aa[aa["runtime"] == 14]
aa[aa["runtime"] == 338]
print("Action filme:", aa[aa.genres.str.contains("Action") == True].title.count())

print("Seiklusfilme:", aa[aa.genres.str.contains("Adventure") == True].title.count())

print("Komöödiafilme:", aa[aa.genres.str.contains("Comedy") == True].title.count())

print("Draamafilme:", aa[aa.genres.str.contains("Drama") == True].title.count())

print("Perefilme:", aa[aa.genres.str.contains("Family") == True].title.count())

print("Õudusfilme:", aa[aa.genres.str.contains("Horror") == True].title.count())

print("Ulmefilme:", aa[aa.genres.str.contains("Fiction") == True].title.count())

print("USA's valminud filmid: ", aa[aa.production_countries.str.contains("United States of America") == True].title.count())
