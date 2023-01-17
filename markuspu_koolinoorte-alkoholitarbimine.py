# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

pd.set_option('display.max_rows', 20)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"));

df = pd.read_csv("../input/student-mat.csv")

df
df.Walc.plot.hist(subplots=True,bins=5, grid=False, rwidth=0.95, color="Red");

df.Dalc.plot.hist(subplots=True,bins=5, grid=False, rwidth=0.95);
import numpy as np 

import pandas as pd

df = pd.read_csv("../input/student-mat.csv");

df.plot.scatter("Walc","G3",alpha=0.3,color="Red");

df.plot.scatter("Dalc","G3",alpha=0.3);


mehed=df[df["sex"]=="M"]

naised=df[df["sex"]=="F"]

mehed.plot.scatter("studytime","G3",alpha=0.3);

naised.plot.scatter("studytime","G3",alpha=0.3,color="Red");
import numpy as np

import pandas as pd

df = pd.read_csv("../input/student-mat.csv")

filt=df[df["age"]<21]

filt.groupby("age")["absences","G1","G2","G3","Walc","Dalc"].mean().round(2)