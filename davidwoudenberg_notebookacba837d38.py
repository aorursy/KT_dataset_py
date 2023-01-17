# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



#print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/arrests.csv')

#print(df.head)



# Any results you write to the current directory are saved as output.
print(df.shape)

mex_cols = [col for col in df.columns if "Mexicans Only" in col]

gen_cols = [col for col in df.columns if "All Illegal Immigrants" in col]

years = np.arange(2000,2017)

print(mex_cols)

print(gen_cols)

print(years)
for i in range(0,25):

    if df.Sector[i] == "All":

        continue

    ax = df.loc[i,mex_cols].plot(title = df.Sector[i])

ax.legend()

ax.set_xticklabels([years[i] for i in range(0,len(years)+1,2)], rotation = 90)
def make_plot(border, columns):

    df_this = df.loc[(df.Border == border),["Sector"] + columns]

    df_this.index = df_this.Sector

    ax = df_this.loc[:,columns].T.plot(title = "%s %s" %(border, columns[0][5:]))

    ax.legend(loc='best')

    ax.set_xticklabels([years[i] for i in range(0,len(years)+1,2)])



    

for border in ["Coast","North","Southwest"]:

    for column in [gen_cols, mex_cols]:

        make_plot(border,column)