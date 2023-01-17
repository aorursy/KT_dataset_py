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

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



cf.go_offline()



import plotly.express as px



from scipy.stats import linregress







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# List arguments in wide form

series1 = [3, 5, 4, 8]

series2 = [5, 4, 8, 3]

fig = px.line(x=[1, 2, 3, 4], y=[series1, series2])

fig.show()
def split(s):

    x=[]

    for i in range(len(s)):

        x+=[s[i]]

    return x

split("ABCD")
score = pd.read_excel("../input/dataset/DP1.xls",sheet_name="BLOSUM50")

score.index=score["Unnamed: 0"]

score.drop("Unnamed: 0",axis=1)
a=np.zeros(shape=(4,4))

df1=pd.DataFrame(a,columns=["-"]+["A","B","C"],index=["-"]+["A","B","C"])

df1["A"][0]=100

def maximo(x):

    a=df1.iloc[0,0]

    for i in range(len(x.index)):

        j=0

        for j in range(len(x.columns)):

            if x.iloc[i,j]>a:

                a=df1.iloc[i,j]

    return a



def posicoes(x):

    b=[]

    for i in range(len(x.index)):

        j=0

        for j in range(len(x.columns)):

            if x.iloc[i,j]==maximo(x):

                b+=[[i,j]]

    return b
posicoes(df1)
def algoritmo(x):

    a=[]

    for i in posicoes(x):

        r=""

        l=""

        b=maximo(x)



            
a=[posicoes(df1)[0][0]-1,posicoes(df1)[0][1]]

c=[posicoes(df1)[0][0],posicoes(df1)[0][1]-1]

m=[posicoes(df1)[0][0]-1,posicoes(df1)[0][1]-1]

df1.iloc[a[0],a[1]]



df1.index[2]
x=[1,2,3,4,5]

y=[1,4,9,16,25]

slope, intercept, r_value, p_value, std_err = linregress(x, y)

slope
from scipy.optimize import curve_fit


def fun(x):

    return x**2
x=[1,2,3,4,5]

y=fun(x)