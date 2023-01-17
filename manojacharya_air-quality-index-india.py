# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! DIR=/kaggle/input/air-quality-index-india; for file in `ls $DIR`; do cp $DIR/$file ~/$file.csv; done
!ls -l ~/*.csv
import pandas as pd

import numpy as np
cheDf=pd.read_csv("~/Chennai_AQI_Dec19_Mar20.csv", index_col=0)

hydDf=pd.read_csv("~/Hyderabad_AQI_Dec19_Mar20.csv", index_col=0)

kolDf=pd.read_csv("~/Kolkata_AQI_Dec19_Mar20.csv", index_col=0)

mumDf=pd.read_csv("~/Mumbai_AQI_Dec19_Mar20.csv", index_col=0)

delDf=pd.read_csv("~/NewDelhi_AQI_Dec19_Mar20.csv", index_col=0)
cheDf

def getDates(df):

    return [(str(row[0])+str(row[1]).zfill(2)+str(row[2]).zfill(2)) for row in np.asarray(df[["Year","Month","Day","Hour"]])]
cheDf["Date"]=getDates(cheDf)

hydDf["Date"]=getDates(hydDf)

kolDf["Date"]=getDates(kolDf)

mumDf["Date"]=getDates(mumDf)

delDf["Date"]=getDates(delDf)
cheDf.columns
che_hour_Df=cheDf.groupby("Hour").mean()

che_hour_Df["Hour"]=che_hour_Df.index



che_hour_Df

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



def plotHist(x,y,df):

    _Df=df.groupby(x).mean()

    _Df[x]=_Df.index

    city=df["Site"][:1][0]

    df=_Df

    objects = df[x]

    y_pos = np.arange(len(objects))

    performance = df[y]



    plt.bar(y_pos, performance, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)

    plt.ylabel(y)

    plt.xlabel(x)

    plt.title(y+ "-" +str(city))

    plt.show()
plotHist("Hour","AQI",cheDf)

plotHist("Hour","AQI",kolDf)

plotHist("Hour","AQI",hydDf)



plotHist("Month","AQI",cheDf)

plotHist("Month","AQI",kolDf)

plotHist("Month","AQI",hydDf)

def plotSactterPlot(x,y,df):

    city=df["Site"][0:1][0]

    X = df[x]

    Y = df[y]

    #colors = (0,0,0)

    area = np.pi*3



    # Plot

    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)

    plt.scatter(X, Y, alpha=0.5)

    plt.title('Scatter plot:'+x+" vs "+y+" for "+ str(city))

    plt.xlabel(x)

    plt.ylabel(y)

    plt.show()
plotSactterPlot("NowCast Conc.","Raw Conc.",hydDf)

plotSactterPlot("NowCast Conc.","Raw Conc.",cheDf)

plotSactterPlot("NowCast Conc.","Raw Conc.",kolDf)

plotSactterPlot("NowCast Conc.","Raw Conc.",delDf)

hydDf
def plotLinePlot(x,y,df):

    city=df["Site"][0:1][0]

    X = df[x]

    Y = df[y]

    #colors = (0,0,0)

    area = np.pi*3



    # Plot

    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)

    plt.plot(X, df[y], linestyle='solid')

    plt.plot(X, df[y], linestyle='solid')

    plt.title('Line plot:'+x+" vs "+y+" for "+ str(city))

    plt.xlabel(x)

    plt.ylabel(y)

    plt.show()

plotLinePlot("Date","Raw Conc.",hydDf)

plotLinePlot("Date","NowCast Conc.",hydDf)
