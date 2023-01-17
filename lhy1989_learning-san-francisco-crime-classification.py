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
%matplotlib inline

import pandas as pd

import numpy as np



## 读csv到DataFrame

trainDF = pd.read_csv('../input/train.csv')



## 使用标准差、均值归一化 

from sklearn import preprocessing

xy_scaler=preprocessing.StandardScaler()

xy_scaler.fit(trainDF[["X","Y"]])

trainDF[["X","Y"]] = xy_scaler.transform(trainDF[["X","Y"]])



## 去除坐标偏移严重的噪声点

trainDF = trainDF[abs(trainDF["Y"])<100]

trainDF.index = range(len(trainDF))





## 投影坐标点 

import matplotlib.pylab as plt 



plt.plot(trainDF["X"],trainDF["Y"],'x') 

plt.show() 



## 按类别分组显示 

from matplotlib.colors import LogNorm 



groups = trainDF.groupby('Category') 



# 定义网格数 

NX = 100 

NY = 100 



# 15*20的画布大小 

plt.figure(figsize=(15,20)) 

ii = 1 

for name, group in groups: 

    # 9行6列的子图 

    plt.subplot(9,6,ii) 

    # 投影到直方图，histo是数量，接着分界区间 

    histo, xedges, yedges = np.histogram2d( 

            np.array(group.X), 

            np.array(group.Y), 

            bins=(NX,NY), 

        )

    myextent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.imshow( 

        histo.T,

        origin='low',

        extent=myextent,

        interpolation='nearest',

        aspect='auto',

        norm=LogNorm()

        )

    plt.title(name,fontsize=8) # 输出子图名称

    # plt.figure(ii)

    # plt.plot(group.X,group.Y,'.')

    ii += 1

del groups

plt.show()


