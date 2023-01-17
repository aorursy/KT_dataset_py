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
temp=pd.read_excel('../input/JD-xiaomi-6.xlsx')

resource=temp
temp.columns
temp.loc[0]
temp.describe()
temp1=temp.pivot_table(index="评价星级",values="点赞数",aggfunc=np.sum)

temptable=temp1
import jieba

import jieba.posseg

import jieba.analyse
temp1=temp[temp["评价星级"]=="star4"]

temp2=temp1["评价内容"]

temp3='。'.join(temp2)
tempkey=jieba.analyse.extract_tags(temp3, topK=10,withWeight=False,allowPOS=())

'、'.join(tempkey)
def topN(arg):

    a=temp[temp["评价星级"]==arg]

    b=a["评价内容"]

    c=' '.join(b)

    d=jieba.analyse.extract_tags(c, topK=10,withWeight=False,allowPOS=())

    e='、'.join(d)

    return e
temptable["TopN"]="none"

temptable
temptable
topN(temptable.index[4])
for i in range(5):

    temptable.iloc[i,1]=topN(temptable.index[i])
temptable