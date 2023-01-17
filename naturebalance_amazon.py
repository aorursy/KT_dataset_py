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
temp=pd.read_csv('../input/Amazon.csv')
temp.describe()
temp1=temp['brand'].value_counts()
temp1[:10]
temp1=temp.pivot_table(index="brand",values="comment_num",aggfunc=np.sum)
temp2=temp1[temp1["comment_num"]>0]
temp3=temp2.sort_values(by='comment_num',ascending=False)
tempa=temp3
tempa[:10]
temp1=temp[temp["star"]>0]
temp2=temp1.pivot_table(index="brand",values="star")
temp3=temp2.sort_values(by='star',ascending=False)
tempb=temp3
tempb[:10]

BrandStats=pd.merge(tempa,tempb,left_index=True, right_index=True)
BrandStats[:20]
tempa=pd.read_csv('../input/Amazon.csv')
tempb=pd.read_csv('../input/AmazonComment.csv')
tempa=tempa.loc[:,["shop_id","brand"]]
temp=pd.merge(tempb,tempa,on='shop_id',how='left')
import jieba
import jieba.posseg as pseg
import jieba.analyse
temp1=temp[temp["brand"]=="SEPTWOLVES 七匹狼"]
temp2=temp1["comment_text"]
temp3=' '.join(temp2)
tempkey=jieba.analyse.extract_tags(temp3, topK=10,withWeight=False,allowPOS=())
'、'.join(tempkey)

def topN(brand):
    a=temp[temp["brand"]==brand]
    b=a["comment_text"]
    c=' '.join(b)
    d=jieba.analyse.extract_tags(c, topK=10,withWeight=False,allowPOS=())
    e='、'.join(d)
    return e
BrandStats["topWords"]="none"
BrandStats.describe()
for i in range(2395):
    BrandStats.iloc[i,2]=topN(BrandStats.index[i])

BrandStats[100:150]