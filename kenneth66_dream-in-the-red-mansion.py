import numpy as np

import pandas as pd

import chardet

#定义虚词



function_word_list = ['之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍',

							  '故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀',

							  '吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越',

							  '再', '更', '比', '很', '偏', '别', '好', '可', '便', '就',

							  '但', '儿',                 # 42 个文言虚词

							  '又', '也', '都', '要',      # 高频副词

							  '这', '那', '你', '我', '他' # 高频代词

							  '来', '去', '道', '笑', '说' #高频动词

							  ]



df = pd.DataFrame(columns=[['标题']+function_word_list])



#读取txt编码

f2 = open('/kaggle/input/hlmtxt/hlm.txt','rb')

data = f2.read()

_encoding= chardet.detect(data)

#以对应编码打开并读取全部行

f = open("/kaggle/input/hlmtxt/hlm.txt", 'r', encoding=_encoding['encoding'])

lines = (f.readlines())

line_start , line_num ,chart_num =0 , 0 , 0

#print(lines[1:10])



for special in lines:

    line_num = line_num + 1

    if '第' in special[0:1]  and '回' in special[0:6] :

        df.loc[ chart_num , ['标题']] = lines[ line_start-1]

        for word_analysis in function_word_list :

            if df.loc[ chart_num , [word_analysis] ].isnull:

                df.loc[ chart_num , [word_analysis] ]=str(lines [line_start:line_num ]).count(word_analysis)

            else :df.loc[ chart_num , [word_analysis] ] = df.loc[ chart_num , word_analysis ].values + str(lines [line_start:line_num ]).count(word_analysis)

        line_start=line_num

        chart_num=chart_num+1

df = df.drop(index=0)

print(df)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=2)

model.fit(df.loc[:,function_word_list].values)

all_predictions = model.predict(df.loc[:,function_word_list].values)

print(all_predictions)



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

pca_demon = df.loc[:,function_word_list].values

scaler.fit(pca_demon)

pca=PCA(n_components=3)

newData=pca.fit_transform(pca_demon)



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



fig = plt.figure()

ax = fig.gca(projection='3d')



z= [z[0] for z in newData]

x= [x[1] for x in newData]

y= [y[2] for y in newData]



ax.scatter(x[:74], y[:74], z[:74], color='r')

ax.scatter(x[75:], y[75:], z[75:], color='b')

ax.legend()

plt.show()