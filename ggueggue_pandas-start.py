# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
obj = pd.Series( [4, 7, -5, 3])
obj
obj = pd.Series( [4, 7, -5,1, 3, 3, 3 ])
obj
obj.values
obj.index
obj.dtypes
obj2 = pd.Series( [4, 7, -5, 3] , index =['d','b','a','c'])
obj2
obj2.index
obj2.dtypes
obj
data = { '대남': '1cm' , '민규': '14cm', '청길': '교양동', '현규': '20cm'}
obj3 =pd.Series(data)
obj3
obj3.name = '키'
obj3.index.name = "이름"

obj3
data= { 'name' : ['대남','청길','민규','현규'],
        'tall(cm)'  : [152 , 144, 184 , 192],
        'age' : [ '23', '25', '25', '25']}
df = pd.DataFrame(data)
df.dtypes
df.index.name = '11' #index행추가
df.columns.name = 'info'
df.columns #열 방향 index
df.values #값 얻기
df.index.name = 'num'
df
df2 = pd.DataFrame(data, columns = ['name', 'age', 'tall(cm)', '노답지수'],
                    index=[ 'one', 'two', 'three', 'four'] )
df2

df2.index 
df2.index = [1,2,3,4]
df2
df2.노답지수 = 3
df2.describe() #DataFrame의 계산 가능한 값들에 대해 다양한 계산값을 나타냄
df2['age'] # == df2.age
df2[ [ 'name','age','tall(cm)' ]]
df2['노답지수'] = [0.1, 0.3 , 0.5 , 0.8]
df2
#새로운 열 추가
df2['4번째'] = np.arange(4) #'4번째'index의 값의 범위설정
df2
df2.index = ['one', 'two', 'three', 'four']
# Series추가 원하는 index의 값을 추가가능
val = pd.Series( [-1.2 , 1] , index = ['two','four'])
df2['4번째'] = val

df2 # two , four에 값이 들어간것을 확인
