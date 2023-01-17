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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#excell dosylarini okumak icin kullandigim kodlar

df1 = pd.read_excel('/kaggle/input/examresult/py_mind.xlsx',sheet_name=None, ignore_index=True)

df2 = pd.read_excel('/kaggle/input/examresult/py_opinion.xlsx',sheet_name=None, ignore_index=True)

df3 = pd.read_excel('/kaggle/input/examresult/py_science.xlsx',sheet_name=None, ignore_index=True)

df4 = pd.read_excel('/kaggle/input/examresult/py_sense.xlsx',sheet_name=None, ignore_index=True)

#sinif listeleri olusturdum

names1 = [list(df1.items())[i][0] for i in range(len(df1))]

names2 = [list(df2.items())[i][0] for i in range(len(df2))]

names3 = [list(df3.items())[i][0] for i in range(len(df3)-1)]

names4 = [list(df4.items())[i][0] for i in range(len(df4))]

keys = dict(zip(range(20),[pd.read_excel('/kaggle/input/examresult/py_mind.xlsx',

                sheet_name = 'emrullah').loc[i,'Cevap A.'] for i in range(20)])) 

print(keys)
fixlist1 = pd.DataFrame(dict(zip(names1,[pd.read_excel('/kaggle/input/examresult/py_mind.xlsx'     ,

                          sheet_name=i, ignore_index=True).loc[:,'ogr.C'] for i in names1])))

#fixlist2 = pd.DataFrame(dict(zip(names2,[pd.read_excel('/kaggle/input/examresult/py_opinion.xlsx'  ,

#                        sheet_name=i, ignore_index=True).iloc[:1] for i in names2])))

fixlist3 = pd.DataFrame(dict(zip(names3,[pd.read_excel('/kaggle/input/examresult/py_science.xlsx'  ,

                         sheet_name=i, ignore_index=True).loc[:,'ogr.C'] for i in names3])))    

fixlist4 = pd.DataFrame(dict(zip(names4,[pd.read_excel('/kaggle/input/examresult/py_sense.xlsx'    ,

                          sheet_name=i, ignore_index=True).loc[:,'ogr.C'] for i in names4])))
def class_add(res,i,clss):

    res = res.set_value(23, i, clss)



for i in names1:

    class_add(fixlist1,i,'py_mind')

#for i in names2:

#    class_add(fixlist2,i,'py_opinion')

for i in names3:

    class_add(fixlist3,i,'py_science')

for i in names4:

    class_add(fixlist4,i,'py_sense')

#print(fixlist1)

#print(fixlist2)

#print(fixlist3)

#print(fixlist4)