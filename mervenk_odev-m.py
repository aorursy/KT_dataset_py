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
xl = pd.ExcelFile('/kaggle/input/examresult/py_mind.xlsx')

xs = xl.sheet_names

df1 = pd.DataFrame()



for i in xs:

    if i=="emrullah":

        df1["dogru_cevap"] = xl.parse(i).iloc[:,1]

        df1[i] = xl.parse(i).iloc[:,2]

    else:

        df1[i] = xl.parse(i).iloc[:,1]

    #print(df1)   

df1.loc[len(df1)] = "py_mind"





xl = pd.ExcelFile('/kaggle/input/examresult/py_sense.xlsx')

xs = xl.sheet_names

df2 = pd.DataFrame()



for i in xs:

    df2[i] = xl.parse(i).iloc[:,1]

df2.loc[len(df2)] = "py_sense"





xl = pd.ExcelFile('/kaggle/input/examresult/py_science.xlsx')

xs = xl.sheet_names

df3 = pd.DataFrame()

#print(xs)

for i in xs[:-1]:

    df3[i] = xl.parse(i).iloc[:,1]

df3.loc[len(df3)] = "py_science"





xl = pd.ExcelFile('/kaggle/input/examresult/py_opinion.xlsx')

xs = xl.sheet_names

df4 = pd.DataFrame()

for i in xs[:-1]:

    df4[i] = xl.parse(i).iloc[:,1]

df4.loc[len(df4)] = "py_opinion"

    

son_list1=pd.concat([df1, df2, df3, df4], axis=1, sort=False)

son_list=son_list1.fillna(0)

sinif_mev=len(son_list.columns)

i=1

d = {}

while i < sinif_mev:

    d_say=0

    y_say=0

    b_say=0

    j=0

    while j < 20:

        

        if son_list.iloc[:,0][j]==son_list.iloc[:,i][j]:

            d_say += 1

            j+=1

        elif son_list.iloc[:,i][j] == 0 :

            b_say += 1

            j+=1

        else:

            y_say += 1

            j+=1

            

  

    d.setdefault(son_list.columns[i],[]).append(d_say) 

    d.setdefault(son_list.columns[i],[]).append(y_say)

    d.setdefault(son_list.columns[i],[]).append(b_say) 

    d.setdefault(son_list.columns[i],[]).append(son_list.iloc[:,i][23])

    i+=1

        
b_list=pd.DataFrame(d, index=['d_say','y_say','b_say','sinif'])

dfObj = b_list.sort_values(by ='d_say', axis=1,ascending=False)

dfObj1 = b_list.sort_values(by ='y_say', axis=1,ascending=False)

dfObj2 = b_list.sort_values(by ='b_say', axis=1,ascending=False)

print("en fazla dogru yapan ogrenciler siralamasi ")

print(dfObj)

print("en fazla yanlis yapan ogrenciler siralamasi ")

print(dfObj1)

print("en fazla bos birakan ogrenciler siralamasi ")

print(dfObj2)
i=0

s = {}

while i < 20:

    a_say=0

    b_say=0

    c_say=0

    d_say=0

    e_say=0

    bos_say=0

    boss_say=0

    dog_say=0

    yan_say=0

    j=1

    while j < sinif_mev:

        

        if son_list.iloc[:,0][i]==son_list.iloc[:,j][i]:

            dog_say += 1

            

        elif son_list.iloc[:,j][i] == 0 :

            boss_say += 1

           

        else:

            yan_say += 1

          

        

        if son_list.iloc[:,j][i]=='A':

            a_say += 1

            j+=1

        elif son_list.iloc[:,j][i]=='B':

            b_say += 1

            j+=1

        elif son_list.iloc[:,j][i]=='C':

            c_say += 1

            j+=1

        elif son_list.iloc[:,j][i]=='D':

            d_say += 1

            j+=1

        elif son_list.iloc[:,j][i]=='E':

            e_say += 1

            j+=1

        else:

            bos_say += 1

            j+=1

            

    soru=i+1

    soru=(str(soru) + "_soru")

    s.setdefault(soru,[]).append(a_say) 

    s.setdefault(soru,[]).append(b_say)

    s.setdefault(soru,[]).append(c_say)

    s.setdefault(soru,[]).append(d_say)

    s.setdefault(soru,[]).append(e_say)

    s.setdefault(soru,[]).append(dog_say)

    s.setdefault(soru,[]).append(yan_say)

    s.setdefault(soru,[]).append(boss_say)

    i+=1

        



s_list=pd.DataFrame(s, index=['a','b','c','d','e','d_say','y_say','b_say'])

dfObj = s_list.sort_values(by ='d_say', axis=1,ascending=False)

dfObj1 = s_list.sort_values(by ='y_say', axis=1,ascending=False)

dfObj2 = s_list.sort_values(by ='b_say', axis=1,ascending=False)

print("En fazla dogru yapilan sorular siralamasi")

print(dfObj)

print("En fazla yanlis yapilan sorular siralalmasi ")

print(dfObj1)

print("En fazla bos birakilan sorular siralamasi")

print(dfObj2)
