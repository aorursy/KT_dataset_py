# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/examresult/py_sense.xlsx'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

xls_py_mind = pd.ExcelFile('/kaggle/input/examresult/py_mind.xlsx')

xls_py_science = pd.ExcelFile('/kaggle/input/examresult/py_science.xlsx')

xls_py_sense = pd.ExcelFile('/kaggle/input/examresult/py_sense.xlsx')

xls_py_opinion = pd.ExcelFile('/kaggle/input/examresult/py_opinion.xlsx')



outerIndex_sinif_ismi=["py_mind", "py_mind", "py_mind", "py_mind","py_mind", "py_mind","py_mind", "py_mind","py_mind", "py_mind",\

                       "py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion",\

                       "py_science","py_science","py_science","py_science","py_science","py_science","py_science","py_science",\

                       "py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","cevap_A."]

sheet_to_df_map = {}

for sheet_name in xls_py_mind.sheet_names:

    sheet_to_df_map[sheet_name] = xls_py_mind.parse(sheet_name)

for sheet_name in xls_py_opinion.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_opinion.parse(sheet_name)

for sheet_name in xls_py_science.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_science.parse(sheet_name)

for sheet_name in xls_py_sense.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_sense.parse(sheet_name)

degerler=list(sheet_to_df_map.values())


InnerIndex_isimler=list(sheet_to_df_map.keys())

InnerIndex_isimler.append("Cevap A.")

InnerIndex_isimler.remove("Blad9")

InnerIndex_isimler.remove("Blad11")

hierarch1=list(zip(outerIndex_sinif_ismi,InnerIndex_isimler))

hierarch1=pd.MultiIndex.from_tuples(hierarch1)
sinavsonuclari=[]

kolonlar=[]

for i in range(len(degerler)):

    kolonlar.append(degerler[i].columns)

cevap_anahtari=[]

for i in range(len(degerler)):

        if len(kolonlar[i])==3:

            sinavsonuclari.append( list(degerler[i][kolonlar[0][2]])[:20])

            cevap_anahtari.append( list(degerler[i][kolonlar[0][1]])[:20])

            print(list(degerler[i][kolonlar[0][2]])[:20])

        elif len(kolonlar[i])==2 :

            sinavsonuclari.append( list(degerler[i][kolonlar[i][1]])[:20])

sinavsonuclari.append(cevap_anahtari[0])

df1=pd.DataFrame(sinavsonuclari,hierarch1,columns=[i for i in range(20)])



correct=[]

blank=[]

wrong=[]

for i in  range(len(degerler)-1):

    countc=0

    countb=0

    countw=0

    for j in range(20):

        if str(df1.iloc[i,j])==str("nan"):

            countb +=1

        elif df1.iloc[i,j]==df1.xs("cevap_A.")[j][0]:

            countc+=1

        elif df1.iloc[i,j]!=df1.xs("cevap_A.")[j][0]:

            countw+=1

    correct.append(countc)

    blank.append(countb)

    wrong.append(countw)

df1["True"]=correct

df1["Wrong"]=wrong

df1["Blank"]=blank



print(df1)
print("The number of people Who attend the second Quiz:{}".format(len(InnerIndex_isimler)-1))
print("The most 3 succesful students according to their True answer numbers\n",df1.sort_values("True",ascending=False).nlargest(4,"True")["True"][1:4])

means={}

means["py_mind"]=df1.loc["py_mind"].mean()

means["Py_opinion"]=df1.loc["py_opinion"].mean()

means["py_science"]=df1.loc["py_science"].mean()

means["Py_sense"]=(df1.loc["py_sense"].mean())

print("Mean of the Classes\n",pd.DataFrame(means))
print("the most succesfull class according to their true answers\n",pd.DataFrame(means).T.sort_values("True",ascending=False).nlargest(1,"True"))



print("the most succesful student of the every class\n")

print("Py Mind:",df1.loc["py_mind"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py Opinion:",df1.loc["py_opinion"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py science:",df1.loc["py_science"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py sense:",df1.loc["py_sense"].sort_values("True",ascending=False).nlargest(1, "True")["True"])
