# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output



from csv import reader


#CODBDI
CALL=78
PUT=82


def s(a,i,f): return a[i-1:f].strip()
def f(a,i,f): return float(a[i-1:f])/100
def i(a,i,f): return int(a[i-1:f])

tabela=[]
for arquivo in os.listdir('../input')[:1]:    
    with open('../input/'+arquivo,'r') as arquivo_txt:
        for linha in reader(arquivo_txt):
            linha=linha[0]

            TIPREG=i(linha,1,2)
            if TIPREG==0 or TIPREG==99: continue;
            
            DATA=i(linha,3,10)
            CODBDI=i(linha,11,12)
            CODNEG=s(linha,13,24)
            TPMERC=s(linha,25,27)
            NOMRES=s(linha,28,39)
            ESPECI=s(linha,40,49).strip()
            PREABE=f(linha,57,69)
            PREMAX=f(linha,70,82)
            PREMIN=f(linha,83,95)
            PREMED=f(linha,96,108)
            PREULT=f(linha,109,121)
            TOTNEG=i(linha,148,152)
            QUATOT=i(linha,153,170)
            VOLTOT=f(linha,171,188)
            PREEXE=f(linha,189,201)
            INDOPC=i(linha,202,202)
            DATVEN=i(linha,203,210)
            FATCOT=i(linha,211,217)

            #Se for uma opção de compra lista a data, o código de negociação, o preço de exercicio e data de vencimento
            if CODBDI==CALL:
                tabela+=[[DATA, CODNEG,PREULT,PREEXE, DATVEN]]

df=pd.DataFrame(tabela, columns=['DATA', 'CODNEG','PREULT','PREEXE', 'DATVEN'])
print(df)
