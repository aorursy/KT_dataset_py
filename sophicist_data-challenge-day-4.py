# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



Sales=pd.read_excel('../input/Sales J1.xlsx',sheet_name='Sheet1')



# Any results you write to the current directory are saved as output.
Sales.head()
Sales.columns
DF=pd.DataFrame(Sales['Seller SKU'])

DF['Frequency']=1





DF.head()
DF['Seller SKU'].unique()
products=['c2','c3','F3','H10','H30','HB','otg','Le002','T10','TCL48','F5']

Frequency=[]

def freq(x):

    for i in x:

        g=len(DF[DF['Seller SKU'].str.contains(i)])

        Frequency.append(g)

    return  Frequency

 



#len(freq(products))

b=freq(products)        

g=b[:11]
SalesFQ=pd.DataFrame(g,index=products)

SalesFQ.head(3)

prod=[i for i in range(len(products))]
plt.bar(prod,SalesFQ[0], color='teal',alpha=0.5,align='center')

plt.title('One Month sales of electronic products',color='turquoise')

plt.ylabel('Sale Frequency')

plt.xlabel('Products')

products=['c2','c3','F3','H10','H30','HB','otg','Le002','T10','TCL48','F5']

y_pos = np.arange(len(products))

plt.xticks( y_pos,products)

plt.show()


plt.barh(prod,SalesFQ[0], color='teal',alpha=0.5,align='center')

plt.title('One Month sales of electronic products',color='turquoise')

plt.ylabel('Sale Frequency')

plt.xlabel('Products Sold')

products=['c2','c3','F3','H10','H30','HB','otg','Le002','T10','TCL48','F5']

y_pos = np.arange(len(products))

plt.yticks( y_pos,products)



plt.show()