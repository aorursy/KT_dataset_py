# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#CLEANING THE DATA
dataa = pd.read_csv("Extra_material_2.csv")
dataa['shop_id'] = dataa.apply(lambda x: int(x['shop_id']), axis = 1)
datab = pd.read_csv("Extra_material_3.csv")
dataa.drop_duplicates(keep='first', inplace=True)
dataa.sort_values('brand', inplace=True)
dataa.reset_index(inplace=True)
datab.drop_duplicates(keep='first', inplace=True)
datab.reset_index(inplace=True)
datab['date_id']=pd.to_datetime(datab['date_id'])
datab = datab[(datab['date_id']>='2019-05-10') & (datab['date_id']<='2019-05-31')]
datab['itemid'] = datab.apply(lambda x: str(int(x['itemid'])), axis = 1)

#CALCULATING GROSS REVENUE FOR EACH PRODUCTS
datab['subtotal']=datab.apply(lambda x: x.amount*x.item_price_usd, axis = 1)
datac = pd.DataFrame({'itemid':datab['itemid'].unique()}, columns=['itemid'])
gross_revenue = []
for x in datac['itemid']:
    gross_revenue.append(datab[datab['itemid']==x]['subtotal'].sum(axis=0))
datac['gross_revenue']=gross_revenue

#IDENTIFYING WHAT 'SHOPID' DOES CERTAIN 'ITEMID' BELONG TO
shopid = []
for x in datac['itemid']:
    shopid.append(datab[datab['itemid']==x]['shopid'].unique()[0])
datac['shopid']=shopid

#IDENTIFYING TOP THREE PRODUCTS ON CERTAIN 'SHOPID'
Answers = []
for x in dataa['brand'].unique():
    list = dataa[(dataa['brand']==x) & (dataa['shop_type']=='Official Shop')]['shop_id']
    product_list = datac[datac['shopid'].isin(list)]
    best3 = product_list.nlargest(3, 'gross_revenue').reset_index()
    if len(best3) == 3:
        Answers.append(x +', '+best3.loc[0,'itemid']+', '+best3.loc[1,'itemid']+', '+best3.loc[2,'itemid'])
    elif len(best3) == 2:
        Answers.append(x +', '+best3.loc[0,'itemid']+', '+best3.loc[1,'itemid'])
    elif len(best3) == 1:
        Answers.append(x +', '+best3.loc[0,'itemid'])
    else :
        Answers.append(x +', N.A')
        
#CREATING SUBMISSION FILE
Answers = pd.DataFrame(Answers, columns=['Answers'])
Index = range(1, 271)
Answers.insert(0, 'Index', Index)
Answers.to_csv("Answers.csv", index = False)