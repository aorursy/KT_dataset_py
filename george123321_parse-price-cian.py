import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





df=pd.read_excel('../input/cian_sale_cost.xlsx')





def get(x):

    

    x=x.split(' ')

    a=x[1]

    dm={'млн.':10**6, 'млрд.':10**9, 'тыс.':10**3}

    

    pr=float(x[0].replace(',','.'))

    pr=pr*dm[a]

    

    return {'price':pr}



df0=pd.DataFrame(df.HEADER.map(lambda x: get(x) ).tolist(), index=df.index)    

df=df0.join(df) 

df