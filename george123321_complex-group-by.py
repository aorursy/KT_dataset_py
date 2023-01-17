import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





f=pd.read_excel('../input/address.xlsx')

print(f)


fnn=pd.DataFrame(columns=['клиент'])



 

cc=['адрес','tel']



for i in cc:

    fn=f.groupby('клиент')[i].apply(lambda x : 

        

        x.iloc[np.argmax(x.map(lambda y: len(str(y)) ).tolist())] 

        

        ).reset_index()

        

    fnn=fn.merge(fnn,on='клиент',how='outer')

    



    

fnn.to_excel('full.xlsx',index=False)


