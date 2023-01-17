



get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')





# In[2]:





import pandas as pd

import numpy as np

from datetime import timedelta          



oil =  pd.read_csv('../input/unzipped-oil-csv/oil.csv', parse_dates=['date'])

print(oil.isna().sum())

print(oil.shape)

start_date = oil.date.min()

end_date = oil.date.max()

print(start_date, end_date)

# In[57]:

new_index = pd.date_range(start_date, end_date)



oil.index = pd.DatetimeIndex(oil.date)

oil.shape



oil.isna().sum()

oil = oil.reindex(new_index, fill_value = 0)

oil2 = oil.copy()

oil2.drop('date', axis=1, inplace=True)

oil2.index.name= 'date'

oil3 = oil2.reset_index()





dcoilwtico = oil3.dcoilwtico.copy()

dcoilwtico[0] = 93.14



dcoilwtico2 = dcoilwtico.copy()

dcoilwtico2.name = 'inputed_price'





# In[158]:





check = False

for idx,ele in enumerate(dcoilwtico2):

    if check:

        if (ele == 0 or np.isnan(ele)):

            count = count + 1

        else:

            difference = float(ele) - avg_start_val

            for i in range(0, count):

                dcoilwtico2[start_idx + i] = avg_start_val + (i+1) * float(difference/float(count+1))    

            check = False

            del avg_start_val

            

    elif (ele == 0.0 or np.isnan(ele)):

        count = 1

        avg_start_val = float(dcoilwtico2[idx - 1])

        start_idx = idx

        check = True





# In[159]:





newf = pd.concat([oil3,dcoilwtico2], axis=1)





# In[160]:





newf = newf.drop('dcoilwtico', axis=1)





# In[163]:



print(newf)

#newf.to_csv('/home/ubuntu/projects/grocery/new_oil')





# In[162]:





#plt.plot(newf.imputed_price)


