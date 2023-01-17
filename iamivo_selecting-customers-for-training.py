# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot

pyplot.style.use('ggplot')
dtype = {

    'ncodpers':np.int32,

    'antiguedad':np.object_,

    'fecha_dato':np.object_

}



TARGETS = [

    'ind_ahor_fin_ult1',

    'ind_aval_fin_ult1',

    'ind_cco_fin_ult1',

    'ind_cder_fin_ult1',

    'ind_cno_fin_ult1',

    'ind_ctju_fin_ult1',

    'ind_ctma_fin_ult1',

    'ind_ctop_fin_ult1',

    'ind_ctpp_fin_ult1',

    'ind_deco_fin_ult1',

    'ind_deme_fin_ult1',

    'ind_dela_fin_ult1',

    'ind_ecue_fin_ult1',

    'ind_fond_fin_ult1',

    'ind_hip_fin_ult1',

    'ind_plan_fin_ult1',

    'ind_pres_fin_ult1',

    'ind_reca_fin_ult1',

    'ind_tjcr_fin_ult1',

    'ind_valo_fin_ult1',

    'ind_viv_fin_ult1',

    'ind_nomina_ult1',

    'ind_nom_pens_ult1',

    'ind_recibo_ult1',

]



for target in TARGETS:

    dtype[target] = np.float16



IDX = 'ncodpers'



df = pd.read_csv('../input/train_ver2.csv', usecols=['ncodpers','antiguedad', 'fecha_dato'] + TARGETS, dtype=dtype)

df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
# Let's convert targets and fill missing values

for target in TARGETS:

    df[target] = df[target].fillna(0).astype(np.int8)



df['antiguedad'] = df['antiguedad'].astype(np.float32)
vc_fecha_dato = df.fecha_dato.value_counts().sort_index()
vc_fecha_dato.plot(kind="bar", figsize=(8.5, 3))
def product_list(pdiff):

    # Unstack all the columns to make a Series

    pdiff1 = pdiff.unstack(1)

    # Select those items in the Series where the difference is greater than 0 -> product is added

    pdiff1 = pdiff1[pdiff1 > 0]

    # Reset the index, the index had two columns: the target and ncodpers

    pdiff1 = pdiff1.reset_index()

    # There is no need for the column telling us a product was added since it is constant 1

    del pdiff1[0]

    # Rename the level_0 (target label) to target

    p1 = pdiff1.rename(columns={'level_0':'target'})  

    return p1



def select_customers(df, date1, date2):

    d1 = df.loc[df.fecha_dato == date1, :]

    d2 = df.loc[df.fecha_dato == date2, :]

    d1.set_index(IDX, inplace=True)

    d2.set_index(IDX, inplace=True)



    # Old customers that had history in previous month

    d2a = d2.ix[d1.index] 

    # Make a subtraction to see if there were change in products

    p1 = product_list(d2a[TARGETS] - d1[TARGETS])

    # Flag these guxs as old customers

    p1['flag_new_customer'] = 0



    # New customers: ncodpers ids that are not present in the previous month, 

    # but we have them in th ecurrent month

    d1_idx = set(d1.index)

    d2_idx = set(d2.index)

    # Selecting only ncodpers that are not in d1 (the previous month, but they are in d2)

    d2b = d2.loc[list(d2_idx - d1_idx), TARGETS]

    p2 = product_list(d2b)

    # Flag these guys as new customers

    p2['flag_new_customer'] = 1

    

    # Concat the two dataframes

    p = pd.concat((p1, p2))

    

    # Add a batch id, the current month

    p['batch'] = d2['fecha_dato'].iloc[0]



    print("{:10s} {:10s} {:6d} {:6d} {:6d}".format(date1, date2, p1.shape[0], p2.shape[0], d2.shape[0]))

    

    return p    
import warnings

warnings.filterwarnings('ignore')
dates = vc_fecha_dato.index
p = select_customers(df, dates[0], dates[1])
p.head(10)
ds = [select_customers(df, date1, date2) for date1, date2 in list(zip(dates, dates[1:]))]
p = pd.concat(ds)
p.to_csv('customers-with-products-added.csv', index=False)
p1 = pd.merge(

    p, 

    df[['ncodpers','fecha_dato','antiguedad']], 

    left_on=['ncodpers', 'batch'], 

    right_on=['ncodpers','fecha_dato'],

    how='left'

)
p1.antiguedad.isnull().mean()
ant_median = p1.groupby('fecha_dato').antiguedad.median()

n_new_customers = p1.groupby('fecha_dato').flag_new_customer.sum()
ant_median.plot(kind='bar', figsize=(8.5, 3))
n_new_customers.plot(kind='bar', figsize=(8.5, 3))