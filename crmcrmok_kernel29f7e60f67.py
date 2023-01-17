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
usecols = ['fecha_dato','ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',

       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',

       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',

       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
# all_days = [3,2,1]
# mapk_avg = 0
# num_features = ['age','antiguedad','renta']

# cat_features = ['segmento','tiprel_1mes']

# prod_features = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',

#                     'ind_ctju_fin_ult1','ind_ctma_fin_ult1', 'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',

#                     'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',

#                     'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',

#                     'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_ctop_fin_ult1', 'ind_ecue_fin_ult1', 'ind_cno_fin_ult1', 

#                    'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 

#                    'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 

#                    'ind_ctju_fin_ult1', 'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_pres_fin_ult1', 'ind_deme_fin_ult1', 

#                    'ind_cder_fin_ult1', 'ind_deco_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
#     for n_days in all_days:

    

#         print('\nLoading and transforming sample...')
x_train = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv.zip', encoding='utf8')
# !unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip
# !pwd
# !ls
# x = pd.read_csv('train_ver2.csv', encoding='utf8')
# !unzip /kaggle/input/santander-product-recommendation/test_ver2.csv.zip
# x_test = pd.read_csv('test_ver2.csv', encoding='utf8')
# x = pd.concat([x_train,x_test])
# del x_train,x_test
# x.head()
# x.sort_values(['ncodpers','fecha_dato'],inplace=True)
# x.reset_index(inplace=True,drop=True)
# startshape = x.shape[0]
# print('Start shape',startshape)
# x['n_nan'] = x.isnull().sum(1)
# dftr_prev = x[x['fecha_dato'].isin(['28-04-2016'])]
# dftr_curr = x[x['fecha_dato'].isin(['28-05-2016'])]
# x = pd.concat([dftr_prev,dftr_curr])
# x = pd.read_csv('train_ver2.csv', parse_dates=['fecha_dato'], infer_datetime_format=True)
# dftr_prev = x[x['fecha_dato'].isin(['28-04-2016'])]
# dftr_curr = x[x['fecha_dato'].isin(['28-05-2016'])]
# dftr_prev.drop(['fecha_dato'], axis=1, inplace = True)

# dftr_curr.drop(['fecha_dato'], axis=1, inplace = True)
# dfm = pd.merge(dftr_curr,dftr_prev, how='inner', on=['ncodpers'], suffixes=('', '_prev'))
# prevcols = [col for col in dfm.columns if '_ult1_prev' in col]

# currcols = [col for col in dfm.columns if '_ult1' in col and '_ult1_prev' not in col]
l = [1375586,

 1050611,

 1050612,

 1050613,

 1050614,

 1050615,

 1050616,

 1050617,

 1050619,

 1050620,

 1050621,

 1050622,

 1050623,

 1050624,

 1050625,

 1050626,

 1050610,

 1050627,

 1050609,

 1050605,

 1050582,

 1050586,

 1050588,

 1050589,

 1050591,

 1050592,

 1050595,

 1050596,

 1050597,

 1050598,

 1050599,

 1050601,

 1050602,

 1050603,

 1050604,

 1050607,

 1050580,

 1050628,

 1050630,

 1050669,

 1050670,

 1050676,

 1050679,

 1050680,

 1050686,

 1050688,

 1050693,

 1050694,

 1050697,

 1050703,

 1050704,

 1050706,

 1050707,

 1050710,

 1050663,

 1050629,

 1050662,

 1050660,

 1050632,

 1050633,

 1050634,

 1050635,

 1050636,

 1050639,

 1050641,

 1050642,

 1050647,

 1050648,

 1050651,

 1050652,

 1050655,

 1050658,

 1050659,

 1050661,

 1050711,

 1050579,

 1050577,

 1050507,

 1050508,

 1050509,

 1050511,

 1050512,

 1050513,

 1050514,

 1050515,

 1050516,

 1050517,

 1050520,

 1050521,

 1050522,

 1050523,

 1050524,

 1050505,

 1050525,

 1050504,

 1050502,

 1050487,

 1050488,

 1050489,

 1050490]
qqq = x_train[x_train.ncodpers.isin(l)]
qqq.to_csv('nbp1.csv')
#     from IPython.display import FileLink

#     FileLink('nbp1.csv')
# Set your own project id here

# PROJECT_ID = 'My Project 93830'

# from google.cloud import storage

# storage_client = storage.Client(project=PROJECT_ID)
# def implicit():

#     from google.cloud import storage



#     # If you don't specify credentials when constructing the client, the

#     # client library will look for credentials in the environment.

#     storage_client = storage.Client()



#     # Make an authenticated API request

#     buckets = list(storage_client.list_buckets())

#     print(buckets)
# implicit()
# def create_download_link(title = "Download CSV file", filename = "data.csv"):  

#     html = '<a href={filename}>{title}</a>'

#     html = html.format(title=title,filename=filename)

#     return HTML(html)
# create_download_link(filename='nbp1.csv')
# import html
# from IPython.display import HTML