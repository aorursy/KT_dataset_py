# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

matplotlib.style.use('ggplot') #set plot style to ggplot



# Input data files are available in the "../input/" directory.
%ls ../input
columns_dict = {'ind_empleado':'Employee index: A active, B ex employed, F filial, N not employee, P pasive', 

                 'pais_residencia':'Customer\'s Country residence', 

                 'sexo':'Customer\'s sex',

                 'age': 'Age', 

                 'fecha_alta':'The date in which the customer became as the first holder of a contract in the bank',

                 'ind_nuevo':'New customer Index. 1 if the customer registered in the last 6 months.', 

                 'antiguedad':'Customer seniority (in months)', 

                 'indrel':'1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)', 

                 'ult_fec_cli_1t':'Last date as primary customer (if he isn\'t at the end of the month)', 

                 'indrel_1mes':'Customer type at the beginning of the month,1 (First/Primary),2 (co-owner),P(Potential),3 (former primary),4(former co-owner)', 

                 'tiprel_1mes':'Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)', 

                 'indresi':'Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)',

                 'indext':'Foreigner index (S (Yes) or N (No) if the customer\'s birth country is different than the bank country)',

                 'conyuemp':'Spouse index. 1 if the customer is spouse of an employee',

                 'canal_entrada':'channel used by the customer to join',

                 'indfall':'Deceased index. N/S', 

                 'tipodom':'Addres type. 1, primary address', 

                 'cod_prov':'Province code (customer\'s address)',

                 'nomprov':'Province name', 

                 'ind_actividad_cliente':'Activity index (1, active customer; 0, inactive customer)',

                 'renta':'Gross income of the household', 

                 'segmento':'segmentation: 01 - VIP, 02 - Individuals 03 - college graduated',

                 'ind_ahor_fin_ult1':'Saving Account',

                 'ind_aval_fin_ult1':'Guarantees',

                 'ind_cco_fin_ult1':'Current Accounts',

                 'ind_cder_fin_ult1':'Derivada Account',

                 'ind_cno_fin_ult1':'Payroll Account',

                 'ind_ctju_fin_ult1':'Junior Account',

                 'ind_ctma_fin_ult1':'Más particular Account',

                 'ind_ctop_fin_ult1':'particular Account',

                 'ind_ctpp_fin_ult1':'particular Plus Account',

                 'ind_deco_fin_ult1':'Short-term deposits',

                 'ind_deme_fin_ult1':'Medium-term deposits',

                 'ind_dela_fin_ult1':'Long-term deposits',

                 'ind_ecue_fin_ult1':'e-account',

                 'ind_fond_fin_ult1':'Funds',

                 'ind_hip_fin_ult1':'Mortgage',

                 'ind_plan_fin_ult1':'Pensions',

                 'ind_pres_fin_ult1':'Loans',

                 'ind_reca_fin_ult1':'Taxes',

                 'ind_tjcr_fin_ult1':'Credit Card',

                 'ind_valo_fin_ult1':'Securities',

                 'ind_viv_fin_ult1':'Home Account',

                 'ind_nomina_ult1':'Payroll',

                 'ind_nom_pens_ult1':'Pensions',

                 'ind_recibo_ult1':'Direct Debit'

                }

def plot_nominal(dataName, columnName, logy=False, figsize=(6,4)):

    df = pd.read_csv('../input/' + dataName + '_ver2.csv', usecols=[columnName])

    cats_counts = df.groupby([columnName]).size()

    cats_counts.sort_values(inplace=True, ascending=False)

    cats_counts.plot.bar(alpha=0.8, title= columns_dict[columnName] , logy=logy, figsize=figsize)

    

def plot_ordinal(dataName, columnName, logy=False, figsize=(6,4)):

    df = pd.read_csv('../input/' + dataName + '_ver2.csv', usecols=[columnName])

    cats_counts = df.groupby([columnName]).size()

    cats_counts.plot(alpha=0.8, title= columns_dict[columnName] , logy=logy, figsize=figsize)

    
plot_nominal('train','ind_empleado',True, figsize=(6,4))
plot_nominal('train','pais_residencia',True, figsize=(10,6) )
plot_nominal('train','sexo',figsize=(6,4))
plot_ordinal('train','age',figsize=(8,6))
plot_nominal('train','segmento')
plot_ordinal('train','fecha_alta',figsize=(8,6))