import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["wc", "-l", "../input/train_ver2.csv"]).decode("utf8"))
import random

n = 13647310 

s = 10000 

filename = "../input/train_ver2.csv"

skip = sorted(random.sample(range(1, n), n-s+1))

df = pd.read_csv(filename, skiprows=skip)
products = ['ind_ahor_fin_ult1',    #Saving Account

            'ind_aval_fin_ult1',	#Guarantees

            'ind_cco_fin_ult1',	    #Current Accounts

            'ind_cder_fin_ult1',	#Derivada Account 

            'ind_cno_fin_ult1',	    #Payroll Account

            'ind_ctju_fin_ult1',	#Junior Account

            'ind_ctma_fin_ult1',	#Más particular Account

            'ind_ctop_fin_ult1',	#particular Account

            'ind_ctpp_fin_ult1',	#particular Plus Account

            'ind_deco_fin_ult1',	#Short-term deposits

            'ind_deme_fin_ult1',	#Medium-term deposits

            'ind_dela_fin_ult1',	#Long-term deposits

            'ind_ecue_fin_ult1',	#e-account

            'ind_fond_fin_ult1',	#Funds

            'ind_hip_fin_ult1',	    #Mortgage

            'ind_plan_fin_ult1',	#Pensions

            'ind_pres_fin_ult1',	#Loans

            'ind_reca_fin_ult1',	#Taxes

            'ind_tjcr_fin_ult1',	#Credit Card

            'ind_valo_fin_ult1',	#Securities

            'ind_viv_fin_ult1',	    #Home Account

            'ind_nomina_ult1',	    #Payroll

            'ind_nom_pens_ult1',	#Pensions

            'ind_recibo_ult1']
v_subset = pd.DataFrame({'tot_products':df[products].sum(axis=1), 

                         'renta':df.renta, 

                         'renta_bin':pd.qcut(df.renta, q=100, labels=False),

                         'antiguedad':pd.to_numeric(df.antiguedad, errors='coerce')})

v_subset['antiguedad_bin'] = pd.qcut(v_subset.antiguedad, q=50, labels=False)

v_subset.head()
(v_subset[['antiguedad', 'tot_products']]

.groupby('antiguedad', as_index=False)

.mean()

.plot(kind='scatter', x='antiguedad', y='tot_products'))
(v_subset[['renta_bin', 'tot_products']]

.groupby('renta_bin', as_index=False)

.mean()

.plot(kind='scatter', x='renta_bin', y='tot_products'))
(v_subset[['antiguedad_bin', 'tot_products']]

.groupby('antiguedad_bin', as_index=False)

.mean()

.plot(kind='scatter', x='antiguedad_bin', y='tot_products'))