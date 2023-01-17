import matplotlib.pyplot as plt 

from pandas import read_excel, DataFrame

my_sheet = 'Jio'

file_name = '../input/data-rates/data rates updated.xlsx'

jio_df = read_excel(file_name, sheet_name = my_sheet)

jio_df
jio_df.plot(x ='data value', y='data per day', kind = 'scatter')
my_sheet = 'Bsnl'

file_name = '../input/data-rates/data rates updated.xlsx'

bsnl_df = read_excel(file_name, sheet_name = my_sheet)

bsnl_df
bsnl_df.plot(x ='data value', y='data per day', kind = 'scatter')	
my_sheet = 'Idea vodafone'

file_name = '../input/data-rates/data rates updated.xlsx'

iv_df = read_excel(file_name, sheet_name = my_sheet)

iv_df
iv_df.plot(x ='data value', y='data per day', kind = 'scatter')	
my_sheet = 'Airtel'

file_name = '../input/data-rates/data rates updated.xlsx'

airtel_df = read_excel(file_name, sheet_name = my_sheet)

airtel_df
airtel_df.plot(x ='data value', y='data per day', kind = 'scatter')
j_dv=jio_df['data value'].values.tolist()

j_dpd=jio_df['data per day'].values.tolist()

b_dv=bsnl_df['data value'].values.tolist()

b_dpd=bsnl_df['data per day'].values.tolist()

a_dv=airtel_df['data value'].values.tolist()

a_dpd=airtel_df['data per day'].values.tolist()

iv_dv=iv_df['data value'].values.tolist()

iv_dpd=iv_df['data per day'].values.tolist()
plt.scatter(j_dv, j_dpd)

plt.scatter(a_dv, a_dpd)

plt.scatter(b_dv, b_dpd)

plt.scatter(iv_dv, iv_dpd)



plt.legend(["JIO", "AIRTEL","IDEA","BSNL"])
plt.scatter(j_dv[:5], j_dpd[:5])

plt.scatter(a_dv[:5], a_dpd[:5])

plt.scatter(b_dv[:5], b_dpd[:5])

plt.scatter(iv_dv[:5], iv_dpd[:5])



plt.legend(["JIO", "AIRTEL","IDEA","BSNL"])
my_sheet = 'TOP 10'

file_name = '../input/data-rates/data rates updated.xlsx'

top_df = read_excel(file_name, sheet_name = my_sheet)

top_df.head(10)