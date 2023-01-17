# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


#Importing data set BaseDPEvolucaoMensalCisp
raw_BaseDP = pd.read_csv('../input/BaseDPEvolucaoMensalCisp.csv', sep=';', encoding = 'ISO-8859-1')
raw_PopEvoluc = pd.read_csv('../input/PopulacaoEvolucaoMensalCisp.csv', sep=';', encoding = 'ISO-8859-1')
raw_Delegacias = pd.read_csv('../input/delegacias.csv', sep=';', encoding = 'UTF-8')

print('DPE Dataset')
df = pd.DataFrame(raw_BaseDP)
print(df.head())

print(df.tail())
print(df.columns)
df.shape
df.info()
#Label columns

print('\nVerifying frequency of data\n')
print(df['munic'].value_counts(dropna=False))

print('\nVerifying frequency of Regions\n')
print(df['Regiao'].value_counts(dropna=False).head())

print('\nVerifying frequency of month\n')
print(df['mes'].value_counts(dropna=False).head())

print('\nVerifying frequency of month/year\n')
print(df['mes_ano'].value_counts(dropna=False).head())

print('\nVerifying frequency of fase\n')
print(df['fase'].value_counts(dropna=False).head())



#As we saw above, thess feature labels needing to be ajusted
# Ajust regiao field, putting all as Upper Letter and removing spaces
df.Regiao = [x.strip().upper() for x in df.Regiao]
#changing data type
df.Regiao = df.Regiao.astype('category')

# Ajust municipe field, putting all as Upper Letter and removing spaces
df.munic = [x.strip().upper() for x in df.munic]
#changing datatype
df.munic = df.munic.astype('category')

# Ajust mes_ano values
df.mes_ano = [x.strip().replace('m', '/') for x in df.mes_ano]

#Convert to datetime
df['mes_ano'] = pd.to_datetime(df['mes_ano'],format='%Y-%m')

print(df.info())
print('Suspycious columns\n')

print('\nVerifying frequency of cycle stoles\n')
print(df['roubo_bicicleta'].value_counts(dropna=False).head())

print('\nVerifying frequency of areacisp\n')
print(df['area_cisp'].value_counts(dropna=False).head())

print('\nVerifying frequency of grp\n')
print(df['grp'].value_counts(dropna=False).head())

print('\nVerifying frequency of grp\n')
print(df['apf_cmp'].value_counts(dropna=False).head())

print('\nVerifying frequency of apreensoes\n')
print(df['apreensoes'].value_counts(dropna=False).head())

print('\nVerifying frequency of gaai\n')
print(df['gaai'].value_counts(dropna=False).head())

print('\nVerifying frequency of aaapai_cmba\n')
print(df['aaapai_cmba'].value_counts(dropna=False).head())  

print('Summaryzing the dataset\n')
df.describe()

#Change df index
df.index = df['mes_ano']
df.index

df.boxplot(column ='lesao_corp_dolosa' , by = 'Regiao', rot=90)
plt.show()

df.boxplot(column ='hom_doloso' , by = 'Regiao', rot=90)
plt.show()

df.boxplot(column ='roubo_veiculo' , by = 'Regiao', rot=90)
plt.show()

df.boxplot(column ='roubo_carga' , by = 'Regiao', rot=90)
plt.show()
# Tidy dataset (Melt process)
tdy = pd.melt(frame=df , id_vars= 'Regiao', value_vars = ['lesao_corp_dolosa','hom_doloso'], var_name = 'Crimes', value_name = 'Result')

#Aggregating values
#agg_tb = pd.DataFrame(tdy.groupby(['Regiao','Crimes'], as_index=False)['Result'].sum())

#Pivotting tidy sum
tdy.pivot_table(index ='Regiao', columns ='Crimes'  , values ='Result', aggfunc= np.sum )
#Pivotting tidy avg
tdy.pivot_table(index ='Regiao', columns ='Crimes'  , values ='Result', aggfunc= np.mean )
# Tidy dataset (Melt process)
tdy = pd.melt(frame=df , id_vars= 'Regiao', value_vars = ['roubo_veiculo','roubo_carga'], var_name = 'Crimes', value_name = 'Result')

#Aggregating values
pd.DataFrame(tdy.groupby(['Regiao','Crimes'], as_index=False)['Result'].sum())
tdy.pivot_table(index = 'Regiao', columns ='Crimes' , values ='Result' , aggfunc = np.sum )
tdy.pivot_table(index = 'Regiao', columns ='Crimes' , values ='Result' , aggfunc = np.mean )
print('\nPopulation Evolution Dataset')
df_2 = pd.DataFrame(raw_PopEvoluc) 
print(df_2.head())
print(df_2.tail())
print(df_2.columns)
df_2.shape
df_2.info()

print('\nDelegacias Dataset')
df_3 = pd.DataFrame(raw_Delegacias) 
print(df_3.head())
print(df_3.tail())
print(df_3.columns)
df_3.shape
df_3.info()


med = df.resample('Y').mean()
#med[['hom_doloso','lesao_corp_morte','latrocinio','tentat_hom','lesao_corp_dolosa','estupro','hom_culposo','lesao_corp_culposa', 'roubo_comercio','roubo_residencia','roubo_veiculo','roubo_carga','roubo_transeunte','roubo_em_coletivo','roubo_banco','roubo_cx_eletronico','roubo_celular','roubo_conducao_saque','roubo_bicicleta']]
homicidios = med[['hom_doloso','tentat_hom','hom_culposo']]
homicidios.plot(subplots=True)
plt.xlabel('Time/Period')
plt.ylabel('Median of homicides')
plt.show()

#med[['hom_doloso','lesao_corp_morte','latrocinio','tentat_hom','lesao_corp_dolosa','estupro','hom_culposo','lesao_corp_culposa', 'roubo_comercio','roubo_residencia','roubo_veiculo','roubo_carga','roubo_transeunte','roubo_em_coletivo','roubo_banco','roubo_cx_eletronico','roubo_celular','roubo_conducao_saque','roubo_bicicleta']]
lesoes = med[['lesao_corp_morte','lesao_corp_dolosa','lesao_corp_culposa']]
lesoes.plot(subplots=True)
plt.xlabel('Time/Period')
plt.ylabel('Median of bodily injury')
plt.show()
#med[['hom_doloso','lesao_corp_morte','latrocinio','tentat_hom','lesao_corp_dolosa','estupro','hom_culposo','lesao_corp_culposa', 'roubo_comercio','roubo_residencia','roubo_veiculo','roubo_carga','roubo_transeunte','roubo_em_coletivo','roubo_banco','roubo_cx_eletronico','roubo_celular','roubo_conducao_saque','roubo_bicicleta']]
place_rouberies = med[['roubo_comercio','roubo_residencia','roubo_banco','roubo_cx_eletronico']]
place_rouberies.plot(subplots=True)
plt.xlabel('Time/Period')
plt.ylabel('Median of roubery')
plt.show()

place_rouberies = med[['roubo_veiculo','roubo_carga','roubo_em_coletivo']]
place_rouberies.plot(subplots=True)
plt.xlabel('Time/Period')
plt.ylabel('Median of roubery against transports')
plt.show()

print('Crimes against load transports, stores and banks growded when the events occour. We can conclude that events don\'t reduce crimes,  the events just changed the focus. Instead to make simple crimes, crimonouses choose more sofisticated crimes that can return more lucre than simple crimes.')
print('Crimes against life decreased on events period. This trend can be because on this periods, the country putted more policemans on the streets, to reduce simple crimes. But this hipoteses needing to be evaluated comparring the effective of police on these periods.')

#Rename df_2 column Circ to CISP
df_2.columns = ['CISP','mes','vano','pop_circ']

#treat area_cisp separator value
df.area_cisp = [x.strip().replace(',', '.') for x in df.area_cisp]

#Joint data sets
join = pd.merge(df,df_3,how='inner',on ='CISP' )
all_data = pd.merge(join,df_2, how='inner', on=['CISP','mes','vano'])

all_data.head()
#Verifying quantity of rows
total_rows=len(all_data.axes[0])
print('Total Rows in Data Set : ',total_rows)

#Replacing NaN values by the mean of values for each column, to not exclude columns
all_data = all_data.fillna(all_data.mean())

print(all_data.head())
print(all_data.tail())
print(all_data.columns)
all_data.shape
all_data.info()

#Verifying NaN quantity on each column
nl = all_data.isnull().sum()
print('Null Values Quantity :\n', nl)

#Fraudulent homicid (Doloso) by year
#Sum homicids
h_dol_year = pd.DataFrame(all_data.groupby(['vano'], as_index=False)['hom_doloso'].sum())

year = h_dol_year['vano']
vl = h_dol_year['hom_doloso']
#Getting the Max year
h_dol_year_max = h_dol_year.loc[h_dol_year['hom_doloso'].idxmax()]

# Showing results
print('Fraudulent homicide (Doloso) by year:')
print(h_dol_year[:])
print('More violent year by Fraudulent homicide (Doloso):')
print(h_dol_year_max)


# to Input Graph
plt.plot(year, vl , color="red", linewidth=2.5, linestyle="-")
plt.xlabel('Years')
plt.ylabel('Fraudulent Homicides')
plt.title('Homicides by Year')
plt.grid(True)
plt.show()


#Fraudulent homicid (Doloso) by year and municipe
h_dol_mun_year = pd.DataFrame(all_data.groupby(['vano','munic'], as_index=False)['hom_doloso'].sum())

#Fraudulent homicid (Doloso) by municipe
h_dol_munr = pd.DataFrame(all_data.groupby(['munic'], as_index=False)['hom_doloso'].sum())


print('Top 10 Fraudulent Homicid by Municipe'  )
# to Input Graph
h_dol_munr['Ranking'] = h_dol_munr['hom_doloso'].rank(ascending=1)
rak = h_dol_munr.query('Ranking > 80')  
    
    
labels = rak['munic']
sizes = rak['hom_doloso']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','green', 'purple','skyblue']

# Plot
plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140, labeldistance =1.2)

plt.axis('equal')
plt.show()

print('Total Criminal Incidents by Year')

#Fraudulent homicid (Doloso) by year and municipe
h_dol_mun_year = pd.DataFrame(all_data.groupby(['vano'], as_index=False)['total_furtos'].sum())

year = h_dol_mun_year['vano']
vl = h_dol_mun_year['total_furtos']

# to Input Graph
plt.plot(year, vl , color="blue", linewidth=1.5, linestyle="-")
plt.xlabel('Years')
plt.ylabel('Total stoles')
plt.title('Total Criminal Incidents by Year')
plt.grid(True)
plt.show()

#Calculing the average of crimes
#Fraudulent homicid (Doloso) by year and municipe
h_dol_mun_year = pd.DataFrame(all_data.groupby(['vano'], as_index=False)['total_furtos'].mean())

year = h_dol_mun_year['vano']
vl = h_dol_mun_year['total_furtos']

# to Input Graph
plt.plot(year, vl , color="blue", linewidth=1.5, linestyle="-")
plt.xlabel('Years')
plt.ylabel('Avg stoles')
plt.title('Average of Criminal Incidents by Year')
plt.grid(True)
plt.show()


#Analysing Correlation Between Total Crimes and Population


crimes = all_data['total_furtos']
pop = all_data['pop_circ']
year = all_data['vano']

plt.scatter(pop, crimes
, s= 0.8)
plt.xlabel('Population')
plt.ylabel('Crimes')
plt.title('Correlation Between Total Crimes and Population')
plt.show()

correlation = np.corrcoef(pop,crimes)
print("We could analyse that the correlation exist, being positive but, isn't significant to confirm at all that where are the greater population, have major incident of crimes")
print(correlation)

#Analayse total incidents by Region

reg_crimes = pd.DataFrame(all_data.groupby(['Regiao'], as_index=False)['total_furtos','pop_circ'].sum())


print("How representative are each region, analysing total amount of stoles")
# Pie Plot
plt.pie(reg_crimes.total_furtos, labels=reg_crimes.Regiao, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140, labeldistance =1.2)

plt.axis('equal')
plt.show()

# Sort values to better presentation
reg_crimes = reg_crimes.sort_values("pop_circ")

print("Analysing correlation between population and total crimes by region")
pop = reg_crimes.pop_circ 
crimes = reg_crimes.total_furtos

plt.scatter(pop, crimes
, s= 20)
plt.xlabel('Population')
plt.ylabel('Crimes')
plt.title('Correlation Between Total Crimes and Population')
plt.plot(pop,crimes, marker='o', linestyle='dashed',
        linewidth=2, markersize=12)
plt.show()

correlation = np.corrcoef(pop,crimes)
print(correlation)

print("After sum by region the fields, population and total crimes, we could affirm that exist a correlation strong and positive.")
reg_crimes.head()