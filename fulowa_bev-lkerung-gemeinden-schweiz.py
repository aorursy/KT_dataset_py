import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_excel('../input/je-d-21.03.01.xlsx', skiprows=7)
df_clean = pd.DataFrame({'Gemeinde': df['Unnamed: 1'], 'Bevoelkerung_2017': df['Unnamed: 2']})

df_clean = df_clean.dropna()

df_clean = df_clean.loc[df_clean['Gemeinde']!='Schweiz']

df_clean['Bevoelkerung_2017'] = df_clean['Bevoelkerung_2017'].astype('int32')
df_clean.head()
df_clean.sort_values(by='Bevoelkerung_2017', ascending=False).head(10)
def look_up_Gemeinde(name):

    if type(name) == str:

        print(df_clean.loc[df_clean['Gemeinde']==name, :])

    else:

        print('Bitte überprüfen Sie Schreibeweise und benützen Sie Anführungszeichen z.B.: "Zürich"')
look_up_Gemeinde('Aarau')
str_nums = [str(x) for x in df_clean.Bevoelkerung_2017]

str_nums = [list(x) for x in str_nums]

flat_list = [item for sublist in str_nums for item in sublist]

df_nums = pd.DataFrame({'numbers': [int(x) for x in flat_list]})

df_nums.numbers.value_counts(normalize=True).plot.barh();
df_nums.numbers.value_counts(normalize=True)