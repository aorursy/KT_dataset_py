import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

df.head(10)
!pip install plotly.express
import plotly.express as px
import plotly.graph_objects as go



import pandas as pd



fig = go.Figure()



Ages = ['0-17', '18-25', '26-35', '36-45','46-50', '51-55', '55+']



for Age in Ages:

    fig.add_trace(go.Violin(x=df['Age'][df['Age'] == Age],

                            y=df['Purchase'][df['Age'] == Age],

                            name=Age,

                            box_visible=True,

                            meanline_visible=True))

fig.update_layout(title_text='Consumo por Faixa de idade')



fig.show()
mais_comprado=pd.DataFrame(df["Product_ID"].value_counts()).head(10)

mais_comprado=mais_comprado.reset_index()

mais_comprado=mais_comprado.rename(columns={'index':'Product_Id', 'Product_ID':'Qtd'})

mais_comprado
fig = px.bar(mais_comprado, x='Product_Id', y='Qtd')

fig.show()
freq_occup = list(df['Occupation'].value_counts().index)[0:5]

freq_occup
sns.set(rc = {'figure.figsize' : (15, 10)})

df_17 = df.query('Age == "0-17"')

df_17 = df_17[df_17['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_17['Occupation'], y = df_17['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (Até 17 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (15, 10)})

df_25 = df.query('Age == "18-25"')

df_25 = df_25[df_25['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_25['Occupation'], y = df_25['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (18 à 25 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (15, 10)})

df_45 = df.query('Age == "36-45"')

df_45 = df_45[df_45['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_45['Occupation'], y = df_45['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (36 à 45 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (15, 10)})

df_50 = df.query('Age == "46-50"')

df_50 = df_50[df_50['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_50['Occupation'], y = df_50['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (46 à 50 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (15, 10)})

df_55 = df.query('Age == "51-55"')

df_55 = df_55[df_55['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_55['Occupation'], y = df_55['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (51 à 55 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (15, 10)})

df_max = df.query('Age == "55+"')

df_max = df_max[df_max['Occupation'].isin(freq_occup)]

sns.violinplot(x = df_max['Occupation'], y = df_max['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (Acima de 55 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'point',

            col = 'Occupation',

            data = df[df['Purchase'] > 9000],

            aspect = .4,

            col_wrap = 7)