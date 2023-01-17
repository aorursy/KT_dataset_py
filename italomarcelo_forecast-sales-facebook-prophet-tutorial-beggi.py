import pandas as pd

import numpy as np

from fbprophet import Prophet

from fbprophet.plot import plot_components_plotly, plot_plotly

import plotly.express as px
path = 'https://raw.githubusercontent.com/italomarcelogit/python.free/master/fbprophet/pdv.csv'

df = pd.read_csv(path)
cidades = ['SAO PAULO', 'RIO DE JANEIRO', 'SALVADOR', 'MANAUS', 'BRASILIA', 'PORTO ALEGRE', 'CURITIBA', 'RECIFE']

lojas = [

         {'loja':1, 'cidade':0, 'filial': 'AV_PAULISTA', 'tipo': 'REDE'},

         {'loja':2, 'cidade':1, 'filial': 'IPANEMA', 'tipo': 'REDE'},

         {'loja':3, 'cidade':2, 'filial': 'PELOURINHO', 'tipo': 'REDE'},

         {'loja':4, 'cidade':3, 'filial': 'ZONAFRANCA', 'tipo': 'REDE'},

         {'loja':5, 'cidade':4, 'filial': 'MONUMENTO', 'tipo': 'REDE'},

         {'loja':6, 'cidade':5, 'filial': 'INTER_ESTADIO', 'tipo': 'REDE'},

         {'loja':7, 'cidade':6, 'filial': 'PARQUE', 'tipo': 'REDE'},

         {'loja':8, 'cidade':7, 'filial': 'MARCO_CENTRAL', 'tipo': 'REDE'},

         {'loja':9, 'cidade':0, 'filial': 'JARDINS', 'tipo': 'FRANQUIA'},

]

produtos = ('ABCDEFGHIJ')

nomes = ['John', 'Maria', 'Ana', 'Sylvia', 'Walter', 'Newman', 'Clara', 'Bia', 'Jose']
# function Exploratory Data Analysis

def eda(dfA, all=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nDTypes - Numerics')

    print(dfA.select_dtypes(include=np.number).columns.tolist())

    print(f'\nDTypes - Categoricals')

    print(dfA.select_dtypes(include='object').columns.tolist())

    print(f'\nIs Null: {dfA.isnull().sum().sum()}')

    print(f'{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if all:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Objects')

        print(dfA.describe(include=['object']))
eda(df)
df.head(1)
df.sample(2)
df.tail(1)
topProd = df.produto.value_counts()[:5]

x = [produtos[x] for x in topProd.index]

g1 = px.bar(x=x, y=topProd.values, title='Top Products',

            color=topProd.values, color_continuous_scale=px.colors.sequential.Viridis)

g1.show()
topShop = df.loja.value_counts()[:5]

x = [lojas[x]['filial'] for x in topShop.index]

g1 = px.bar(x=x, y=topShop.values, title='Top Shop',

            color=topShop.values, color_continuous_scale=px.colors.sequential.RdBu_r)

g1.show()
topCity = df.cidade.value_counts()[:5]

x = [cidades[x] for x in topCity.index]

g1 = px.bar(y=x, x=topCity.values, title='Top Shop', orientation='h',

            color=topCity.values, color_continuous_scale=px.colors.sequential.Inferno)

g1.show()
data = df[['data', 'total']].groupby('data').sum()

g1 = px.line(data, x=data.index, y='total', title='Top Shop')

g1.show()
df1 = df[['data', 'total']].copy()

df1.columns = ['ds', 'y']

df1.ds = pd.to_datetime(df1.ds).dt.to_period('m')
df1.info()
df1.head(3)
x = df1.groupby('ds').sum()

# change dtype period[M] to datetime

x.index = x.index.astype('datetime64[ns]')
database = pd.DataFrame()

database['ds'] = x.index

database['y'] = x.values
m = Prophet()

# I'll call fit method and pass the database var

m.fit(database)
# The forecast will be made from the column 'ds' of the dataframe 'database'. 

# This variable contains the dates to which we will create the forecast.

# We will predict a dataframe that spans 4 months. 

# Because it contains values ​​accumulated monthly, we will use the frequency 'M'

f = m.make_future_dataframe(periods=4, freq='M')
# Now, we will assign each row in future (f var) a predicted value

p = m.predict(f)
# And plot the forecast (p = prediction)

plot_plotly(m, p)
plot_components_plotly(m, p)