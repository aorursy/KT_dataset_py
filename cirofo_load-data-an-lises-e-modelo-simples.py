import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
# Le os dados
df = pd.read_csv("../input/Loan payments data.csv")

# Colunas
print(df.columns.values)
# Para fim de estudo e adaptação da base, vamos considerar somente PAIDOFF (pago) e COLLECTION (não pago)
df = df[(df['loan_status'] != 'COLLECTION_PAIDOFF')]

# Lista e conta os valores da variável resposta (loan_status)
print(pd.value_counts(df['loan_status']))

# Plota as classes
ax = sns.countplot('loan_status', data = df)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df['loan_status'])), (p.get_x() + 0.3, p.get_height()))
# Exemplo dos dados
df.sample(5)
# Primeiramente transformar a variavel resposta em 0 e 1
df['loan_status'] = df['loan_status'].map({'PAIDOFF': 1, 'COLLECTION': 0})
# Lib para visualizar os missing
# Nesse caso, somente tem missing em past_due_days (que são os casos em que o empréstimo foi pago)
# E em paid_off_time (que são os casos onde estão em atraso)
import missingno as msno

msno.matrix(df)
# Transformação dos valores para aumentar a variação dos empréstimos
# Soma uma uniforme entre -0.1 e 0.1 e arredonda
import random
random.seed(123)
df['Principal'] = round( (df['Principal']*(1 + np.random.uniform(-0.1, 0.1, df.shape[0])))/10 )*10

print(pd.value_counts(df['Principal']))
df.groupby('loan_status').mean()
df.groupby('terms').mean()
df.groupby('loan_status').mean()
df.groupby(['education', 'loan_status']).mean()
# Funcao para plotar histogramas para duas classes (no caso Paidoff e Collection)
# onde 'var' é a variavel do histograma (continua)
def PlotHistCont(data, var, resp):
    # var: variável para criar os histogramas por paidoff e collection
    
    # Paidoff
    sns.distplot(
        # filtra pela variavel loan_status e seleciona a coluna do input 'var'
        data[data[resp]==1].loc[:, var],
        kde = False,  # desliga a linha do histograma
        color = 'b',
        label = '1'
    );
    # Collection
    sns.distplot(
        data[data[resp]==0].loc[:, var],
        kde = False,
        color = 'r',
        label = '0'
    );
    plt.legend()   # adiciona a legenda
plt.figure(figsize=(12, 3))
PlotHistCont(df, var='Principal', resp='loan_status')
plt.figure(figsize=(12, 3))
PlotHistCont(df, var='age', resp='loan_status')
def PlotHistCat(data, var, resp):
    # Funcao para plotar histogramas para duas classes (no caso Paidoff e Collection)
    # onde 'var' é a variavel do histograma (continua)
    
    # Paidoff
    sns.countplot(
        # filtra pela variavel loan_status e seleciona a coluna do input 'var'
        data[data[resp]==1].loc[:, var],
        color = 'b',
        label = '1'
    );
    # Collection
    sns.countplot(
        # filtra pela variavel loan_status e seleciona a coluna do input 'var'
        data[data[resp]==0].loc[:, var],
        color = 'r',
        label = '0'
    );
    plt.legend()   # adiciona a legenda
# Histograma de variável categórica
plt.figure(figsize=(6, 3))
PlotHistCat(df, var='Gender', resp='loan_status')
# Histograma agrupado 
plt.figure(figsize=(10, 3))
sns.countplot(x="education", hue="loan_status", data=df);
# Ajusta o formato das datas para pd timeseries
df[['effective_date', 'due_date', 'paid_off_time']] = df[['effective_date', 'due_date', 'paid_off_time']].apply(pd.to_datetime,errors='coerce')
df[['effective_date', 'due_date', 'paid_off_time']].head(3)
# Extrai os dias da semana para análises
df[['effective_date_weekday', 'due_date_weekday', 'paid_off_time_weekday']] = df[['effective_date', 'due_date', 'paid_off_time']].apply(lambda x: x.dt.weekday)
# Extrai o horário de 'paid_off_time'
df['paid_off_time_hour'] = df['paid_off_time'].dt.hour

df[['effective_date_weekday', 'due_date_weekday', 'paid_off_time_weekday', 'paid_off_time_hour']].head(5)
# Que dia da semana foram feitos os empréstimos e se foi pago ou nao pago
plt.figure(figsize=(12, 3))

sns.distplot(
    df[df['loan_status']==1].effective_date.dt.day,
    bins = np.arange(0, 31),
    kde = False,
    color = 'b'
)
sns.distplot(
    df[df['loan_status']==0].effective_date.dt.day,
    bins = np.arange(0, 31),
    kde = False,
    color = 'r'
)
plt.xticks(range(0,31))
plt.title('Loan Status') 
plt.ylabel('Qtd')
# Que dia da semana foram feitos os empréstimos e se foi pago ou nao pago
plt.figure(figsize=(12, 3))

sns.distplot(
    df[df['loan_status']==1].effective_date_weekday,
    bins = np.arange(0, 8),
    kde = False,
    color = 'b'
)
sns.distplot(
    df[df['loan_status']==0].effective_date_weekday,
    bins = np.arange(0, 8),
    kde = False,
    color = 'r'
)
plt.xticks(range(0,8))
plt.title('Loan Status') 
plt.ylabel('Qtd')

print('0 é Segunda e 6 Domingo')
# Histograma dos horários em que foram pagos os empréstimos
plt.figure(figsize=(12, 3))

sns.distplot(
    df[df['loan_status']==1].paid_off_time_hour,
    bins = np.arange(0, 25),
    kde = False
)
plt.xticks(range(0,25))
plt.title('Paidoff') 
plt.ylabel('Qtd')
# Histograma dos horários em que foram pagos os empréstimos
plt.figure(figsize=(12, 3))

sns.distplot(
    df[df['loan_status']==1].paid_off_time.dt.day,
    bins = np.arange(0, 31),
    kde = False
)
plt.xticks(range(0,31))
plt.title('Paidoff') 
plt.ylabel('Qtd')
import statsmodels.api as sm
from patsy import dmatrices   # Cria X e y facilmente (variavel resposta e variaveis preditoras)

# Separa X e y de uma forma parecida com o R, além de fazer o One Hot Encoding automáticamente
y, X = dmatrices("loan_status ~ Principal + terms + age + education + Gender", df, return_type = 'dataframe')

lr_model = sm.Logit(y, X).fit()
print(lr_model.summary2())