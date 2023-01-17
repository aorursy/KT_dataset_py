import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn import metrics

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor

plt.rcParams.update({'font.size': 15})

plt.rcParams['figure.figsize'] = 18, 5

pd.set_option('display.max_colwidth', 50)
acc_bal = pd.read_csv('../input/upniddata/account_balances.csv')

acc = pd.read_csv('../input/upniddata/accounts.csv')

b_transf = pd.read_csv('../input/upniddata/bank_transfers.csv')

customer = pd.read_csv('../input/upniddata/customers.csv')

payments = pd.read_csv('../input/upniddata/payments.csv')
acc_bal.head()
acc.head()
b_transf.head()
customer.head()
payments.head()
b_transf = b_transf.rename(columns={'created_at':'transfer_at', '_id':'transfer_id'})
acc = acc.rename(columns={'_id':'account_id', 'created_at':'acc_created_at'})
acc
dataset = acc.merge(b_transf, on='account_id', how='inner')
dataset
dataset['amount_transferred'].plot(kind='box', showfliers=False)
dataset['amount_transferred'].describe()
dataset['transfer_at'] = dataset['transfer_at'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

dataset['birthdate'] = dataset['birthdate'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
dataset['acc_created_at'].head()
dataset['acc_created_at'] = dataset['acc_created_at'].str.replace(" \d\d:\d\d:\d\d UTC", "")
dataset['acc_created_at'] = dataset['acc_created_at'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
dataset.info()
dataset['transfer_year'] = pd.DatetimeIndex(dataset['transfer_at']).year
dataset['transfer_month'] = pd.DatetimeIndex(dataset['transfer_at']).month
dataset.head()
dataset['transfer_year'].unique()
dataset[dataset['transfer_year'] == 2021]
dataset.info()
dataset[dataset['transfer_year'] == 2021]['status'].value_counts()
dataset['available_at'] = pd.to_datetime(dataset['available_at'], format='%Y-%m-%d')
dataset[dataset['available_at'] != dataset['transfer_at']]
saques_mes = dataset[['transfer_month', 'amount_transferred']].groupby(by='transfer_month').agg(sum).sort_values(by='amount_transferred', ascending=False).plot(kind='bar', rot=0, legend=False)

saques_mes.set_ylabel('Soma dos saques em Reais')

saques_mes.set_xlabel('Mês')

saques_mes = dataset[['transfer_month', 'amount_transferred']].groupby(by='transfer_month').agg(sum).sort_values(by='transfer_month', ascending=True).plot(kind='bar', rot=0, legend=False)

saques_mes.set_ylabel('Soma dos saques em Reais')

saques_mes.set_xlabel('Mês')
payments['amount_paid'].plot(kind='box', showfliers=False)
payments['amount_paid'].describe()
payments['paid_at'] = payments['paid_at'].str.replace(' UTC', '')
payments
payments['paid_at_date'] = payments['paid_at'].str.replace(' \d\d:\d\d:\d\d', '')

payments['paid_at_date'] = payments['paid_at_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

payments['paid_month'] = pd.DatetimeIndex(payments['paid_at_date']).month



payments['paid_hour'] = payments['paid_at'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

payments['paid_hour'] = pd.DatetimeIndex(payments['paid_hour']).hour
customer = customer.rename(columns={'_id': 'customer_id'})
dataset2 = payments.merge(customer, on='customer_id', how='inner')
dataset2.head()
paga_mes = dataset2[['amount_paid', 'paid_month']].groupby(by='paid_month').agg(sum).sort_values(by='amount_paid', ascending=False).plot(kind='bar', rot=0, legend=False)

paga_mes.set_ylabel('Total pago em Reais')

paga_mes.set_xlabel('Mês')

paga_mes = dataset2[['amount_paid', 'paid_month']].groupby(by='paid_month').agg(sum).sort_values(by='paid_month').plot(kind='bar', rot=0, legend=False, title='Pagamentos')

paga_mes.set_ylabel('Total pago em Reais')

paga_mes.set_xlabel('Mês de pagamento')
paga_mes = dataset2[['amount_paid', 'paid_month']].groupby(by='paid_month').agg(sum).sort_values(by='paid_month').plot(kind='bar', rot=0, legend=False, title='Pagamentos')

paga_mes.set_ylabel('Soma dos valores pagos')

paga_mes.set_xlabel('Mês de pagamento')

saques_mes = dataset[['transfer_month', 'amount_transferred']].groupby(by='transfer_month').agg(sum).sort_values(by='transfer_month', ascending=True).plot(kind='bar', rot=0, legend=False, title='Saques de vendedores')

saques_mes.set_ylabel('Soma dos saques')

saques_mes.set_xlabel('Mês')
pagamentos = dataset2[['amount_paid', 'paid_month']].groupby(by='paid_month').agg(sum).sort_values(by='paid_month')

saques = saques_mes = dataset[['transfer_month', 'amount_transferred']].groupby(by='transfer_month').agg(sum).sort_values(by='transfer_month', ascending=True)
corr, _ = pearsonr(pagamentos['amount_paid'], saques['amount_transferred'])
corr
dataset2['ip_location'] = dataset2['ip_location'].str.replace('/BR', '')
pag_est = dataset2[['amount_paid', 'ip_location']].groupby(by='ip_location').agg(sum).sort_values(by='amount_paid', ascending=False).plot(kind='bar', rot=0, legend=False)

pag_est.set_ylabel('Pagamentos somados em Reais')

pag_est.set_xlabel('')
pag_est = dataset2[['amount_paid', 'ip_location']].groupby(by='ip_location').agg('mean').sort_values(by='amount_paid', ascending=False).plot(kind='bar', rot=0, legend=False)

pag_est.set_ylabel('Pagamento médio em Reais')

pag_est.set_xlabel('')
comiss = dataset2['marketplace_fee'].hist()

comiss.set_ylabel('Frequência')

comiss.set_xlabel('Valor em Reais')

comiss.set_title('Distribuição do Marketplace Fee')
dataset2['marketplace_fee'].describe()
dataset2['percent_fee'] = (dataset2['marketplace_fee']/dataset2['amount_paid'])*100
comiss_graf = dataset2['percent_fee'].hist()

comiss_graf.set_title('Distribuição do Marketplace Fee em %')

comiss_graf.set_xlabel('% do valor pago')

comiss_graf.set_ylabel('Frequência')
dataset2['percent_fee'].describe()
dataset2['percent_fee'].mode()
acc_bal[acc_bal['created_at'] != acc_bal['available_at']]
acc_bal
dataset3 = acc.merge(acc_bal, on='account_id', how='inner')
dataset3
mov_pay = dataset3[dataset3['movement_type'] == 'bank_transfer']['amount_moved'].plot(kind='hist')

mov_pay.set_ylabel('Frequência')

mov_pay.set_xlabel('Valor movimentado')
mov_pay = dataset3[dataset3['movement_type'] == 'payment']['amount_moved'].plot(kind='hist')

mov_pay.set_ylabel('Frequência')

mov_pay.set_xlabel('Valor movimentado')
fee = dataset3['fee'].plot(kind='hist', title='Distribuição dos valores de comissão')

fee.set_ylabel('Frequência')

fee.set_xlabel('Valor em Reais')
dataset3['fee'].describe()
dataset3['fee'].mode()
dataset3['fee_perc'] = (dataset3['fee']/dataset3['amount_moved'])*100
fee_p = dataset3[dataset3['amount_moved'] > 0]['fee_perc'].hist(bins=8)

fee_p.set_ylabel('Frequência')

fee_p.set_xlabel('Valor em Reais')
dataset3['fee_perc'].describe()
dataset3['fee_perc'].mode()
dataset4 = payments.merge(customer, on='customer_id', how='inner')
dataset4 = dataset4[['_id', 'paid_at', 'amount_paid', 'marketplace_fee', 'ip_location', 'paid_at_date', 'paid_month', 'paid_hour', 'state']]
dataset4
dataset4['paid_at_date'] = dataset4['paid_at_date'].astype(str)
dataset4.head()
vendas_grouped = dataset4.groupby(by='paid_at_date').agg(sum)

vendas_grouped = vendas_grouped.sort_values(by='paid_at_date')

vendas_grouped = vendas_grouped[['amount_paid', 'marketplace_fee']]
vendas_grouped = vendas_grouped.reset_index()

vendas_grouped = vendas_grouped.reset_index()

vendas_grouped = vendas_grouped.rename(columns={'index':'dias_passados'})
vendas_grouped
vendas_grouped['amount_paid'].plot(kind='box')
vendas_grouped['amount_paid'].describe()
vendas_grouped = vendas_grouped[vendas_grouped['amount_paid'] < 886]
X = vendas_grouped['dias_passados'].values.reshape(-1, 1)

y = vendas_grouped['amount_paid'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mostrar = pd.DataFrame({'Dados reais': y_test.flatten(), 'Previsao': y_pred.flatten()})
mostrar = mostrar.head(50)

mostrar.plot(kind='bar')
plt.figure()

plt.scatter(X_test, y_test, color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
RMSE = mean_squared_error(y_test, regressor.predict(X_test))**0.5

RMSE
futuro = []



inicio = 171

for i in range(720):

    inicio+=1

    futuro.append(inicio)



futuro = pd.DataFrame(futuro)

futuro = np.array(futuro)
predicoes = pd.DataFrame(regressor.predict((futuro)))
datelist = pd.date_range(start='2020-12-30', periods=720)

predicoes['data'] = datelist

predicoes.columns = 'vendas', 'data'

predicoes = predicoes[['data', 'vendas']]
predicoes