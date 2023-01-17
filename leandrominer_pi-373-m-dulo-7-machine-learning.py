from pandas_profiling import ProfileReport

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import numpy as np

from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression











pd.set_option('display.max_columns', None)
treino = pd.read_pickle('../input/machine-learning/treino.pkl')

teste = pd.read_pickle('../input/machine-learning/teste_aluno.pkl')
treino.columns
treino.head()
#prof = ProfileReport(treino)

#prof.to_file(output_file="output.html")
#Dropando as colunas que identicam os funcionarios. Bem como a origem do veículo

colunas_drop = ['agencia','funcionario','revendedora','montadora', "flag_telefone",'Current_pincode_ID',

                'flag_aadhar', 'flag_passaporte']

treino.drop(colunas_drop, axis=1, inplace=True)

teste.drop(colunas_drop, axis=1, inplace=True)
# função para corrigir os anos (61 é considerado como 2061, por exemplo)

def corrige_anos(arr):

    

    '''

    Essa função recebe um array de datas no formato string, seleciona se o final dela (ano)

    for menor que 20 (referência a 2020) e então formata como ano com inicio 19 ou 20 

    

    '''

    

    lista = []

    for i in arr:

        if int(i[6:]) < 20:

            lista.append(i[:6]+'20'+i[6:])

        else:

            lista.append(i[:6]+'19'+i[6:])

    return lista
# Convertendo o ano com a função criada

treino.nascimento = corrige_anos(treino.nascimento)

teste.nascimento = corrige_anos(teste.nascimento)
#Convertendo o nascimento para anos 

now = pd.Timestamp('now')

treino['idade'] = (now - pd.to_datetime(treino.nascimento,format='%d-%m-%Y')).astype('<m8[Y]')

teste['idade'] = (now - pd.to_datetime(teste.nascimento,format='%d-%m-%Y')).astype('<m8[Y]')

treino.drop('nascimento', axis =1, inplace=True)

teste.drop('nascimento', axis =1, inplace=True)
#Convertendo a data do contrato para dias



now = pd.Timestamp('now')

treino['dias_contrato'] = (now - pd.to_datetime(treino.data_contrato,format='%d-%m-%y')).dt.days

teste['dias_contrato'] = (now - pd.to_datetime(teste.data_contrato,format='%d-%m-%y')).dt.days

treino.drop('data_contrato', axis =1, inplace=True)

teste.drop('data_contrato', axis =1, inplace=True)
treino.dias_contrato
def str_para_mes(arr):

    

        

    '''

    Essa função recebe um array de datas no formato string: (5yrs 5mon). Retira a referência ao ano e ao mês e 

    retorna a soma entre os anos e meses como um array.    

    '''

    

    

    ano = []

    mes=[]

    arr = arr.str.replace('yrs ', '-').str.replace('mon','')

    for i in arr.index:

        ano.append(int(arr.loc[i].split('-')[0]))

        mes.append(int(arr.loc[i].split('-')[1]))





    return np.array(ano)*12+np.array(mes)

        

#Convertendo para mês

treino['tem_med_emp'] = str_para_mes(treino.tem_med_emp)

teste['tem_med_emp'] = str_para_mes(teste.tem_med_emp)
#Convertendo para mês

treino['tem_pri_emp'] = str_para_mes(treino.tem_pri_emp)

teste['tem_pri_emp'] = str_para_mes(teste.tem_pri_emp)
treino.columns
#Retirando a variável resposta e a id para o preprocessamento

treino_exp= treino.drop(['default', 'id_pessoa'], axis=1)
numericas = ['valor_emprestimo', 'custo_ativo', 'emprestimo_custo',

        'estado','score',  'pri_qtd_tot_emp', 'pri_qtd_tot_emp_atv',

       'pri_qtd_tot_def', 'pri_emp_abt', 'pri_emp_san', 'pri_emp_tom',

       'sec_qtd_tot_emp', 'sec_qtd_tot_emp_atv', 'sec_qtd_tot_def',

       'sec_emp_abt', 'sec_emp_san', 'sec_emp_tom', 'par_pri_emp',

       'par_seg_emp', 'nov_emp_6m', 'def_emp_6m', 'tem_med_emp', 'tem_pri_emp',

       'qtd_sol_emp', 'idade', 'dias_contrato']
treino_exp[numericas] = treino_exp[numericas].astype(float)
categoricas = ['emprego', 'flag_pan', 'flag_eleitor', 'flag_cmotorista',

       'score_desc']
#Pipeline para o pré-processmento



pipe_num = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'median' )),

    ('minmax', MinMaxScaler())

])



pipe_cat = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('encoder', OrdinalEncoder())

])



preproc = ColumnTransformer(transformers = [

    ('proc_cat', pipe_cat, categoricas),

    ('proc_num', pipe_num, numericas)

    

], n_jobs=-1)



pipe_final = Pipeline([

    ('proc', preproc),

])

treino_trans= pipe_final.fit_transform(treino_exp)
#Lista de colunas para renomear o DF

col = treino.columns.drop(['id_pessoa', 'default'])
# Transformando o array em DF

treino_trans = pd.DataFrame(treino_trans, columns = col)
treino_trans.head()
#Reinserindo as colunas de id e resposta

treino_trans.insert(0, column='id_pessoa' , value=treino.id_pessoa.values)

treino_trans.insert(1, column='default' , value=treino.default.values)
treino_trans.head()
teste_exp = teste.drop(['id_pessoa'], axis=1)
teste_exp = pipe_final.transform(teste_exp)
col = teste.columns.drop(['id_pessoa'])
teste_trans = pd.DataFrame(teste_exp, columns = col)
teste_trans.insert(0, column='id_pessoa' , value=teste.id_pessoa.values)
teste_trans.head()
teste_trans.shape