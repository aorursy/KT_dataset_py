import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression  





#Estou selecionado só 10mil para ir mais rápido

df_=pd.read_csv('/kaggle/input/exemplo-regresso-apostas/under.csv')[:10000]



todas_colunas=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']



#Nossa função fit recebe um um string tamanho 16 de 0s e 1s, e retorna a lucratividade da combinação

def somaLog(codigo_genetico):

    global df_, todas_colunas

    

    #Quais colunas que entraram na regressão

    #colunas='A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P'

    colunas=','.join([ todas_colunas[i] for i,e in enumerate([int(c) for c in codigo_genetico]) if e])

    

    #Fitra o df baseado nas colunas

    df=df_[(colunas+',PL').split(',')]





    SLs=[] 

    for i in range(20):

        #Embaralha o dataframe baseado no random_state i 

        df=df.sample(frac=1, random_state=i)



        #Divide em 2 mil linhas para teste e o restante treinamento

        df_test,df_train=df[:2000],df[2000:]



        #Os Xs são todas as colunas exceto a PL que será o Y

        X_train,Y_train = df_train.loc[:,(df_train.columns!='PL') ], df_train.PL

        X_test, Y_test  = df_test.loc[:,(df_test.columns!='PL') ], df_test.PL



        #Treina a regressão os dados de treinamento

        reg=LinearRegression().fit(X_train,Y_train)



        #Veifica a lucratividade nos dados de teste

        SLs+=[sum(np.log(1+y*y_pred) for y_pred,y in zip(reg.predict(X_test),Y_test) if y_pred>0 ) ]





    #Mostra a lucrativida média e colunas selecionadas que deram origem a essa lucratividade

    return np.mean(SLs)





#Exemplo de fit

somaLog('1111010011110001')
#Configurações

TAM_POP=20  #Tamano da população (número para para não zuar o barraco, ok :)

N_BITS=len(todas_colunas)  #´Números de 0s e 1s do cromossomo

TAXA_DE_REPRODUCAO=0.95

TAXA_DE_MUTACAO=0.05



#Gera a população incial

pop=[]

for _ in range(TAM_POP):

    code=''.join([ str(np.random.randint(2)) for i in range(N_BITS)  ])

    pop+=[ {'code':code, 'fit':somaLog(code) } ]



pop
for n_gera in range(20):

    codes=[]

    

    #Para cada 2 indivudos gera novos 2 codigos genéticos a através do cruzamentos dada taxa de reprodução

    for pai,mae in zip(pop[:TAM_POP//2],pop[TAM_POP//2:]):

        corte=1+np.random.randint(N_BITS-1)

        if np.random.random()<TAXA_DE_REPRODUCAO:

            codes+=[pai['code'][:corte]+mae['code'][corte:corte+(N_BITS//2)]+pai['code'][corte+(N_BITS//2):],

                    mae['code'][:corte]+pai['code'][corte:corte+(N_BITS//2)]+mae['code'][corte+(N_BITS//2):] ]

        else:

            codes+=[pai['code'], mae['code']]



    #Para cada codigo genetico, toma bit a bit, troca os 0s por 1s, ou vice e versa dada taxa de mutação

    codes=[''.join([str(int(not(int(bit)))) if np.random.random()<TAXA_DE_MUTACAO  else bit for bit in code]) for code in codes ]





    #Adiciona os individuos a população, nesse momento pop tem 2*TAM_POP individuos

    pop+=[ {'code':code, 'fit':somaLog(code)} for code in codes]





    #Seleção por metodos do torneio para que pop tenha exatos TAM_POP individuos 

    pop=[ind1 if ind1['fit']>ind2['fit'] else ind2 for ind1,ind2 in zip(pop[:TAM_POP],pop[TAM_POP:])]



    #Embaralha os individuos da população

    np.random.shuffle(pop)



    print('Gera#:',n_gera,'Melhor fit:', max([ ind['fit'] for ind in pop ] )  )





print('Melhor combinação de colunas:', ','.join([ todas_colunas[i] for i,e in enumerate([int(c) for c in sorted(pop, key=lambda x: -x['fit'])[0]['code']]) if e]))