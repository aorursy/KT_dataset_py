# coding: utf-8

# Código ensinado no curso de python 2 do instituto Prandiano. 

# Pacotes:
import numpy as np
import pandas as pd
import pickle as pkl


def svm(banco_de_dados):
    '''
    :param banco_de_dados: Banco de Dados de input
    return: A: Matrix de Coeficientes [A],
    B: vector das Saídas [B],
    X: Coeficientes da solução do sistema de equações da matriz A,
    '''

    # valor_c: coeficiente de ajuste
    valor_c = 0.1

    # tolerancia: Tolerancia para a matriz ser ou nao inversivel... Se detA < tolerancia não é inversível. (Somente os valores válidos para o cálculo)
    tolerancia = 0.9


    bd_auxiliar = banco_de_dados
    
    # Remover a útlima coluna - coluna com os resultados experimentais da classificação: Coluna HONESTO
    # axis = 1 indica que irá remover uma coluna.
    # bd_auxiliar.shape[1]: número de colunas. Axis =1 remove no sentido da coluna. Se fosse linha usaria axis = 0.
    # Veja que o índice da última coluna que queremos remover (HONESTO) é o tamanho de colunas - 1, pois o índice começa em zero.
    bd_auxiliar = bd_auxiliar.drop([bd_auxiliar.columns[bd_auxiliar.shape[1] - 1]],
                                   axis = 1)
    
    # Armazenar somente os valores, ou seja, sem os títulos das colunas: Aqui você transforma o dataFrame numa array.
    valores = bd_auxiliar.values
    
    num_de_variaveis = valores.shape[1]
    num_de_licoes = valores.shape[0]
    
    # Armazenar a ultima coluna do banco de dados da variável banco_de_dados, resultados experimentais:
    # Estou pegando todas as linhas, porém apenas 10 primeiras das 11 colunas originais do bd.
    # iloc: integer localization.
    # loc: localiza via texto.
    # shape[0] -> número de linhas do data frame
    # shape[1] -> número de colunas do data frame
    B = banco_de_dados.iloc[:, banco_de_dados.shape[1] - 1]
    
    # num_de_colunas_faltantes: número de colunas faltantes para transformar a matrix de coeficientess A numa matriz quadrada.
    # A ideia é construir uma matriz quadrada. Entao vamos ter que criar colunas preenchidas apenas com 1. Essas colunas faltantes sao as linhas - colunas, ou seja, liçoes menos variaveis.
    if num_de_licoes > num_de_variaveis:
        num_de_colunas_faltantes = num_de_licoes - num_de_variaveis
    elif num_de_licoes == num_de_variaveis:
        num_de_colunas_faltantes = 0
    elif num_de_licoes < num_de_variaveis:
        print("O número de linhas no banco de dadods é " +
              "menor que o número de colunas, ou seja, o" + 
              "banco de dados possui poucos dados quando" + 
              "comparado com o número de variáveis (colunas) " + 
              "Adicionar mais registros (linhas) no banco de dados se for possível.")

    # Caso mais comum com lições maiores que variáveis.
    if num_de_colunas_faltantes != 0:
        
        matrix = np.zeros((num_de_licoes, num_de_colunas_faltantes))
        
        n = 0
        
        # Na variável matrix, são preenchidas as colunas faltantes, sendo que a primeira coluna deverá ser preenchida
        # com 'uns', a segunda coluna com 'dois' e assim por diante
        for k in range(0, num_de_colunas_faltantes, 1):
            n = n + 1
            for i in range(0, num_de_licoes, 1):
                matrix[i, k] = n
        # na variavel matrix serao armazenadas as colunas faltantes conforme descrito no comentários acima.
        
        # Acoplamento do banco de dados (com colunas faltantes) com as colunas preenchidas e que devem ser acoplados ao bd
        valores = np.hstack((valores, matrix))
        
        # Atualização do número de colunas
        num_de_variaveis = valores.shape[1]
        
        # Armazenar matrix [A]
        A = np.zeros((num_de_licoes, num_de_variaveis))
        
    elif num_de_colunas_faltantes == 0:
        # Armazenar matrix [A]
        A = np.zeros((num_de_licoes, num_de_variaveis))
        
    # A variavel abaixo vai armazenar o valor do 'c' para realizar a solução do modelo, este valor vem do main_svm como um input.
    # Este valor é fundamental, principalmente, quando é necessário incluir colunas de uns, ou dois e assim por diante.
    c = float(valor_c)

    # Cada elemento da matriz A é o produto escalar (inner) da primeira linha do banco de dados com a primeira linha, depois primeira com segunda, etc..
    for i in range(0, A.shape[0], 1):
        for j in range(0, A.shape[1], 1):
            A[i,j] = np.inner(valores[i], valores[j]) + c ** 2 # inner = É o produto escalar entre dois vetores.


    # O SVM depende que a matriz quadrada seja inversível, ou seja, DET A != 0. Portanto, primeira coisa a fazer é o teste.
    # Para fins práticos, qualquer determinante de A próximo a zero, ou seja, < tolerância, invalida o método! Nós definimos a tolerância para isso.
    det_A = np.linalg.det(A)
    if np.abs(det_A) <= tolerancia:
        print("")
        print("Observacao: O determinante da Matrix de coeficientes"
                "[A] é muito próximo de 0 zero: ], ", det_A,
                " . Portanto, o método pode não ser adequado.")
        print("")

    # Armazenar matrix [X]
    inv_A = np.linalg.inv(A)
    # matmul é multiplicação de matrizes
    X = np.matmul(inv_A, B)
    
    A = np.array(A)
    B = np.array(B)
    # Vetor de coeficientes solução. Contém o valor dos parâmetros.
    # VERIFICAR SE OS PARAMETROS SAO ESTATISTICAMENTE DIFERENTE DE ZERO.
    # P-valor dos parâmetros
    X = np.array(X)
    
    # matrix_solucao é  a variável que armazena a multiplicaçao de cada linha do BD com o vetor [x] de soluções do sistema de equações
    
    matrix_solucao = np.zeros((num_de_licoes, num_de_variaveis))
    for i in range(0, num_de_licoes, 1):
        # Aqui vamos preencher a matriz solução multiplicando os valores dos coeficientes X pelos valores
        matrix_solucao[i, :] = np.multiply(valores[i, :], X[i])
        
    # soma_vector_C = coef linear da solucao do modelo
    soma_vector_C = X.sum(axis=0)
    
    # a variavel vector_solucao armazena os resultados obitdos dos coeficientes angulares

    # Declara vetor solucao
    vector_solucao = np.zeros(num_de_licoes)
    # Abastece o vetor solucao com s soma das linhas da matriz solucao.
    vector_solucao = matrix_solucao.sum(axis=0)

    print(" Este programa gera o seguinte vetor de resultados: [B, valores, vector_solucao, soma_vector_C] \n"
          " valores: Matriz contendo os valores do banco de dados (sem a saída). \n "
          "vector_solucao: Coeficientes angulares da função teórica. \n "
          "soma_vector_C: valor do coeficente linear. \n")

    return [B, valores, vector_solucao, soma_vector_C]
import numpy as np
import pandas as pd
# Carregamento e Armazenamento:

# Caminho para os dados
path = "../input/covid19/dataset.xlsx"

# Lê os dados do excel.
data_original = pd.read_excel(path, encoding='UTF-8', delimiter=';' , sheet_name="All")

# Cópia do banco original, para NÃO alterar o primeiro, caso seja necessário utilizá-lo posteriormente no código.
data_mod = data_original.copy()
# O banco de dados possui 111 colunas e 5644 linhas (que por motivos didáticos também chamaremos de lições)
data_mod.shape
data_mod.head()
# Criando um dataFrame que será usado na análise de NaN
df_analise = pd.DataFrame(columns=['variavel_name', 'variavel_qtd_NaN', 'variavel_tipo', 'variavel_categorias'])
df_analise
# Criando uma lista com o nome das variáveis do banco original
coluna_name_list = list(data_mod.columns.values)
# Agora vamos criar um dataFrame em que exibimos o nome da variável do banco original, a quantidade de vazios desta variável, o tipo desta variável.
# Também temos uma coluna final que irá armazenar uma lista com o nome das possibilidades que cada variável não numérica assume.
for i in coluna_name_list:
    lista=[i, data_mod[i].isna().sum() , data_mod[i].dtypes, "variavel numérica"]
    df_length = len(df_analise)
    df_analise.loc[df_length] = lista
# CONCLUSÃO

# As seguintes variáveis do nosso banco possuem todos os valores vazios e deverão ser removidas
variaveis_vazias = df_analise[df_analise["variavel_qtd_NaN"]==data_mod.shape[0]]
variaveis_vazias
# Variavel_name passa a ser o índice do dataFrame
df_analise.set_index('variavel_name', inplace=True)
# Agora para as variáveis do tipo "object" ou seja, categóricas, vamos atribuir uma lista com valores únicos que ela assume no banco original.
for i in coluna_name_list:
    if(df_analise.loc[i, "variavel_tipo"] == "object"):
        df_analise.loc[i, "variavel_categorias"] = list(data_mod[i].unique())
    else:
        pass
# CONCLUSÃO

# Variáveis com valores categóricos que precisam ser transformados em numéricos.
df_variavies_qualitativas = df_analise[df_analise["variavel_tipo"]=="object"]
df_variavies_qualitativas
# 4.0) Transformar os valores "not_done" e "Não realizado" em vazios

data_mod = data_mod.replace("not_done", np.NaN).replace("Não Realizado", np.NaN)
# Primeiro vamos montar uma lista com o nome das variáveis vazias
lista_variaveis_vazias = list(variaveis_vazias["variavel_name"])
# Agora para cada nome de variável vamos remover do banco de dados
for i in lista_variaveis_vazias:
    del data_mod[i]
# Criar uma coluna adicional que irá armazenar a quantidade de NaNs que determina linha apresenta.
data_mod["isNaN"] = ""
# Agora vamos analisar cada coluna referente a um teste laboratorial e verificar se ela possui NaN. 
# Assim, a coluna "isNaN" contém o número de vezes que a linha possui o valor NaN.
contador=0
# Avalia se todos os valores desta linha são NA. Se forem, no fim alimenta a coluna  "isNaN" com SIM. Se não, muda de linha e não faz nada.
for lin in range(data_mod.shape[0]):
    for col in range(6, data_mod.shape[1]-1, 1):
        if(pd.isnull(data_mod.iloc[lin, col])):
            contador = contador + 1
        else:
            pass
            # Muda de linha
    data_mod.iloc[lin,106] = contador
    contador=0
data_mod['isNaN'].describe()
# Vamos trabalhar apenas com as linhas que apresentaram pelo menos XXX dos 100 testes com algum valor não nulo.
data_sem_vazios = (data_mod.loc[data_mod['isNaN'] < 80]).copy()
data_sem_vazios.reset_index(drop=True)
# Atribuir aos NaN o valor numérico -10. Este valor é bom pois é distinto dos demais, dado que o menor valor que aparece no dataset é -5,9.
# Importante para o modelo entender que para estes casos não foi realizado teste.
data_sem_vazios = data_sem_vazios.fillna(-10)
# Remove colunas desnecessárias
del data_sem_vazios["isNaN"]
del data_sem_vazios["Patient ID"]
# As seguintes variaveis qualitativas receberão valores numéricos arbitrários. Esta é a maneira com a qual o modelo SVM funciona.
df_variavies_qualitativas
# Variáveis binárias
print("Respiratory Syncytial Virus", data_sem_vazios["Respiratory Syncytial Virus"].unique())
print("Influenza A", data_sem_vazios["Influenza A"].unique())
print("Influenza B", data_sem_vazios["Influenza B"].unique())
print("Parainfluenza 1", data_sem_vazios["Parainfluenza 1"].unique())
print("CoronavirusNL63", data_sem_vazios["CoronavirusNL63"].unique())
print("Rhinovirus/Enterovirus", data_sem_vazios["Rhinovirus/Enterovirus"].unique())
print("Coronavirus HKU1", data_sem_vazios["Coronavirus HKU1"].unique())
print("Parainfluenza 3", data_sem_vazios["Parainfluenza 3"].unique())
print("Chlamydophila pneumoniae", data_sem_vazios["Chlamydophila pneumoniae"].unique())
print("Adenovirus", data_sem_vazios["Adenovirus"].unique())
print("Parainfluenza 4", data_sem_vazios["Parainfluenza 4"].unique())
print("Coronavirus229E", data_sem_vazios["Coronavirus229E"].unique())
print("CoronavirusOC43", data_sem_vazios["CoronavirusOC43"].unique())
print("Inf A H1N1 2009", data_sem_vazios["Inf A H1N1 2009"].unique())
print("Bordetella pertussis", data_sem_vazios["Bordetella pertussis"].unique())
print("Metapneumovirus", data_sem_vazios["Metapneumovirus"].unique())
print("Parainfluenza 2", data_sem_vazios["Parainfluenza 2"].unique())
data_sem_vazios["Influenza A"]= data_sem_vazios["Influenza A"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Influenza B"]= data_sem_vazios["Influenza B"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Respiratory Syncytial Virus"]= data_sem_vazios["Respiratory Syncytial Virus"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["Parainfluenza 1"]= data_sem_vazios["Parainfluenza 1"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["CoronavirusNL63"]= data_sem_vazios["CoronavirusNL63"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Rhinovirus/Enterovirus"]= data_sem_vazios["Rhinovirus/Enterovirus"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Coronavirus HKU1"]= data_sem_vazios["Coronavirus HKU1"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["Parainfluenza 3"]= data_sem_vazios["Parainfluenza 3"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["Chlamydophila pneumoniae"]= data_sem_vazios["Chlamydophila pneumoniae"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Adenovirus"]= data_sem_vazios["Adenovirus"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Parainfluenza 4"]= data_sem_vazios["Parainfluenza 4"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["Coronavirus229E"]= data_sem_vazios["Coronavirus229E"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["CoronavirusOC43"]= data_sem_vazios["CoronavirusOC43"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Inf A H1N1 2009"]= data_sem_vazios["Inf A H1N1 2009"].replace("not_detected", 0).replace("detected", 1)

data_sem_vazios["Bordetella pertussis"]= data_sem_vazios["Bordetella pertussis"].replace("not_detected", 0).replace("detected", 1)
                                                                                                
data_sem_vazios["Metapneumovirus"]= data_sem_vazios["Metapneumovirus"].replace("not_detected", 0).replace("detected", 1) 

data_sem_vazios["Parainfluenza 2"]= data_sem_vazios["Parainfluenza 2"].replace("not_detected", 0).replace("detected", 1) 
# Variáveis binárias do tipo: positive/negative

print("SARS-Cov-2 exam result", data_sem_vazios["SARS-Cov-2 exam result"].unique())
print("Influenza B, rapid test", data_sem_vazios["Influenza B, rapid test"].unique())
print("Influenza A, rapid test", data_sem_vazios["Influenza A, rapid test"].unique())
print("Strepto A", data_sem_vazios["Strepto A"].unique())
data_sem_vazios["SARS-Cov-2 exam result"]= data_sem_vazios["SARS-Cov-2 exam result"].replace("negative", 0).replace("positive", 1)

data_sem_vazios["Influenza B, rapid test"]= data_sem_vazios["Influenza B, rapid test"].replace("negative", 0).replace("positive", 1)

data_sem_vazios["Influenza A, rapid test"]= data_sem_vazios["Influenza A, rapid test"].replace("negative", 0).replace("positive", 1)

data_sem_vazios["Strepto A"]= data_sem_vazios["Strepto A"].replace("negative", 0).replace("positive", 1).replace("not_done", -10)
# Variáveis do tipo absent/presente

print("Urine - Esterase", data_sem_vazios["Urine - Esterase"].unique())
print("Urine - Hemoglobin", data_sem_vazios["Urine - Hemoglobin"].unique())
print("Urine - Bile pigments", data_sem_vazios["Urine - Bile pigments"].unique())
print("Urine - Ketone Bodies", data_sem_vazios["Urine - Ketone Bodies"].unique())
print("Urine - Nitrite", data_sem_vazios["Urine - Nitrite"].unique())
print("Urine - Urobilinogen", data_sem_vazios["Urine - Urobilinogen"].unique())
print("Urine - Protein", data_sem_vazios["Urine - Protein"].unique())
print("Urine - Hyaline cylinders", data_sem_vazios["Urine - Hyaline cylinders"].unique())
print("Urine - Granular cylinders", data_sem_vazios["Urine - Granular cylinders"].unique())
print("Urine - Yeasts", data_sem_vazios["Urine - Yeasts"].unique())
data_sem_vazios["Urine - Esterase"]= data_sem_vazios["Urine - Esterase"].replace("absent", 0).replace("not_done", -10)

data_sem_vazios["Urine - Hemoglobin"]= data_sem_vazios["Urine - Hemoglobin"].replace("absent", 0).replace("present", 1).replace("not_done", -10)

data_sem_vazios["Urine - Bile pigments"]= data_sem_vazios["Urine - Bile pigments"].replace("absent", 0).replace("not_done", -10)

data_sem_vazios["Urine - Ketone Bodies"]= data_sem_vazios["Urine - Ketone Bodies"].replace("absent", 0).replace("not_done", -10)

data_sem_vazios["Urine - Nitrite"]= data_sem_vazios["Urine - Nitrite"].replace("not_done", -10)

data_sem_vazios["Urine - Urobilinogen"]= data_sem_vazios["Urine - Urobilinogen"].replace("normal", 0).replace("not_done", -10)

data_sem_vazios["Urine - Protein"]= data_sem_vazios["Urine - Protein"].replace("absent", 0).replace("not_done", -10)

data_sem_vazios["Urine - Hyaline cylinders"]= data_sem_vazios["Urine - Hyaline cylinders"].replace("absent", 0)

data_sem_vazios["Urine - Granular cylinders"]= data_sem_vazios["Urine - Granular cylinders"].replace("absent", 0)

data_sem_vazios["Urine - Yeasts"]= data_sem_vazios["Urine - Yeasts"].replace("absent", 0)
# Variáveis categóricas
print("Urine - Aspect", data_sem_vazios["Urine - Aspect"].unique())
print("Urine - Crystals", data_sem_vazios["Urine - Crystals"].unique())
print("Urine - Color", data_sem_vazios["Urine - Color"].unique())
data_sem_vazios["Urine - Aspect"]= data_sem_vazios["Urine - Aspect"].replace("clear", 0).replace("cloudy", 1).replace("altered_coloring", 2).replace("lightly_cloudy", 3)

data_sem_vazios["Urine - Crystals"]= data_sem_vazios["Urine - Crystals"].replace("Ausentes", 0).replace("Urato Amorfo --+", 1).replace("Oxalato de Cálcio +++", 2).replace("Oxalato de Cálcio -++", 3).replace("Urato Amorfo +++", 4)

data_sem_vazios["Urine - Color"]= data_sem_vazios["Urine - Color"].replace("light_yellow", 0).replace("yellow", 1).replace("orange", 2).replace("citrus_yellow", 3)
# Variáveis categóricas mas que apenas precisa transformar em int

print("Urine - pH", data_sem_vazios["Urine - pH"].unique())
print("Urine - Leukocytes", data_sem_vazios["Urine - Leukocytes"].unique())
data_sem_vazios["Urine - pH"]= data_sem_vazios["Urine - pH"].replace("Não Realizado", -10)
data_sem_vazios["Urine - pH"] = data_sem_vazios["Urine - pH"].astype(float)

data_sem_vazios["Urine - Leukocytes"]= data_sem_vazios["Urine - Leukocytes"].replace("<1000", 0)
data_sem_vazios["Urine - Leukocytes"] = data_sem_vazios["Urine - Leukocytes"].astype(float)
# Vamos criar uma base de dados específica para rodar a tarefa 1:
data_tarefa1 = data_sem_vazios.copy()
data_tarefa1
# Vamos atribuir à variável explicada -100 quando a pessoa não testar positivo para Covid e 100 quando for positivo.
data_tarefa1["SARS-Cov-2 exam result"]= data_tarefa1["SARS-Cov-2 exam result"].replace(0, -100).replace(1, 100)
data_tarefa1["SARS-Cov-2 exam result"].describe()
# Deletar colunas desnecessárias:
del data_tarefa1["Patient addmited to regular ward (1=yes, 0=no)"]
del data_tarefa1["Patient addmited to semi-intensive unit (1=yes, 0=no)"]
del data_tarefa1["Patient addmited to intensive care unit (1=yes, 0=no)"]
# Agora precisamos levar a coluna da variável explicada para a última posição da direita, pois é assim que o programa recebe. 
# Criamos uma função para fazer isso:
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])
# E com base na função definida anteriormente vamos mandar a coluna que quermos para depois da última coluna, que no caso é: ctO2 (arterial blood gas analysis)
data_tarefa1 = movecol(data_tarefa1, 
             cols_to_move=["SARS-Cov-2 exam result"], 
             ref_col='ctO2 (arterial blood gas analysis)',
             place='After')
# Reiniciano índices
data_tarefa1 = data_tarefa1.reset_index(drop=True)
data_tarefa1
# Vamos criar uma base de dados específica para rodar a tarefa 2:
data_tarefa2_aux = data_sem_vazios.copy()
# Vamos criar a coluna de resultados baseado na regra: se foi admitido apenas no ambulatório: -100. Se foi admitido em qualquer UTI: 100.
data_tarefa2_aux["admission_result"] = data_tarefa2_aux["Patient addmited to semi-intensive unit (1=yes, 0=no)"] +  data_tarefa2_aux["Patient addmited to intensive care unit (1=yes, 0=no)"] - (100*(data_tarefa2_aux["Patient addmited to regular ward (1=yes, 0=no)"]))

data_tarefa2_aux["admission_result"]= data_tarefa2_aux["admission_result"].replace(1, 100)
# Vamos desconsiderar as linhas em que o paciente não foi nem para ambulatório nem para UTI.
data_tarefa2 = (data_tarefa2_aux.loc[data_tarefa2_aux['admission_result'] != 0]).copy()
data_tarefa2
# Deletar colunas desnecessárias:
del data_tarefa2["SARS-Cov-2 exam result"]
del data_tarefa2["Patient addmited to regular ward (1=yes, 0=no)"]
del data_tarefa2["Patient addmited to semi-intensive unit (1=yes, 0=no)"]
del data_tarefa2["Patient addmited to intensive care unit (1=yes, 0=no)"]
# Reiniciano índices
data_tarefa2 = data_tarefa2.reset_index(drop=True)
data_tarefa2
[B, valores, vector_solucao, soma_vector_C] = svm(data_tarefa1)
# Verificação apenas dos valores do banco de dados que foi gerada a colução
pd.DataFrame(valores)
print("Coeficientes de Ajuste")
pd.DataFrame(vector_solucao)
# soma_vector_C = Coeficiente linear da Solução do modelo
print("Coeficiente Linear", soma_vector_C)
# Greando vetor soluções
y_teo = np.zeros(len(B))
for i in range(0, valores.shape[0], 1):
    y_teo[i] = np.matmul(valores[i], vector_solucao) + soma_vector_C
    # Podemos usar este trecho para dizer que qualquer coisa positiva é um tipo de solução, e negativo, outro tipo. 
    y_teo = np.where(y_teo > 0, 100, -100)


vetor_sol_tarefa1 = pd.DataFrame(y_teo)
vetor_sol_tarefa1
data_tarefa1.to_excel("resultado_tarefa1.xlsx")
vetor_sol_tarefa1.to_excel("vetor_sol_tarefa1.xlsx")
[B, valores, vector_solucao, soma_vector_C] = svm(data_tarefa2)
# Verificação apenas dos valores do banco de dados que foi gerada a colução
pd.DataFrame(valores)
print("Coeficientes de Ajuste")
pd.DataFrame(vector_solucao)
# soma_vector_C = Coeficiente linear da Solução do modelo
print("Coeficiente Linear", soma_vector_C)
# Greando vetor soluções
y_teo = np.zeros(len(B))
for i in range(0, valores.shape[0], 1):
    y_teo[i] = np.matmul(valores[i], vector_solucao) + soma_vector_C
    # Podemos usar este trecho para dizer que qualquer coisa positiva é um tipo de solução, e negativo, outro tipo. 
    y_teo = np.where(y_teo > 0, 100, -100)


vetor_sol_tarefa2 = pd.DataFrame(y_teo)
vetor_sol_tarefa2
data_tarefa2.to_excel("resultado_tarefa2.xlsx")
vetor_sol_tarefa2.to_excel("vetor_sol_tarefa2.xlsx")