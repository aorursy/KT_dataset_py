# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
path1 = '../input/candidatos-deputado-federal-e-estadual-2014/'
path2 = '../input/electoral-donations-brazil2014/'
df_candidatos = pd.read_csv(path1+'consulta_cand_2014_Brazil.csv', encoding='latin1', sep=';')
display(df_candidatos.head())
# Separa as colunas de interesse que são relevantes para a análise
cols_of_interest = ['SIGLA_UF','CODIGO_CARGO','DESCRICAO_CARGO','NOME_CANDIDATO','SEQUENCIAL_CANDIDATO',
                    'COD_SITUACAO_CANDIDATURA','DES_SITUACAO_CANDIDATURA','NUMERO_PARTIDO','CODIGO_LEGENDA',
                    'CODIGO_OCUPACAO','DESCRICAO_OCUPACAO','IDADE_DATA_ELEICAO','CODIGO_SEXO','DESCRICAO_SEXO',
                    'COD_GRAU_INSTRUCAO','DESCRICAO_GRAU_INSTRUCAO','CODIGO_ESTADO_CIVIL','DESCRICAO_ESTADO_CIVIL',
                    'CODIGO_COR_RACA','DESCRICAO_COR_RACA','CODIGO_NACIONALIDADE','DESCRICAO_NACIONALIDADE',
                    'DESPESA_MAX_CAMPANHA','COD_SIT_TOT_TURNO','DESC_SIT_TOT_TURNO']

df_candidatos = df_candidatos[cols_of_interest]

# Filtra apenas os candidatos a deputado (federal, estadual e distrital) 
df_candidatos = df_candidatos[(df_candidatos['CODIGO_CARGO'] >= 6) & 
                              (df_candidatos['CODIGO_CARGO'] <= 8)] 

# E somente candidaturas deferidas ou deferido com recurso
df_candidatos = df_candidatos[((df_candidatos['COD_SITUACAO_CANDIDATURA']) == 2) | 
                              ((df_candidatos['COD_SITUACAO_CANDIDATURA']) == 16)] 

quantidade_candidatos = df_candidatos[(df_candidatos['COD_SITUACAO_CANDIDATURA'] == 2) | 
                                      (df_candidatos['COD_SITUACAO_CANDIDATURA'] == 16)].shape[0]

quantidade_eleitos = df_candidatos[(df_candidatos['COD_SIT_TOT_TURNO'] == 2) | 
                                   (df_candidatos['COD_SIT_TOT_TURNO'] == 3)].shape[0]

print ('Total de candidatos a deputados com candidaturas deferidas: ', quantidade_candidatos)
print ('Total de candidatos eleitos:', quantidade_eleitos, ((quantidade_eleitos / quantidade_candidatos)*100),'%')
df_bem = pd.read_csv(path1+'bem_candidato_2014_Brazil.csv', encoding='latin1', sep=';')
df_bem.head()
df_soma_bens = df_bem.groupby(['SQ_CANDIDATO']).agg({'VALOR_BEM': lambda x: x.sum()}).reset_index()
df_soma_bens.head(10)
df_candidatos = df_candidatos.merge(df_soma_bens, left_on = 'SEQUENCIAL_CANDIDATO', right_on = 'SQ_CANDIDATO', how='left')
df_candidatos.drop('SQ_CANDIDATO', axis=1, inplace=True)
display(df_candidatos.head(5))
if df_candidatos.shape[0] == quantidade_candidatos:
    print('Merge OK')
df_doacoes = pd.read_csv(path2+'receitas_candidatos_2014_brasil.txt', encoding='latin1', sep=';', decimal=",")
df_doacoes = df_doacoes[df_doacoes['Cargo'].str.startswith('Deputado')]
df_bem.head(10)
df_doacoes['Fonte recurso'].unique()
df_doacoes.loc[(df_doacoes['Setor econômico do doador originário']!='#NULO') , 'Setor econômico do doador'] = df_doacoes['Setor econômico do doador originário']
df_doacoes.loc[(df_doacoes['CPF/CNPJ do doador originário']!='#NULO') , 'CPF/CNPJ do doador'] = df_doacoes['CPF/CNPJ do doador originário']
df_doacoes.loc[(df_doacoes['Nome do doador originário']!='#NULO') , 'Nome do doador'] = df_doacoes['Nome do doador originário']
df_doacoes.loc[(df_doacoes['Nome do doador originário (Receita Federal)']!='#NULO') , 'Nome do doador (Receita Federal)'] = df_doacoes['Nome do doador originário (Receita Federal)']
df_setores = df_doacoes.groupby(['Setor econômico do doador']).agg({'Valor receita': lambda x: x.sum()}).reset_index()
setores = list(df_setores.sort_values(by=['Valor receita'], ascending=[0])['Setor econômico do doador'])
# Carrega a tabela de cnaes 
df_cnae = pd.read_excel(path1+'Subclasses CNAE 2.2 - Estrutura.xls', header=None, skiprows=5)
# verifica se a primeiro coluna não tem valores
print (df_cnae[0].unique())
# então exclui a coluna
df_cnae.drop(0, axis=1, inplace=True)
# renomeia as colunas para ficar mais amigável
df_cnae.columns = ['secao', 'divisao', 'grupo', 'classe', 'subclasse', 'denominacao']
display(df_cnae)
def is_nan(x):
    return (x is np.nan or x != x)
# completa a tabela para adicionar as seções a todos os cnaes
for ind, secao in enumerate(df_cnae['secao']):
    if not is_nan(secao) and len(secao) == 1:
        secao_ok = secao
    else:
        df_cnae.loc[ind,'secao'] = secao_ok
# exclui as outras colunas que não são necessárias nesse momento.
df_cnae.drop('divisao', axis=1, inplace=True)
df_cnae.drop('grupo', axis=1, inplace=True)
df_cnae.drop('subclasse', axis=1, inplace=True)
df_cnae.drop('classe', axis=1, inplace=True)

display(df_cnae.head(10))
# exclui alguma denominação duplicada
df_cnae.drop_duplicates('denominacao', inplace=True)
# verifica se o merge ficou OK
total_doacoes = df_doacoes.shape[0]
df_doacoes = df_doacoes.merge(df_cnae, left_on='Setor econômico do doador', right_on='denominacao', how='left')
if total_doacoes == df_doacoes.shape[0]:
    print('Merge OK')
else:
    print(total_doacoes, df_doacoes.shape[0])
df_doacoes.drop('denominacao', axis=1, inplace=True)
display(df_doacoes.head(10))
# apenas um cnae não foi encontrado, dessa forma atribui manualmente 
df_doacoes.loc[df_doacoes['Setor econômico do doador'] == 
               'Comércio varejista de outros artigos de uso doméstico não especificados anteriormente','secao'] = 'G'
df_doacoes.loc[df_doacoes['Setor econômico do doador'] == '#NULO','secao'] = 'SETOR_NAO_IDENTIFICADO'
df_soma_doacoes = df_doacoes.groupby(['Sequencial Candidato','secao']).agg({'Valor receita': lambda x: x.sum()}).reset_index()
display(df_soma_doacoes)
# pivot do dataframe 
df_soma_doacoes = df_soma_doacoes.pivot(index='Sequencial Candidato', columns='secao')['Valor receita'].fillna(0)
renamed = {chr(i):'SETOR_'+chr(i) for i in range(ord('A'),ord('S')+1)} 
df_soma_doacoes.rename(columns=renamed, inplace=True)
display(df_soma_doacoes.head(5))
df_candidatos = df_candidatos.merge(df_soma_doacoes, left_on = 'SEQUENCIAL_CANDIDATO', right_index=True, how='left')
if df_candidatos.shape[0] == quantidade_candidatos:
    print('Merge OK')
print(df_doacoes['Tipo receita'].unique())
map_tp_receita = {'Recursos de pessoas jurídicas': 'TP_RECEITA_JURIDICA',
                  'Recursos de outros candidatos/comitês': 'TP_RECEITA_OUTRO',
                  'Recursos de partido político': 'TP_RECEITA_PARTIDO',
                  'Doações pela Internet': 'TP_RECEITA_INTERNET',
                  'Recursos de pessoas físicas': 'TP_RECEITA_FISICA',
                  'Recursos de origens não identificadas': 'TP_RECEITA_NAO_IDENTIFICADA',
                  'Recursos próprios': 'TP_RECEITA_PROPRIO',
                  'Rendimentos de aplicações financeiras' : 'TP_RECEITA_APLICACAO',
                  'Comercialização de bens ou realização de eventos': 'TP_RECEITA_EVENTO'
                 }
df_doacoes['Tipo receita map'] = df_doacoes['Tipo receita'].map(map_tp_receita)
print(df_doacoes['Tipo receita map'].unique())
df_soma_doacoes = df_doacoes.groupby(['Sequencial Candidato','Tipo receita map']).agg({'Valor receita': lambda x: x.sum()}).reset_index()

display(df_soma_doacoes.head(5))
df_soma_doacoes = df_soma_doacoes.pivot(index='Sequencial Candidato', columns='Tipo receita map')['Valor receita'].fillna(0)
display(df_soma_doacoes.head(5))
df_candidatos = df_candidatos.merge(df_soma_doacoes, 
                                    left_on = 'SEQUENCIAL_CANDIDATO', 
                                    right_index=True, how='left')
if df_candidatos.shape[0] == quantidade_candidatos:
    print('Merge OK')
df_soma_doacoes = df_doacoes.groupby(['Sequencial Candidato']).agg({'Valor receita': lambda x: x.sum()}).reset_index()
df_soma_doacoes.rename(columns={'Valor receita':'VALOR_RECEITA'}, inplace=True)
display(df_soma_doacoes.head(5))
df_candidatos = df_candidatos.merge(df_soma_doacoes,
                                    left_on = 'SEQUENCIAL_CANDIDATO', 
                                    right_on='Sequencial Candidato', 
                                    how='left')
df_candidatos.drop('Sequencial Candidato', axis=1, inplace=True)
if df_candidatos.shape[0] == quantidade_candidatos:
    print('Merge OK')
display(df_candidatos.head(30))
df_candidatos.columns.values
#Salva o Dataset
df_candidatos.to_csv('consolidado_eleicao_2014.csv', index=False)
# IF cod_sist_tot
df_candidatos['FOI_ELEITO'] = np.where((df_candidatos['COD_SIT_TOT_TURNO'] == 2) | (df_candidatos['COD_SIT_TOT_TURNO'] == 3), 1, 0)
display(df_candidatos.head(10))
predictors = ['NUMERO_PARTIDO', 'CODIGO_LEGENDA',
       'CODIGO_OCUPACAO', 'IDADE_DATA_ELEICAO',
       'CODIGO_SEXO', 'COD_GRAU_INSTRUCAO',
       'CODIGO_ESTADO_CIVIL', 'CODIGO_COR_RACA', 'DESPESA_MAX_CAMPANHA',
       'VALOR_BEM', 'SETOR_A', 'SETOR_B', 'SETOR_C', 'SETOR_D', 'SETOR_E',
       'SETOR_F', 'SETOR_G', 'SETOR_H', 'SETOR_I', 'SETOR_J', 'SETOR_K',
       'SETOR_L', 'SETOR_M', 'SETOR_N', 'SETOR_O', 'SETOR_P', 'SETOR_Q',
       'SETOR_R', 'SETOR_S', 'SETOR_NAO_IDENTIFICADO',
       'TP_RECEITA_APLICACAO', 'TP_RECEITA_EVENTO', 'TP_RECEITA_FISICA',
       'TP_RECEITA_INTERNET', 'TP_RECEITA_JURIDICA',
       'TP_RECEITA_NAO_IDENTIFICADA', 'TP_RECEITA_OUTRO',
       'TP_RECEITA_PARTIDO', 'TP_RECEITA_PROPRIO', 'VALOR_RECEITA']

predictors = ['NUMERO_PARTIDO', 'IDADE_DATA_ELEICAO',
       'COD_GRAU_INSTRUCAO','DESPESA_MAX_CAMPANHA',
       'VALOR_BEM','VALOR_RECEITA']
predictors = ['VALOR_BEM','VALOR_RECEITA']

colums_of_data = df_candidatos[predictors]
colums_of_data.describe()
data = df_candidatos

# Imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_imputed = my_imputer.fit_transform(data[predictors])

# Defining the target
y = data.FOI_ELEITO
# Creating the vector of independent variables
X = data_imputed

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
min_mae = 9999999999999999999
best_max_leaf = 0

for max_leaf_nodes in [5, 50, 500, 5000, 10000,20000,50000]:
    new_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    if new_mae < min_mae:
        min_mae = new_mae
        best_max_leaf = max_leaf_nodes
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, new_mae))
            
print("\n\nBEST MAX LEAF NODES: %d" %(best_max_leaf))
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes=best_max_leaf, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Feature Scaling | REMOVE THIS IF YOU WANT TO SEE THE ACTUAL VALUES OF THE TREE
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X_train = sc_X.fit_transform(X_train)
sc_X_test = sc_X.transform(X_test)


#Visualizing the Training set result
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#X_set, y_set = sc_X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 100000),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 100000))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.35, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == i, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('2014 Brazil Elections. Elected x Not Elected')
plt.xlabel(predictors[0])
plt.ylabel(predictors[1])
plt.legend()
plt.show()
# Export our trained model as a .dot file
from sklearn.tree import export_graphviz
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
with open("tree1.dot", 'w') as f:
    f = export_graphviz(classifier,
                             out_file=f,
                             max_depth = 10,
                             impurity = True,
                             feature_names = list(predictors),
                             class_names = ['Nao Eleito', 'Eleito'],
                             rounded = True,
                             filled= True )
# Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          'Brazil 2014 Election Results: VALOR RECEITA x VALOR_BEM', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('decision_tree_2014_Election.png')
PImage("decision_tree_2014_Election.png")