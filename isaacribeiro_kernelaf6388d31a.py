# Import das dependências básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Definição dos Indicadores
# VR = IC.BUS.EASE.XQ: Ease of doing business index - Parametro Qualitativo Ordinal (Categórico)
df_indicadores = pd.read_csv('../input/doing-business-csv-zip-108-kb-/DBSeries.csv')
df_indicadores_details = df_indicadores[['Series Code','Indicator Name', 'Long definition']]

# Renomeaia as colunas para serem acessíveis como Data Frames
df_indicadores_details.columns = ['IndicatorCode','Name','Definition']

# Ajusta a coluna 'IndicatorCode' removendo todos os espaços inválidos
df_indicadores_details['IndicatorCode'] = df_indicadores_details['IndicatorCode'].str.strip() 
df_indicadores_details.head(15)

# Carga de dados e separação randômica entre conjunto para Treino e para Test
# Modelo construído com base no ano de 2017
df = pd.read_csv('../input/doing-business-csv-zip-108-kb-/DBData.csv')

# Remoção das colunas referentes ao intervalo entre 2005 e 2016. 
# Pela pureza dos dados, esta analise basea-se apenas nos dados referentes a 2017
df_2017 = df[['Country Code','Indicator Code', '2017']]

# Renomeaia as colunas para serem acessíveis como Data Frames
df_2017.columns = ['Country', 'IndicatorCode', 'Value']

#### Operação básicas de ajuste do Dataset

# Define a coluna 'VALUE' como numérica
# Apenas os indicadores base, descritos no bloco 'Definição dos Indicadores', serão válidos na construção do modelo.
# Todos os demais são removidos
df_2017_indicators = df_2017['IndicatorCode']
df_valid_indexes = df_2017_indicators.isin(df_indicadores_details['IndicatorCode'])
df_2017_valid_entries = df_2017[df_valid_indexes] 

df_2017_valid_entries.head(10)
# Reorganiza a distribuição dos dados
# Em síntese, os dados originais foram agrupados tendo como chave o 'Country' e o 'Indicator'
# Para o nosso modelo, utilizaremos apenas o Country como chave, e cada 'Indicator' passa a ser uma coluna
df_2017_adjusted = df_2017_valid_entries.pivot(index = 'Country', columns = 'IndicatorCode', values = 'Value')
df_2017_adjusted.head()
# Remove os países que não receberam uma classificação válida
# Ou seja, cujo 'IC.BUS.EASE.XQ' = NaN
df_2017_clear = df_2017_adjusted.dropna(subset=['IC.BUS.EASE.XQ'])
df_2017_sorted = df_2017_clear.sort_values(by='IC.BUS.EASE.XQ')

# Ajusta as colunas e linhas válidas para o tipo esperado de acordo com a descrição de cada indicador:
#   IC.BUS.EASE.XQ: Ease of doing business index (1=easiest to 183...): CATEGORICO
#   IC.CRD.INFO.XQ: Depth of credit information index (0=low to 6=...): CATEGORICO
#   IC.LGL.CRED.XQ: Credit: Strength of legal rights index (0=weak...): CATEGORICO

# O agrupamento será feito em ranges, por essa razão, a variável 'IC.BUS.EASE.XQ' ainda não será convertida ainda 
#df_2017_sorted['IC.BUS.EASE.XQ'] = df_2017_sorted['IC.BUS.EASE.XQ'].astype('category') 
df_2017_sorted['IC.CRD.INFO.XQ'] = df_2017_sorted['IC.CRD.INFO.XQ'].astype('category') 
df_2017_sorted['IC.LGL.CRED.XQ'] = df_2017_sorted['IC.LGL.CRED.XQ'].astype('category')

df_2017_sorted.head(10)
# O objetivo deste modelo é dividir entre países com facilidade para a execução de negócios de acordo com a sua colocação
# no ranking do banco mundial

_50th_percentile = df_2017_sorted.quantile(.5)

# Os resultados serão agrupados na nova coluna "Class"
df_2017_sorted['Class'] = np.where(df_2017_sorted['IC.BUS.EASE.XQ'] <= _50th_percentile['IC.BUS.EASE.XQ'], 1, 0)

#df_2017_sorted['Class'] = df_2017_sorted['Class'].astype('category')
#df_2017_sorted['IC.BUS.EASE.XQ'] = df_2017_sorted['IC.BUS.EASE.XQ'].astype('category') 

# Com isso, obteremos uma distribuição normal dos dados
#n, bins, patches = plt.hist(df_2017_sorted['Class'], bins=2, alpha=0.75)
#plt.title('Distribuição da Variável Resposta')
#plt.xlabel('Class')
#plt.ylabel('Frequência')
#plt.show()
# Analise Exploratória dos Dados
# IC.BUS.EASE.XQ: Ease of doing business index (1=easiest to 183...): CATEGORICO
# IC.CRD.INFO.XQ: Depth of credit information index (0=low to 6=...): CATEGORICO
# IC.CRD.PRVT.ZS: Private credit bureau coverage (% of adults): 
# IC.CRD.PUBL.ZS: Public credit registry coverage (% of adults)
# IC.GE.NUM: Procedures required to connect to electricity ...
# IC.ISV.DURS: Time to resolve insolvency (years)
# IC.LGL.CRED.XQ: Credit: Strength of legal rights index (0=weak...): CATEGORICO
# IC.REG.DURS: Time required to start a business (days)
# IC.REG.PROC: Procedures required to start a business (number)
# IC.TAX.DURS: Time to prepare and pay taxes (hours)
# IC.TAX.PAYM: Tax payments (number)
# IC.TAX.TOTL.CP.ZS: Total tax rate (% of profit)   

# Agrupamentos reference a variavel Resposta
df_2017_sorted.groupby('Class').mean()
# IC.CRD.INFO.XQ: Depth of credit information index (0=low to 6=...): CATEGORICO
df_2017_sorted.groupby('IC.CRD.INFO.XQ').mean()
# IC.LGL.CRED.XQ: Credit: Strength of legal rights index (0=weak...): CATEGORICO
df_2017_sorted.groupby('IC.LGL.CRED.XQ').mean()
# Matriz de Correlação entre as variáveis núméricas
df_2017_sorted.corr()
# Plot da Matriz de Correlação acima
plt.matshow(df_2017_sorted.corr())
plt.title('Matriz de Correlação')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()
# Divisão entre dados de treino e de teste pseudo-aleatória
# 70% -> Dados de Treino
# 30% -> Dados de Teste

# Remove linhas com entradas inválidas: NaN
df_2017_model = df_2017_sorted.dropna()

X = df_2017_model.loc[:,np.logical_and(df_2017_model.columns != 'Class',df_2017_model.columns != 'IC.BUS.EASE.XQ')]
y = df_2017_model.loc[:,df_2017_model.columns == 'Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=11)
# Redução da Dimensionalidade 
# Feature Selection - Definição das melhores candidatas para classificar um país quanto a sua facilidade para negócios

# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Selecionaremos apenas as 3 melhores features
rfe = RFE(logreg, 5)
rfe = rfe.fit(X_train, Y_train.values.ravel())
selected_features = X_train.columns[rfe.support_]
print(selected_features)
# Ajuste com as features selecionadas
X = X_train[selected_features]
y = Y_train['Class']
# Implementação do Modelo
import statsmodels.api as sm
logit_model = sm.Logit(Y_train, X.astype(float))
result = logit_model.fit()
print(result.summary2())
# Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=22)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg)
y_pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# ROC - Receiver Operating Characteristic

# é uma representação gráfica que ilustra o desempenho (ou performance) 
# de um sistema classificador binário e como o seu limiar de discriminação é variado.

# https://pt.wikipedia.org/wiki/Caracter%C3%ADstica_de_Opera%C3%A7%C3%A3o_do_Receptor

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Positivos Verdadeiros')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# CART - Classification and Regression Tree

from sklearn import tree

# Vamos definir o algoritmo, usando como critério entropia, indicando o número de atributos =4 e o número mínimo de folhas =5
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4,
                                   min_samples_leaf=5)

# Aplicamos o modelo aos dados de treino
clf = clf.fit(X,y)

# vamos medir a proporção de acertos com uma função
def acuracidade(tree,X_test,Y_test):
    predict = clf.predict(X_test)
    erro = 0.0
    for x in range(len(predict)):
        if predict[x] != Y_test[x]:
            erro += 1.
    acuracy = (1-(erro/len(predict)))
    return acuracy

acuracidade(clf,X_test,y_test)
y_pred_dt = clf.predict(X_test)
y_pred_dt
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_dt)
print(confusion_matrix)     
# Visualização da Árvore
# Necessita a instalação do GraphViz 2.38
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
          (0,0,255)) # RGB desired col
img.save('sample-out.png')
PImage("sample-out.png")
