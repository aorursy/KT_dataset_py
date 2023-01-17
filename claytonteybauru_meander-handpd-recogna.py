from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns # plotting
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# Meander_HandPD.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df = pd.read_csv('/kaggle/input/Meander_HandPD.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'Meander_HandPD.csv'
nRow, nCol = df.shape
print(f'Existem {nRow} linhas e {nCol} colunas')
plotPerColumnDistribution(df, 4, 2)
plotCorrelationMatrix(df, 6)
plotScatterMatrix(df, 20, 10)
df.head(5)
# Como já havia estruturado o dataset anteriormente, não existem dados faltantes no dataset, mas ainda tenho dados
# categóricos que terão que ser manipulados.

df.info()
# Verificando a quantidade de amostras por sexo, sendo 60% do sexo masculino.

df['GENDER'].value_counts(), (df['GENDER'].value_counts()*100)/len(df)
# Quantidade e idade dos indivíduos por sexo, indivíduos femininos apresentam idades inferiores aos indivíduos do sexo masculino.

agebygender = df.groupby(by=df['AGE']).GENDER.value_counts()
agebygender
# O Boxplot abaixo apresenta os "outliers" e uma melhor visualização do comportamento das amostras com relação ao sexo e idade.

#%matplotlib notebook
df.boxplot(column='AGE', by='GENDER')
# Alterando as features Sex de valores categóricos para discretos.

def change_feature_sex(value):
    if value == 'F':
        return 1
    else:
        return 0

df['GENDER'] = df['GENDER'].map(change_feature_sex)
df.head()
# Verificando a quantidade de indivíduos, destros com 93% e canhotos com aproximadamente 7%

df['RIGH/LEFT-HANDED'].value_counts(), (df['RIGH/LEFT-HANDED'].value_counts()*100)/len(df)
# Alterando as features destro ou conhoto de valores categórico para discretos.

def change_feature_hand(value):
    if value == 'L':
        return 1
    else:
        return 0

df['RIGH/LEFT-HANDED'] = df['RIGH/LEFT-HANDED'].map(change_feature_hand)
df.head()
# Por fim, removemos a coluna de nome da imagem que não será utilizada para análise.

df = df.drop(columns=['IMAGE_NAME'])
df.head()
# Verificando a correlação entre as features

df.corr()
df.head(3)
# Iniciando os vetores para o modelo

X = df.iloc[:,3:]
y = df.iloc[:,2]
y
# importanto as bibliotecas necessárias para nosso treinamento e classificação

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
# Iniciando o model

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Iniciando o classificador Random Forest

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)
print(classification_report(y_test, pred))
# Matriz de confusão recebe como parâmetro o conjunto de teste(classes) e os resultados para efetuar as comparações
# entre o valor real e o valor predito pelo algoritmo.

print(pd.crosstab(y_test, pred, rownames=['Real'], colnames=['       Predito'], margins=True))
cm = confusion_matrix(y_test, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Paciente', 'Controle']); ax.yaxis.set_ticklabels(['Controle', 'Paciente']);
# Utilizando a biblioteca shap para visualização e melhor entendimento dos resultados

import shap

shap.initjs()
# Criando dois objetos para análise

view = shap.TreeExplainer(modelo)
shap_values = view.shap_values(X_train)
# Temos como saída dois vetores, valores que representam as duas classes do dataset.

shap_values[1].shape
# Analisando o comportamentos das feaures

shap.force_plot(view.expected_value[1], shap_values[1], X_train)
# As features mais importante e a probabilidade de cada uma pertencer a uma determinada classe, o gráfico mostra claramente que dentre
# todas as features, o indivíduo ser destro ou canhoto não influencia na análise e nos resultados.

shap.summary_plot(shap_values[1], X_train)
# Analisando a influência de uma determinada feature na previsão.

shap.dependence_plot('GENDER', shap_values[1], X_train, interaction_index=None)
# Analisando a influência de uma determinada feature na previsão.

shap.dependence_plot('GENDER', shap_values[1], X_train, interaction_index='AGE')
# Criando o classificador

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

