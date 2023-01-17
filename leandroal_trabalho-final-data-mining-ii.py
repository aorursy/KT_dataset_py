import pandas as pd

pd.options.display.max_columns = 200

pd.options.display.max_rows = 100

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

from sklearn.utils import resample

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import  warnings

warnings.filterwarnings("ignore")
file = '/kaggle/input/hmeq-data/hmeq.csv'

df = pd.read_csv(file)

# Visualização de informações gerais e missings

df.info()



# Visualização da distribuição de classes

print('\nDistribuição da Categoria BAD:\n', df['BAD'].value_counts(normalize=True))



# Visualização das 5 primeiras linhas do dataset

df.head(5)
fig, ax = plt.subplots(2,5, figsize=(30,15))

row_1 = ['LOAN','MORTDUE','VALUE','YOJ','DEROG']

row_2 = ['DELINQ','CLAGE','NINQ','CLNO','DEBTINC']

row_3 = []

for i, col in enumerate(row_1):

    sns.boxplot(x=df['BAD'], y=df[col], ax=ax[0,i])

for i, col in enumerate(row_2):

    sns.boxplot(x=df['BAD'], y=df[col], ax=ax[1,i])
fig, ax = plt.subplots(figsize=(8,7))

ax = sns.barplot(x="REASON", y="BAD", hue='BAD', data=df, estimator=lambda x: len(x))

ax.annotate(df['REASON'][(df['REASON'] == 'HomeImp') & (df['BAD'] == 0)].shape[0], xy=(0.11, 0.42), xycoords="axes fraction")

ax.annotate(df['REASON'][(df['REASON'] == 'HomeImp') & (df['BAD'] == 1)].shape[0], xy=(0.32, 0.12), xycoords="axes fraction")

ax.annotate(df['REASON'][(df['REASON'] == 'DebtCon') & (df['BAD'] == 0)].shape[0], xy=(0.61, 0.97), xycoords="axes fraction")

ax.annotate(df['REASON'][(df['REASON'] == 'DebtCon') & (df['BAD'] == 1)].shape[0], xy=(0.82, 0.23), xycoords="axes fraction")

plt.ylabel('Count')

plt.tick_params(

    axis='y',          

    left=False,

    labelleft=False)
df = df.dropna(thresh=df.shape[1]/2, axis=0)

df.info()
# =============================================================================

# Tratando colunas categóricas

# =============================================================================

for col in df.select_dtypes(include='object').columns:

    if df[col].isna().sum() > 0:

         df[col].fillna(df[col].mode()[0], inplace=True)   
# =============================================================================

# Tratando colunas numéricas

# =============================================================================

for col in df.select_dtypes(exclude='object').columns:

    if df[col].isna().sum() > 0:

        df[col].fillna(-1, inplace=True)      
# =============================================================================

# Resultado

# =============================================================================

df.info()
def showBalance(df, col):

    for c in col:

        print('Distribuição da Coluna: ', c,'\n',df[c].value_counts(normalize=True),'\n')

    else:

       pass

        

showBalance(df, col=['REASON','JOB','BAD'])
# =============================================================================

# Função para realizar balanceamento

# =============================================================================

def balance(df, col):

    df_maior = df[df[col] == df[col].mode()[0]]

    df_menor = df[df[col] != df[col].mode()[0]]

 

    # Upsample da menor classe

    df_menor_upsampled = resample(df_menor, 

                                  replace=True,     

                                  n_samples=df_maior.shape[0],

                                  random_state=42) 

 

    # Combinar as classe predominante com a menor classe aumentada

    df_upsample = pd.concat([df_maior, df_menor_upsampled])



    # Display new class counts

    print('Contagem de registros')

    print(df_upsample['BAD'].value_counts())

    print('\nDistribuição dos registros')

    print(df_upsample['BAD'].value_counts(normalize=True))



    return df_upsample

    

df_upsample = balance(df, 'BAD')

print('\n')

showBalance(df_upsample, col=['REASON','JOB','BAD'])
df_upsample.hist(figsize=(20,20))
dummies = ['REASON', 'JOB']

df_upsample = pd.get_dummies(df_upsample, columns=dummies, drop_first=True, dtype='int64')



cols = df_upsample.columns.tolist()

cols = cols[1:] + cols[:1]

df_upsample = df_upsample[cols]

del cols
# =============================================================================

# Observação sobre o dataset rebalanceado

# =============================================================================

df_upsample.info()

df_upsample.describe().T
# =============================================================================

# Divisão Treino e Teste

# =============================================================================

train, test = train_test_split(df_upsample, test_size=0.1, random_state=0)

# =============================================================================

# X e Y dos dados de treino

# =============================================================================

X = train.iloc[:,:-1].values

y = train.iloc[:,-1].values

# =============================================================================

# Divisão Treino e Validação

# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# =============================================================================

# Random Forest Classifier

# =============================================================================

''' Scikit-Learn Definition

A random forest is a meta estimator that fits a number of decision tree classifiers 

on various sub-samples of the dataset and uses averaging to improve the predictive 

accuracy and control over-fitting. The sub-sample size is always the same as the 

original input sample size but the samples are drawn with replacement if bootstrap=True (default).

'''

rf = RandomForestClassifier(n_estimators=700, criterion='entropy', random_state=42)



# =============================================================================

# Extra Trees Classifier

# =============================================================================

''' Scikit-Learn Definition

This class implements a meta estimator that fits a number of randomized 

decision trees (a.k.a. extra-trees) on various sub-samples of the dataset 

and uses averaging to improve the predictive accuracy and control over-fitting.

'''

extc = ExtraTreesClassifier(n_estimators=700,

                            criterion='entropy',

                            min_samples_split=5,

                            max_depth=50,

                            min_samples_leaf=5,

                            random_state=42) 
# =============================================================================

# Random Forest

# =============================================================================

rf.fit(X_train, y_train)

y_rfpred = rf.predict(X_test)



# =============================================================================

# Extra Trees

# =============================================================================

extc.fit(X_train, y_train)

y_extcpred = extc.predict(X_test)
# =============================================================================

# Dados de Validação

# =============================================================================

print('Validação Random Forest....Acurácia de: ', "{0:.1f}".format(accuracy_score(y_test, y_rfpred)*100),'%')

print('Validação Extra Trees......Acurácia de: ', "{0:.1f}".format(accuracy_score(y_test, y_extcpred)*100),'%')



# =============================================================================

# Dados de Teste

# =============================================================================

y2_rf = rf.predict(test.iloc[:,:-1].values)

y2_extc = extc.predict(test.iloc[:,:-1].values)

print('\nTeste Random Forest....Acurácia de: ', "{0:.1f}".format(accuracy_score(test.iloc[:,-1].values, y2_rf)*100),'%')

print('Teste Extra Trees......Acurácia de: ', "{0:.1f}".format(accuracy_score(test.iloc[:,-1].values, y2_extc)*100),'%')
scores = cross_val_score(rf, X_train, y_train, n_jobs=-1, cv=5)

print('Scores: ', [round(x,2) for x in scores])

print('Score médio: ', round(scores.mean(),2))
scores = cross_val_score(extc, X_train, y_train, n_jobs=-1, cv=5)

print('Scores: ', [round(x,2) for x in scores])

print('Score médio: ', round(scores.mean(),2))
fig, ax = plt.subplots(1,2, figsize=(16,10), sharex=True)

rf_feat_importances = pd.Series(rf.feature_importances_, index=df_upsample.iloc[:,:-1].columns)

ex_feat_importances = pd.Series(extc.feature_importances_, index=df_upsample.iloc[:,:-1].columns)



rf_feat_importances.plot(kind='barh', ax=ax[0], title='Random Forest')

ex_feat_importances.plot(kind='barh', ax=ax[1], title='Extra Trees')



plt.xlim((0.0, 0.35))

plt.show()
cm = confusion_matrix(y_test, y_rfpred, labels=[1, 0])

'''

default:

[[TN, FP,

  FN, TP]]

  

labels = [1,0]:

[[TP, FP,

  FN, TN]]

'''



fig, ax = plt.subplots(figsize=(7,6))

sns.heatmap(cm, annot=True, ax=ax, fmt='.0f'); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('Real labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['1', '0'])

ax.yaxis.set_ticklabels(['1', '0']);

plt.show()



sens = cm[0,0] / (cm[0,0] + cm[1,0])

esp = cm[1,1] / (cm[0,1] + cm[1,1])



print('Sensibilidade: ', round(sens,2))

print('Especificidade: ', round(esp,2))