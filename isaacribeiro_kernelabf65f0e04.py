# Dependency Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', 400)
#pd.set_option('display.max_rows', 400)
# Data collection
## For the time's sake, 2018's Presidential Election Results have been extracted from https://especiais.gazetadopovo.com.br/eleicoes/2018/resultados/mapa-eleitoral-de-presidente-por-municipios-2turno/)
df_election = pd.read_excel('../input/brazilian-runoff-election-result-by-municipality/2018_BR_Resultados_PRES_Municipios_Ganhador.xlsx')

# Uppercase columns and State Initials
df_election.columns = map(str.upper, df_election.columns)
df_election['UF'] = df_election['UF'].str.upper()

# Remove useless columns
df_election = df_election.drop(columns=['CARGO','NUMEROCANDIDATOMAISVOTADO','FAIXA', 'COR'])
df_election.head()
# 2018's Runoff Results by City
# Only relevant for electores who have voted in Brazil
df_runoff = df_election[np.logical_and(df_election['TURNO'] == 2,df_election['NACIONALIDADE'] == 'Brasil')]

# Rename meaninfully the remaing columns
df_runoff = df_runoff.rename(columns={'NOME' : 'MUNICIPIO',
                                      'CANDIDATO' : 'VENCEDOR'})

# Extra cleanup - Remove new useless columns
## There is no equivalence between IBGE City Code and the code provided from dataset
df_runoff = df_runoff.drop(columns=['TURNO','NACIONALIDADE', 'TOTALVOTOS', 'TOTALVOTOSCANDIDATOMAISVOTADO', '% DO TOTAL DE VOTOS', 'CODIGO']).set_index(['UF','MUNICIPIO'])

df_runoff.head(10)
# 5570? If so, then all the Municipalities have been taken into account
len(df_runoff) == 5570
# MultiIndex find routine -- for further referente, uncomment case needed
# TODO remove before publishing
# sample_A = df_runoff[(df_runoff.index.get_level_values('UF') == 'AC') & (df_runoff.index.get_level_values('CIDADE') == 'BUJARI')]
# sample_A.head()
# 2010's Brazilian Census data
df_brazil = pd.read_excel('../input/brazilian-censuses/atlas2013_dadosbrutos_pt.xlsx', sheet_name = 'MUN 91-00-10')
# Explanatory Variables Description
df_features_desc = pd.read_excel('../input/brazilian-censuses/atlas2013_dadosbrutos_pt.xlsx', sheet_name = 'Siglas')
# Basic Data manipulation to avoid multiple execution of the same cell
df_brazil_2010 = df_brazil[df_brazil['ANO'] == 2010]
df_brazil_2010 = df_brazil_2010.rename(columns={'UF' : 'UF_CODE',
                                                'Município' : 'MUNICIPIO'})
df_brazil_2010['MUNICIPIO'] = df_brazil_2010['MUNICIPIO'].str.upper()
df_brazil_2010.head()
# Auxiliary Manipulation: 
# 2010's Census data provides an UF code according to the IBGE value
# This block creates an auxiliar dataframe with the respective map between both datasets
data = {'UF_CODE' : [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53],
        'UF' : ["RO", "AC", "AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA", "MG", "ES", "RJ", "SP", "PR", "SC", "RS", "MS", "MT", "GO", "DF"] }
df_uf_codes = pd.DataFrame(data).set_index('UF_CODE')
# Merge DF_UF_CODES into DF_BRAZIL_2010
df_brazil_2010 = pd.merge(df_uf_codes, df_brazil_2010, on='UF_CODE')

# Clean unnecessary columns
df_brazil_2010 = df_brazil_2010.drop(columns=['UF_CODE','ANO','Codmun6', 'Codmun7'])
# As we don't have an UUID to map Cities and Their most voted candidates, let's assume that ( UF + MUNICIPIO ) is enough to find 
# an unique entry
df_brazil_2010 = df_brazil_2010.set_index(['UF', 'MUNICIPIO'])
df_brazil_2010.head()
# Performance Boost: First part of Data Manipulation is ready
# Let's create a copy as a milestone for joining process
df_brazil_2010_cp = df_brazil_2010.copy()
df_brazil_merged = pd.merge(df_brazil_2010_cp, df_runoff, on=['UF','MUNICIPIO'])

df_brazil_merged.head()
# In 2010, Brazil had 5565 municipalities, 5 less than 2018.
# Besides that, as we are using a String as part of key, always there is the risk of missing or different data.
# After several treatment routines, we came up with 5521 aligned entries that will be used to train and evaluate our model.
# 5565 - 5521, Pareto principle: adjusting 44 entries (0,7%) of our population is not worth it.
print("Missing municipalities: %s" % (len(df_brazil_2010) - len(df_brazil_merged)))
### How to figure out which municipalities are missing
### Auxiliar routine --> Uncomment, case needed
## df_brazil_merged_with_error = df_brazil_merged.copy().drop(columns=['TOTAL_VOTOS','TOTAL_VOTOS_VENCEDOR','VENCEDOR'])
## df_brazil_missing = pd.concat([df_brazil_merged_with_error, df_brazil_2010_cp]).drop_duplicates(keep=False)
## df_brazil_missing
# Correlation Matrix - Multicolinearity Issues!!! yes, we have! --> Two many feature, hard to find an adequate visualization

corr = df_brazil_merged.corr()

fig, ax = plt.subplots(figsize=(15,15))
ax.set_title('Correlation Matrix before Cleanup')
ax.matshow(corr)
# Adjust of Population Data
# T_FEM_TOTAL: Proportion of Women to the total population
df_brazil_merged['T_FEM_TOTAL'] = df_brazil_merged['MULHERTOT'] / df_brazil_merged['pesotot']
df_brazil_merged['T_MASC_TOTAL'] = df_brazil_merged['HOMEMTOT'] / df_brazil_merged['pesotot']
df_brazil_merged['T_IDOSO'] = df_brazil_merged['PESO65'] / df_brazil_merged['pesotot']
# Clean up cell --> Taking the problem into account, let's manually remove some features that don't answer our question
df_brazil_cleanup = df_brazil_merged.copy()

# Since it is mandatory for every elector to have, at least, 16-years-old, let's remove the features related to other age groups
_nonRelevantColumns_A = df_brazil_merged.iloc[0:0,:].filter(regex='(T_ANALF|T_FUND|T_FUNDIN|T_MED|PIND|PP|T_DES|T_FREQ|T_ATRASO)')
_18yrsFilter = _nonRelevantColumns_A.iloc[0:0,:].filter(regex='18M')
_nonRelevantColumns_A = _nonRelevantColumns_A.drop(columns=_18yrsFilter)

_nonRelevantColumns_B = df_brazil_merged.iloc[0:0,:].filter(regex='(MULH|MULHER|HOMEM|HOMENS|RDPC|RDPC|T_M)[0-9]')
_populationalColumns = df_brazil_merged.iloc[0:0,:].filter(regex='(PESO|Peso|peso|POP|POPT|HOMEMTOT|MULHERTOT|IDHM|PEA|PIA|PMP|T_ATIV|T_FB|T_FL|CORTE|PREN|REN|R1040|R2040)')
df_brazil_cleanup = df_brazil_cleanup.drop(columns=_nonRelevantColumns_A)
df_brazil_cleanup = df_brazil_cleanup.drop(columns=_nonRelevantColumns_B)
df_brazil_cleanup = df_brazil_cleanup.drop(columns=_populationalColumns)
# Ok, we have a 99,3% dataset that maps the winner candidate by city correctly
# Then, let's start to prepare the data for the model
df_brazil = df_brazil_cleanup.copy()

# Creates the 'CLASS' column in accordance with the following:
## BOLSONARO --> 1
## HADDAD --> 0
df_brazil['CLASS'] = np.where(df_brazil['VENCEDOR'] == 'Bolsonaro', 1, 0)
df_brazil = df_brazil.drop(columns=['VENCEDOR'])

# How balanced is our dataset?
number_of_cities_haddad_won = len(df_brazil[df_brazil['CLASS'] == 0])
number_of_cities_bolsonaro_won = len(df_brazil[df_brazil['CLASS'] == 1])
# Income is a strong hypothesis to differentiate the election decision. However, we need to avoid multicolinearity issues. 
# Hence, let's correlate the main income-based vars and check how it would affect our model
# We still have a high number of features
# Let's recreate our correlation matrix

corr_cl = df_brazil_cleanup.corr()

fig, ax = plt.subplots(figsize=(15,15))
ax.set_title('Correlation Matrix after Cleanup')
ax.matshow(corr_cl)
# It looks much better now, let's try another visualization
import seaborn as sns
mask = np.zeros_like(corr_cl, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax.set_title('Correlation Matrix after Cleanup')

sns.heatmap(corr_cl, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
number_of_cities_bolsonaro_won - number_of_cities_haddad_won
# HADDAD has won in more municipalities than Bolsonaro, but our result set is almost balanced
balanceament_factor = number_of_cities_bolsonaro_won / (number_of_cities_bolsonaro_won + number_of_cities_haddad_won)
print(balanceament_factor)
# General Dataset Properties
df_brazil.describe()
# Currently, we have five education-related features in our dataset:
# T_ANALF18M, T_FUND18M, T_MED18M, T_SUPER25M
# How are they correlated with the 'CLASS'?

education_features = ['T_ANALF18M', 'T_FUND18M', 'T_MED18M', 'T_SUPER25M', 'CLASS']
df_brazil_education = df_brazil[education_features]

# Correlation Matrix
education_corr = df_brazil_education.corr()

mask = np.zeros_like(education_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax.set_title('Educational Features Correlation Matrix')

sns.heatmap(education_corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

df_education_BYCLASS = df_brazil_education.groupby(by='CLASS')
df_education_BYCLASS.describe()
# Currently, we have five education-related features in our dataset:
# RDPC, GINI
# How are they correlated with the 'CLASS'?

income_features = ['RDPC', 'GINI', 'CLASS']
df_brazil_income = df_brazil[income_features]

# Correlation Matrix
income_corr = df_brazil_income.corr()

mask = np.zeros_like(income_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax.set_title('Income Features Correlation Matrix')

sns.heatmap(income_corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

df_income_BYCLASS = df_brazil_income.groupby(by='CLASS')
df_income_BYCLASS.describe()
# Currently, we have five education-related features in our dataset:
# RDPC, GINI
# How are they correlated with the 'CLASS'?

women_features = ['T_FEM_TOTAL', 'CLASS']
df_brazil_women = df_brazil[women_features]

# Municipalities where there are more women. Keep in mind this number is not the same as 'women able to vote', but gives a good 
# idea about proportionality
df_brazil_women = df_brazil_women[df_brazil_women['T_FEM_TOTAL'] > 0.5]
df_women_BYCLASS = df_brazil_women.groupby(by='CLASS')
df_women_BYCLASS.describe()
# Currently, we have five education-related features in our dataset:
# RDPC, GINI
# How are they correlated with the 'CLASS'?

elderly_features = ['T_IDOSO', 'CLASS']
df_brazil_elderly = df_brazil[elderly_features]

# Municipalities where there are more women. Keep in mind this number is not the same as 'women able to vote', but gives a good 
# idea about proportionality
df_brazil_elderly = df_brazil_elderly[df_brazil_elderly['T_IDOSO'] > df_brazil['T_IDOSO'].mean()]
df_elderly_BYCLASS = df_brazil_elderly.groupby(by='CLASS')
df_elderly_BYCLASS.describe()
# Splits train and test data: 70/30
from sklearn.model_selection import train_test_split

features = np.array(df_brazil.columns[0:-1])

X = df_brazil[features]
y = df_brazil['CLASS']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
# Standardized Values (Mean = 0; STD = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# PCA --> variance retention of 75% 
from sklearn.decomposition import PCA

pca = PCA(.75)
pca.fit(X_train)

# Nice, 9 features retains more 75% of total variance
print('%s of variance explained by %s components' % (sum(pca.explained_variance_ratio_) * 100, pca.n_components_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logRegr = LogisticRegression()
logRegr.fit(X_train, Y_train)
Y_pred = logRegr.predict(X_test)
logRegr.score(X_test, Y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
# ROC - Receiver Operating Characteristic

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y_test, logRegr.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logRegr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
X = df_brazil[features]
y = df_brazil['CLASS']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
# Second technique to reduce the number of features 

# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# To compare with PCA, we will keep the same number of features = 9
rfe = RFE(logreg, 9)
rfe = rfe.fit(X_train, Y_train.values.ravel())
selected_features = X_train.columns[rfe.support_]
print(selected_features)
X_train = X_train[selected_features]
X_test = X_test[selected_features]
# Model Performance

import statsmodels.api as sm
logit_model = sm.Logit(Y_train, X_train.astype(float))
result = logit_model.fit()
print(result.summary2())
# Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print(logreg)
y_pred_rfe = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, y_pred_rfe)
print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred_rfe))
# ROC - Receiver Operating Characteristic

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# CART - Classification and Regression Tree

from sklearn import tree

classTree = tree.DecisionTreeClassifier(criterion='entropy')

classTree = classTree.fit(X_train, Y_train)

# vamos medir a proporção de acertos com uma função
def acuracy(classTree,X_test,Y_test):
    predict = classTree.predict(X_test)
    erro = 0.0
    for x in range(len(predict)):
        if predict[x] != Y_test[x]:
            erro += 1.
    acuracy = (1-(erro/len(predict)))
    return acuracy

acuracy(classTree,X_test,Y_test)
Y_pred_dt = classTree.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred_dt)
print(confusion_matrix)   
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_dt))
# ROC - Receiver Operating Characteristic

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y_test, classTree.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, classTree.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# Quick Glimpse into the Decision Tree Complexity- Requires GraphViz 2.38
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(classTree,
                              out_file=f,
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")
