# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# visulaisation
from matplotlib.pyplot import xticks
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


data = pd.read_csv("../input/RELATORIO_VENDA_REDUZIDO_18_02_2020.csv",sep=';')
data.head(5)

# DATA INSPECTION

#checking duplicates
#sum(data.duplicated(subset ='CODIGO')) == 0
# No duplicate values

#data.shape

data.info


data.describe()
# iterating the columns 
#for col in data.columns: 
#    print(col)



# removing NAs

# iterating the columns 
# for col in data.columns: 
#    print(col)
#    data.info(col)


# data.isnull().sum
round(100*(data.isnull().sum()/len(data.index)), 2)

# we will drop the columns having more than 75% NA values.
data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>75)].columns, 1)



data['OCORRENCIA'].describe()
data.OCORRENCIA.unique()
sns.countplot(data['OCORRENCIA'])
xticks(rotation=90)

# data.LOGRADOURO.describe()

# Rest missing values are under 2% so we can drop these rows.
data.dropna(inplace = True)

round(100*(data.isnull().sum()/len(data.index)), 2)

#checking duplicates
sum(data.duplicated(subset = 'CODIGO')) == 0
# No duplicate values

data.to_csv('/kaggle/working/cleaned-beta-rel-18-02-2020.csv',sep=';')
cleand = pd.read_csv('/kaggle/working/cleaned-beta-rel-18-02-2020.csv',sep=';')

#for dirname, _, filenames in os.walk('/kaggle'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
# cleand[cleand['OCORRENCIA'] == "IMPUTADA"].sum()
# len(cleand['OCORRENCIA'].index))*100
cleand.info()
Converted = (sum(cleand['OCORRENCIA'] == 'IMPUTADA')/len(cleand['OCORRENCIA'].index))*100
Converted


# cleand['OCORRENCIA'] = cleand['OCORRENCIA'].apply({'IMPUTADA':0}.get)
def create_converted(ocorrencia) :
    
    if (ocorrencia == 'IMPUTADA'):
        return 1
    else:
        return 0
cleand['Converted'] = cleand['OCORRENCIA'].apply(create_converted)

sum(cleand['Converted'] == 1)

# DROP unique valued columns
for col in cleand.columns:
    if len(cleand[col].unique()) == 1:
        cleand.drop(col,inplace=True,axis=1)

cleand.to_csv('/kaggle/working/no-uniq-beta-rel-18-02-2020.csv',sep=';')
nouniq = pd.read_csv('/kaggle/working/no-uniq-beta-rel-18-02-2020.csv',sep=';')

nouniq.info()



total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "HISTORICO DE VENDA", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)

total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "TIPO AQUISICAO MOVEL", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)

total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "PRODUTO MOVEL", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "VENCIMENTO", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
#nouniq['ESTADO'] = nouniq['ESTADO'].replace(['CE,PA,AM,RR,PE,PB,RO,TO,AC,AP,AL,SE,PI'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['CE'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['PA'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['AM'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['RR'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['PE'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['PB'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['RO'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['TO'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['AC'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['AP'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['AL'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['SE'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['PI'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['RN'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['MA'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['DF'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['GO'],'OUTROS')
nouniq['ESTADO'] = nouniq['ESTADO'].replace(['MS'],'OUTROS')

total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "ESTADO", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
total = float(len(nouniq)) # one person per row 

ax = sns.countplot(x = "SITE", hue = "Converted", data = nouniq)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
nouniq = nouniq.drop(columns=['EVIDENCIA'])
nouniq.info()
Fnded = ((sum(nouniq['OCORRENCIA'] == 'FND ONLINE') + sum(nouniq['OCORRENCIA'] == 'FND'))/len(nouniq['OCORRENCIA'].index))*100


no_actionDF = nouniq
no_actionDF['OCORRENCIA'] = no_actionDF['OCORRENCIA'].replace(['FND ONLINE'],'NO REVERT')
no_actionDF['OCORRENCIA'] = no_actionDF['OCORRENCIA'].replace(['FND'],'NO REVERT')
no_actionDF['OCORRENCIA'] = no_actionDF['OCORRENCIA'].replace(['JA POSSUI O PLANO SOLICITADO'],'NO REVERT')
no_actionDF['OCORRENCIA'] = no_actionDF['OCORRENCIA'].replace(['SUSPEITA DE FRAUDE'],'NO REVERT')

non_input = nouniq.copy(deep=False)

non_input.drop(non_input[non_input.OCORRENCIA == 'IMPUTADA'].index,inplace=True)


def create_fnded(ocorrencia) :
    
    if ((ocorrencia == 'FND ONLINE') | 
        (ocorrencia == 'FND') | 
        (ocorrencia == 'FND CONTATO SEM SUCESSO') | 
        (ocorrencia == 'JA POSSUI O PLANO SOLICITADO') |
        (ocorrencia == 'SUSPEITA DE FRAUDE')):
        return 1
    else:
        return 0
    
non_input['Fnded'] = non_input['OCORRENCIA'].apply(create_fnded)

non_input.OCORRENCIA.unique()


non_input.drop(non_input[non_input.ESTADO == 'OUTROS'].index,inplace=True)


total = float(len(non_input)) # one person per row 

ax = sns.countplot(x = "ESTADO", hue = "Fnded", data = non_input)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
total = float(len(non_input)) # one person per row 

ax = sns.countplot(x = "VENCIMENTO", hue = "Fnded", data = non_input)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
xticks(rotation = 90)
dummy1 = pd.get_dummies(nouniq[['HISTORICO DE VENDA','ESTADO','TIPO AQUISICAO MOVEL','PRODUTO MOVEL','SITE']],drop_first = True)

num_data_imp = pd.concat([nouniq,dummy1],axis = 1)
num_data_imp = num_data_imp.drop(['HISTORICO DE VENDA','ESTADO','TIPO AQUISICAO MOVEL','PRODUTO MOVEL','SITE'], axis = 1)
num_data_imp = num_data_imp.drop(['VENCIMENTO','NOME/RAZAO SOCIAL','CPF/CNPJ','DATA CAD. VENDA','HORA CAD. VENDA','DATA ULT. ALT. VENDA','HORA ULT. ALT. VENDA','BAIRRO','RESPONSAVEL OCORRENCIA','DATA HORA OCORRENCIA','TELEFONE 1','CLIENTE EMAIL','LOGRADOURO','NUMERO','CEP','CIDADE','OCORRENCIA'],axis = 1)
num_data_imp = num_data_imp.drop(num_data_imp.columns[[0,1]],axis = 1)
# num_data_imp.head()

X = num_data_imp.drop(['CODIGO','Converted'],axis = 1)
X.head()

Y = num_data_imp['Converted']
Y.head()

num_data_imp.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Converted = (sum(num_data_imp['Converted'])/len(num_data_imp['Converted'].index))*100
Converted
import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 10)             # running RFE with 5 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col1 = col.drop('PRODUTO MOVEL_CA_ ANUAL VIVO CONTROLE 3 5GB 39 99',1)
col2 = col1.drop('ESTADO_OUTROS',1)
col2
X_train_sm = sm.add_constant(X_train[col2])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['CODIGO'] = y_train.index
y_train_pred_final.head(100)

y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.1 else 0)

# Let's see the head
y_train_pred_final.head(100)
from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col2].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col2].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
### From the curve above, 0.2 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.12 else 0)

y_train_pred_final.head()
y_train_pred_final['Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
X_test[['ESTADO_ES','ESTADO_MG','ESTADO_MT']] = scaler.fit_transform(X_test[['ESTADO_ES','ESTADO_MG','ESTADO_MT']])

X_test.head()
X_test = X_test[col2]
X_test.head()


X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test.head()
# Putting CustID to index
y_test_df['Customer ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Customer ID','Converted','Converted_prob'], axis=1)
# Let's see the head of y_pred_final
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.12 else 0)
y_pred_final['Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))

y_pred_final.head(100)

score = y_pred_final.sort_values(by=['Score'],ascending=False)
score.head()
max_score_customerIDs = score['Customer ID']
df1 = X_test[X_test.index == 595] 
df2 = X_test[X_test.index == 3772]
df3 = X_test[X_test.index == 6634]
df4 = X_test[X_test.index == 6647]
df5 = X_test[X_test.index == 1784]
frames = [df1,df2,df3,df4,df5]
best_score_sets = pd.concat(frames)
best_score_sets
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)