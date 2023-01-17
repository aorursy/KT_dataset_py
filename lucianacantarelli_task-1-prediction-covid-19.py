#importando bibliotecas

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid")

%matplotlib inline
#importando os dados
import pandas as pd

df = pd.read_excel("../input/covid19/dataset.xlsx")

df_raw = df
df.shape
#Listing 5 first rows
df.head()
#Distribuition os Positive case compared to Negative case. Note that 11% of cases are positive. Unbalanced dataset.
df.groupby("SARS-Cov-2 exam result").size()

# Drop columns with insuficient data.
Colunas=['Serum Glucose',
'Mycoplasma pneumoniae',
'Alanine transaminase',
'Aspartate transaminase',
'Gamma-glutamyltransferase ',
'Total Bilirubin',
'Direct Bilirubin',
'Indirect Bilirubin',
'Alkaline phosphatase',
'Ionized calcium ',
'Strepto A',
'Magnesium',
'pCO2 (venous blood gas analysis)',
'Hb saturation (venous blood gas analysis)',
'Base excess (venous blood gas analysis)',
'pO2 (venous blood gas analysis)',
'Fio2 (venous blood gas analysis)',
'Total CO2 (venous blood gas analysis)',
'pH (venous blood gas analysis)',
'HCO3 (venous blood gas analysis)',
'Rods #',
'Segmented',
'Promyelocytes',
'Metamyelocytes',
'Myelocytes',
'Myeloblasts',
'Urine - Esterase',
'Urine - Aspect',
'Urine - pH',
'Urine - Hemoglobin',
'Urine - Bile pigments',
'Urine - Ketone Bodies',
'Urine - Nitrite',
'Urine - Density',
'Urine - Urobilinogen',
'Urine - Protein',
'Urine - Sugar',
'Urine - Leukocytes',
'Urine - Crystals',
'Urine - Red blood cells',
'Urine - Hyaline cylinders',
'Urine - Granular cylinders',
'Urine - Yeasts',
'Urine - Color',
'Partial thromboplastin time (PTT) ',
'Relationship (Patient/Normal)',
'International normalized ratio (INR)',
'Lactic Dehydrogenase',
'Prothrombin time (PT), Activity',
'Vitamin B12',
'Creatine phosphokinase (CPK) ',
'Ferritin',
'Arterial Lactic Acid',
'Lipase dosage',
'D-Dimer',
'Albumin',
'Hb saturation (arterial blood gases)',
'pCO2 (arterial blood gas analysis)',
'Base excess (arterial blood gas analysis)',
'pH (arterial blood gas analysis)',
'Total CO2 (arterial blood gas analysis)',
'HCO3 (arterial blood gas analysis)',
'pO2 (arterial blood gas analysis)',
'Arteiral Fio2',
'Phosphor',
'ctO2 (arterial blood gas analysis)',
'Influenza B, rapid test',
'Influenza A, rapid test'
]

df.drop(Colunas , axis = 1, inplace = True) # 'axis' = 0 (row) | axis = 1 (column)
# Drop columns with unnecessary data.
Colunas = ['Patient ID',"Patient addmited to regular ward (1=yes, 0=no)",
           "Patient addmited to semi-intensive unit (1=yes, 0=no)",
           "Patient addmited to intensive care unit (1=yes, 0=no)",
           'Respiratory Syncytial Virus',
            'Influenza A',
            'Influenza B',
            'Parainfluenza 1',
            'CoronavirusNL63',
            'Rhinovirus/Enterovirus',
            'Coronavirus HKU1',
            'Parainfluenza 3',
            'Chlamydophila pneumoniae',
            'Adenovirus',
            'Parainfluenza 4',
            'Coronavirus229E',
            'CoronavirusOC43',
            'Inf A H1N1 2009',
            'Bordetella pertussis',
            'Metapneumovirus',
            'Parainfluenza 2',
]
df.drop(Colunas , axis = 1, inplace = True) # 'axis' = 0 (row) | axis = 1 (column)
df.shape
#Rename coluns to simplify coding

df.rename(columns={'Patient age quantile': 'age'}, inplace=True)
df.rename(columns={'SARS-Cov-2 exam result': 'target'}, inplace=True)
df.rename(columns={'Mean platelet volume ': 'Mean_platelet_volume'}, inplace=True)
df.rename(columns={'Red blood Cells': 'Red_blood_Cells'}, inplace=True)

#Resulting columns
df.columns
df.isnull().sum()
# removing NaN from dataset
df_NotNull = df.dropna()
df_NotNull.shape
mask = {'positive': 1, 
        'negative': 0,
       }

df_NotNull = df_NotNull.replace(mask)
# number of each paciente per COVID-19 Result
df_NotNull.groupby("target").size()
# Correlação de Pearson (checando possibilidades de atributos colineares)
df_NotNull.corr(method = 'pearson')
correlations = df_NotNull.corr()
k = 22  #number of columns
cols = correlations.nlargest(k, "target")["target"].index
cm = np.corrcoef(df_NotNull[cols].values.T)
sns.set(font_scale = 1.25)
fig, ax = plt.subplots(figsize = (12, 6))
ax = sns.heatmap(cm, vmin = -1, vmax = 1, cmap = "Reds", cbar = True, annot = True, square = False, 
                 fmt = ".3f", annot_kws = {"size": 12}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()
# Drop columns with colinear data.
Colunas = ['Hemoglobin','Red_blood_Cells']
df_NotNull.drop(Colunas , axis = 1, inplace = True) # 'axis' = 0 (row) | axis = 1 (column)
sns.distplot(df_NotNull.age, fit = stats.norm)
sns.regplot(x = 'age', y = 'target', data = df_NotNull, x_ci = "sd", logistic = True, lowess = False, truncate = True, color = "r", marker = '.')
sns.regplot(x = 'age', y = 'target', data = df_NotNull, x_ci = "sd", logistic = False, lowess = True, truncate = True, color = "g", marker = '.')
sns.distplot(df_NotNull.Hematocrit, fit = stats.norm)
sns.regplot(x = 'Hematocrit', y = 'target', data = df_NotNull, x_ci = "sd", logistic = True, lowess = False, truncate = True, color = "r", marker = '.')
sns.regplot(x = 'Hematocrit', y = 'target', data = df_NotNull, x_ci = "sd", logistic = False, lowess = True, truncate = True, color = "g", marker = '.')
sns.distplot(df_NotNull.Platelets, fit = stats.norm)
sns.regplot(x = 'Platelets', y = 'target', data = df_NotNull, x_ci = "sd", logistic = True, lowess = False, truncate = True, color = "r", marker = '.')
sns.regplot(x = 'Platelets', y = 'target', data = df_NotNull, x_ci = "sd", logistic = False, lowess = True, truncate = True, color = "g", marker = '.')
sns.distplot(df_NotNull.Mean_platelet_volume, fit = stats.norm)
sns.regplot(x = 'Mean_platelet_volume', y = 'target', data = df_NotNull, x_ci = "sd", logistic = True, lowess = False, truncate = True, color = "r", marker = '.')
sns.regplot(x = 'Mean_platelet_volume', y = 'target', data = df_NotNull, x_ci = "sd", logistic = False, lowess = True, truncate = True, color = "g", marker = '.')
sns.distplot(df_NotNull.Lymphocytes, fit = stats.norm)
sns.regplot(x = 'Lymphocytes', y = 'target', data = df_NotNull, x_ci = "sd", logistic = True, lowess = False, truncate = True, color = "r", marker = '.')
sns.regplot(x = 'Lymphocytes', y = 'target', data = df_NotNull, x_ci = "sd", logistic = False, lowess = True, truncate = True, color = "g", marker = '.')
cols = list(df_NotNull.columns.values)
print(cols)
# Move column target to the end of the dataset

df_NotNull = df_NotNull [['age', 'Hematocrit', 'Platelets', 'Mean_platelet_volume', 'Lymphocytes', 'Mean corpuscular hemoglobin concentration\xa0(MCHC)', 'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes', 'Red blood cell distribution width (RDW)', 'Neutrophils', 'Urea', 'Proteina C reativa mg/dL', 'Creatinine', 'Potassium', 'Sodium','target']]
cols = list(df_NotNull.columns.values)
print(cols)
df_NotNull.shape
# MACHINE LEARNING - SELEÇÃO DE MODELO - auto
# Import dos módulos
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

dados =  df_NotNull
array = dados.values

# Separando o array em componentes de input e output
X = [array[:, 0:19]]
strX =  ["array[:, 0:19]"]
Y = array[:, 19]

# Definindo os valores para o número de folds
num_folds = 10
seed = 7
a = 0

for i in X: 
    # Preparando a lista de modelos
    modelos = []
    modelos.append(('LR', LogisticRegression()))
    modelos.append(('LDA', LinearDiscriminantAnalysis()))
    modelos.append(('NB', GaussianNB()))
    modelos.append(('KNN', KNeighborsClassifier()))
    modelos.append(('CART', DecisionTreeClassifier()))
    modelos.append(('MLPClassifier', MLPClassifier()))
    modelos.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
    
    

    # Avaliando cada modelo em um loop
    resultados = []
    nomes = []

    for nome, modelo in modelos:
        kfold = KFold(n_splits = num_folds, shuffle = True, random_state = seed)
        print(nome)
        predictions = cross_val_predict(modelo, i, Y, cv = kfold)
        print(classification_report(Y, predictions, digits = 4))
        print("fbeta :",fbeta_score(Y, predictions, average='macro', beta=3))
        print()
    
    a = a + 1
# Import dos módulos
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

dados = df_NotNull
array = dados.values

# Separando o array em componentes de input e output
X = array[:, 0:19]
Y = array[:, 19]

# Definindo os valores para o número de folds
num_folds = 10
seed = 7

# Separando os dados em folds
kfold = KFold(num_folds, shuffle = True, random_state = seed)

# Import dos módulos
from sklearn.metrics import classification_report

# Criando o modelo
model_LR = LogisticRegression()
model_LDA = LinearDiscriminantAnalysis()
model_NB = GaussianNB()
model_KNN = KNeighborsClassifier()
model_CART = DecisionTreeClassifier()
model_MLP =  MLPClassifier()
model_Boosting =  GradientBoostingClassifier()

# Cross Validation
resultado_LR = cross_val_score(model_LR, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_LDA = cross_val_score(model_LDA, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_NB = cross_val_score(model_NB, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_KNN = cross_val_score(model_KNN, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_CART = cross_val_score(model_CART, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_MLP = cross_val_score(model_MLP, X, Y, cv = kfold, scoring = 'roc_auc')
resultado_Boosting = cross_val_score(model_Boosting, X, Y, cv = kfold, scoring = 'roc_auc')


# Print do resultado

print("AUC LR: %.3f" % (resultado_LR.mean() * 100))
print("AUC LDA: %.3f" % (resultado_LDA.mean() * 100))
print("AUC NB: %.3f" % (resultado_NB.mean() * 100))
print("AUC KNN: %.3f" % (resultado_KNN.mean() * 100))
print("AUC CART: %.3f" % (resultado_CART.mean() * 100))
print("AUC MLP: %.3f" % (resultado_MLP.mean() * 100))
print("AUC Boosting: %.3f" % (resultado_Boosting.mean() * 100))

import matplotlib.patches as patches
from sklearn.metrics import roc_curve,auc
from scipy import interp

# plot arrows
fig1 = plt.figure(figsize = [12, 12])
ax1 = fig1.add_subplot(111, aspect = 'equal')
ax1.add_patch(
    patches.Arrow(0.45, 0.5, -0.25, 0.25, width = 0.3, color = 'green', alpha = 0.5)
    )
ax1.add_patch(
    patches.Arrow(0.5, 0.45, 0.25, -0.25, width = 0.3, color = 'red', alpha = 0.5)
    )

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 1
for train,test in kfold.split(X,Y):
    prediction = model_NB.fit(pd.DataFrame(X).iloc[train],pd.DataFrame(Y).iloc[train]).predict_proba(pd.DataFrame(X).iloc[test])
    fpr, tpr, t = roc_curve(Y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw = 2, alpha = 0.3, label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i + 1

plt.plot([0, 1],[0, 1], linestyle = '--', lw = 2, color = 'black')
mean_tpr = np.mean(tprs, axis = 0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color = 'blue',
         label = r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw = 2, alpha = 1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Naive Bayles')
plt.legend(loc = "lower right")
plt.text(0.32, 0.7, 'More accurate area', fontsize = 12)
plt.text(0.63, 0.4, 'Less accurate area', fontsize = 12)
plt.show()
# Python script for confusion matrix creation.
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

dados = df_NotNull
array = dados.values

# Separando o array em componentes de input e output
X = array[:, 0:19]
Y = array[:, 19]

# Definindo os valores para o número de folds
num_folds = 10
seed = 7

# Criando o modelo
modelo_opt = GaussianNB(priors=None)
# modelo_opt.fit(X, Y)

# Fazendo as previsões e construindo a Confusion Matrix
# previsoes_opt = modelo_opt.predict(X)
# matrix_opt = confusion_matrix(Y, previsoes_opt)

kfold = KFold(n_splits = num_folds, shuffle = True, random_state = seed)
previsoes_opt = cross_val_predict(modelo_opt, X, Y, cv = kfold)
matrix_opt = confusion_matrix(Y, previsoes_opt)

# Imprimindo a Confusion Matrix
print('Confusion Matrix :')
print(matrix_opt)
print()
print('roc_auc Score :',roc_auc_score(Y, previsoes_opt))
print()
print('Report : ')
print(classification_report(Y, previsoes_opt))
