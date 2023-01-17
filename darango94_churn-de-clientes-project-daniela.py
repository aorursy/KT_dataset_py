# Basic Libraries
import numpy as np 
import pandas as pd 

# warnings — Warning control
import warnings
warnings.filterwarnings('ignore')

# Html document analysis (web Scraping)
import requests
from bs4 import BeautifulSoup
import re

# convert to dates
import datetime
from datetime import datetime, timedelta

#Coding categorical labels in numbers
from sklearn.preprocessing import LabelEncoder

# Division dataset Train/test
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# collections
import collections
import os
print(os.listdir("../input"))

# Visaulization
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
%matplotlib inline

# Análisis VIF
from sklearn.linear_model import LinearRegression

#Continuous variable normalization
from scipy.stats import zscore

#Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

# Features Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV # Recursive feature elimination with cross validation 

# Classifier (machine learning algorithm) 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Desbalanced Class
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import KFold


# Evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict

# Parameter Tuning
from sklearn.model_selection import GridSearchCV

# Settings
pd.options.mode.chained_assignment = None # Stop warning when use inplace=True of fillna
clientes_dic = pd.read_csv('../input/cosecha-dic-ene/clientes_dic.csv')
consumo_dic = pd.read_csv('../input/cosecha-dic-ene/consumo_dic.csv')
productos_dic = pd.read_csv('../input/cosecha-dic-ene/productos_dic.csv')
financiacion_dic = pd.read_csv('../input/cosecha-dic-ene/financiacion_dic.csv')
clientes_ene = pd.read_csv('../input/cosecha-dic-ene/clientes_ene.csv')
consumo_ene = pd.read_csv('../input/cosecha-dic-ene/consumo_ene.csv')
productos_ene = pd.read_csv('../input/cosecha-dic-ene/productos_ene.csv')
financiacion_ene= pd.read_csv('../input/cosecha-dic-ene/financiacion_ene.csv')
df_dic = pd.read_csv('../input/cosecha-dic-y-ene/df_dic.csv')
df_ene = pd.read_csv('../input/cosecha-dic-y-ene/df_ene.csv')
df1_dic= clientes_dic.merge(consumo_dic, how='outer', indicator='union')
df2_dic = df1_dic.merge(productos_dic,how='outer', indicator='exists')
df3_dic= df2_dic.merge(financiacion_dic,how='outer', indicator='exists2')
df3_dic=df3_dic.drop(['union', 'exists'], axis=1)
df1_ene = clientes_ene.merge(consumo_ene, how='outer', indicator='union')
df2_ene= df1_ene.merge(productos_ene,how='outer', indicator='exists')
df3_ene = df2_ene.merge(financiacion_ene,how='outer', indicator='exists2')
df3_ene=df3_ene.drop(['union', 'exists'], axis=1)
df_dic_ene = pd.merge(df3_dic, df3_ene, how='outer', on='id', suffixes=('_dic', '_ene'), indicator=True)
df_dic_ene =df_dic_ene .drop(['exists2_ene'], axis=1)
filtro = df_dic_ene['_merge'].isin(["both","left_only"])
df_dic_ene=df_dic_ene[filtro]
df_dic_ene['_merge']= (df_dic_ene['_merge'] == 'left_only') +0
df_dic_ene.head()
print(df_dic_ene.shape)
print(pd.value_counts(df_dic_ene['_merge'], sort = True))
# Splitting
#y = df_dic_ene._merge
#x = df_dic_ene.drop('_merge',axis=1)
 #X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=86,
                                                    #stratify = y)
data_dic = df_dic_ene.loc[:, ~df_dic_ene.columns.str.contains("ene$")]

data_ene = df_dic_ene.loc[:, ~df_dic_ene.columns.str.contains("dic$")]
df_dic.head()
df_ene.head()
df_dic.info()
print(df_dic.shape)
df_dic.describe()
df_dic.isnull().sum()
print(df_ene.shape)
df_ene.describe()
df_ene.isnull().sum()
# Continuous Data Plot
def cont_plot(df, feature_name, target_name, palettemap, hue_order, feature_scale): 
    df['Counts'] = "" # A trick to skip using an axis (either x or y) on splitting violinplot
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    sns.distplot(df[feature_name], ax=axis0);
    sns.violinplot(x=feature_name, y="Counts", hue=target_name, hue_order=hue_order, data=df,
                   palette=palettemap, split=True, orient='h', ax=axis1)
    axis1.set_xticks(feature_scale)
    plt.show()
    # WARNING: This will leave Counts column in dataset if you continues to use this dataset

# Categorical/Ordinal Data Plot
def cat_plot(df, feature_name, target_name, palettemap): 
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
    sns.countplot(x=feature_name, hue=target_name, data=df,
                  palette=palettemap,ax=axis1)
    plt.show()

    
survival_palette = {0: "black", 1: "orange"} # Color map for visualization
cont_plot(df_dic, 'edad', '_merge', survival_palette, [1, 0], range(0,85,10))
#ages_labels=pd.cut(x=df_dic['edad_dic'], bins=[18,19,30,45,50,60,85],
                         #labels=["18-19", "19-30", "30-45","45-50","50-60","60+"]).copy()
#cat_plot(df_dic, 'ages_labels', '_merge', survival_palette)
cont_plot(df_dic, 'facturacion', '_merge', survival_palette, [1, 0], range(0,400,15))
cont_plot(df_dic, 'num_lineas', '_merge', survival_palette, [1, 0], range(1,6,1))
df_dic.incidencia.fillna("NO",inplace=True)
cat_plot(df_dic, 'incidencia', '_merge', survival_palette)
plot = pd.crosstab(index=df_dic['_merge'],
            columns=df_dic['incidencia']
                  ).apply(lambda r: r/r.sum() *100,
                          axis=0).plot(kind='bar', stacked=True)
num_dt_nonan = df_dic[['num_dt','_merge']].copy().dropna(axis=0) # Copy dataframe so method won't leave Counts column in train_set
cont_plot(num_dt_nonan , 'num_dt', '_merge', survival_palette, [1, 0], range(1,5,1))
cat_plot(df_dic, 'num_dt', '_merge', survival_palette)
cat_plot(df_dic, 'TV','_merge', survival_palette)
cat_plot(df_dic, 'conexion','_merge', survival_palette)
cat_plot(df_dic, 'vel_conexion','_merge', survival_palette)
cont_plot(df_dic, 'num_llamad_ent', '_merge', survival_palette, [1, 0], range(0,300,63))
cont_plot(df_dic, 'num_llamad_sal', '_merge', survival_palette, [1, 0], range(0,100,25))
cont_plot(df_dic, 'mb_datos', '_merge', survival_palette, [1, 0], range(0,25000,6000))
cont_plot(df_dic, 'seg_llamad_ent', '_merge', survival_palette, [1, 0], range(0,20000,5000))
cont_plot(df_dic, 'seg_llamad_sal', '_merge', survival_palette, [1, 0], range(0,20000,5055))
df_dic.financiacion.fillna("NO",inplace=True)
cat_plot(df_dic, 'financiacion', '_merge', survival_palette)
plot = pd.crosstab(index=df_dic['_merge'],
            columns=df_dic['financiacion']
                  ).apply(lambda r: r/r.sum() *100,
                          axis=0).plot(kind='bar', stacked=True)
df_dic.descuentos.fillna("NO",inplace=True)
cat_plot(df_dic, 'descuentos', '_merge', survival_palette)
plot = pd.crosstab(index=df_dic['_merge'],
            columns=df_dic['descuentos']
                  ).apply(lambda r: r/r.sum() *100,
                          axis=0).plot(kind='bar', stacked=True)
imp_financ_nonan = df_dic[['imp_financ','_merge']].copy().dropna(axis=0) #Copy dataframe so method won't leave Counts column in train_set
cont_plot(imp_financ_nonan, 'imp_financ', '_merge', survival_palette, [1, 0], range(5,45,8))
imp_financ_nonan = df_dic[['imp_financ','_merge']].copy()
imp_financ_nonan ['Counts'] = "" 
fig, axis = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='imp_financ', y="Counts", hue='_merge', hue_order=[1, 0], data=imp_financ_nonan,
               palette=survival_palette, split=True, orient='h', ax=axis)
axis.set_xticks(range(5,45,8))
axis.set_xlim(-15,100)
plt.show()
df_dic.describe()
df_ene.describe()
# Completando Missings Cosecha Enero
df_ene.descuentos.fillna("NO",inplace=True)
df_ene.incidencia.fillna("NO",inplace=True)
df_ene.financiacion.fillna("NO",inplace=True)
# Imputaciones Cosecha Enero
#df_ene["edad"].fillna(df_ene["edad"].median(), inplace=True)
#df_ene["facturacion"].fillna(df_ene["facturacion"].median(), inplace=True)
#df_ene["num_lineas"].fillna(df_ene["num_lineas"].median(), inplace=True)
#df_ene["num_llamad_ent"].fillna(df_ene["num_llamad_ent"].median(), inplace=True)
#df_ene["num_llamad_sal"].fillna(df_ene["num_llamad_sal"].median(), inplace=True)
#df_ene["mb_datos"].fillna(df_ene["mb_datos"].median(), inplace=True)
#df_ene["seg_llamad_ent"].fillna(df_ene["seg_llamad_ent"].median(), inplace=True)
#df_ene["seg_llamad_sal"].fillna(df_ene["seg_llamad_sal"].median(), inplace=True)

    
#df_ene["antiguedad"].fillna(df_ene["antiguedad"].value_counts().index[0], inplace=True)
#df_ene["provincia"].fillna(df_ene["provincia"].value_counts().index[0], inplace=True)
#df_ene["conexion"].fillna(df_ene["conexion"].value_counts().index[0], inplace=True)
#df_ene["vel_conexion"].fillna(df_ene["vel_conexion"].value_counts().index[0], inplace=True)
#df_ene["TV"].fillna(df_ene["TV"].value_counts().index[0], inplace=True)
figu, axis1 = plt.subplots(1,1,figsize=(10,5))
sns.boxplot(data = df_dic, x = 'financiacion', y = df_dic['imp_financ'].dropna(), 
                  showfliers = True,palette='bright',ax=axis1)

plt.title('Distribucion de Pago Mensual en función de Terminales Financiados')
plt.xlabel('Número de Líneas')
plt.ylabel('Pago TérminalesFinanciados ')

plt.ticklabel_format(style='plain', axis='y')
df_dic.groupby('financiacion')['imp_financ'].median()
# Proporción de Missings por variable
prop_missings_dic = df_dic.apply(lambda x:x.isnull().mean()).copy()
prop_missings_dic
prop_missings_ene = df_ene.apply(lambda x:x.isnull().mean()).copy()
prop_missings_ene
df_dic.drop(["imp_financ"], axis=1, inplace=True)
df_ene.drop(["imp_financ"], axis=1, inplace=True)
def make_soup(url: str) -> BeautifulSoup:
    res = requests.get(url)
    res.raise_for_status()
    return BeautifulSoup(res.text, 'html.parser')

def extract_purchases(soup: BeautifulSoup) -> list:
    table = soup.find('th', text=re.compile('Provincia')).find_parent('table')
    purchases = []
    for row in table.find_all('tr')[1:]:
        Cca_cell,pro_cell= row.find_all('td')[::-2]
        p = {
            'CCAA': pro_cell.text.strip(),
            'Provincia': Cca_cell.text.strip(),
            #'CPRO' : cpro_cell.text.strip(),
        }
        purchases.append(p)
    return purchases

if __name__ == '__main__':
    url = 'https://www.ine.es/daco/daco42/codmun/cod_ccaa_provincia.htm'
    soup = make_soup(url)
    purchases = extract_purchases(soup)

    from pprint import pprint
    pprint(purchases)
info = pd.DataFrame(purchases)

# renombrando columnas no coincidentes por escritura
info.Provincia.replace({'Balears, Illes':'Islas Baleares','Palmas, Las':'Las Palmas','Girona':'Gerona',
                   'Lleida':'Lérida','Alicante/Alacant':'Alicante', 'Castellón/Castelló':'Castellón',
                    'Valencia/València':'Valencia', 'Coruña, A':'La Coruña','Ourense':'Orense',
                    'Araba/Álava':'Álava', 'Bizkaia':'Vizcaya','Gipuzkoa':'Guipúzcoa',
                     'Rioja, La':'La Rioja'},inplace=True)
# Acortando Títulos largos para mejor ajuste en visualización
info.CCAA.replace({'Asturias, Principado de':'Asturias','Balears, Illes':'Balears',
                 'Madrid, Comunidad de':'Madrid','Murcia, Región de':'Murcia',
                  'Navarra, Comunidad Foral de':'Navarra','Rioja, La': 'Rioja'},inplace=True)

info.drop([50,51,52],axis=0,inplace=True)

# Actualizando Diccionario
clave=info['Provincia']
valor = info['CCAA']
Dict = dict(zip(clave,valor))  
combined_set = [df_dic,df_ene] # combined 2 datasets for more efficient processing
#impFinanc_bins = [13,21,22,25,99999]
#impFinanc_labels = ['13','21','22','25+']


# Extrae información con Split
def get_info(dataset, feature_name):
    return dataset[feature_name].map(lambda name:name.split('/')[0].split('/')[0].strip())

# Extrae información mensual
def get_month(dataset, feature_name):
    return pd.to_datetime(dataset[feature_name]).map(lambda x: x.month)
# Extrae información año columna actual
def get_year(dataset, feature_name):
    return pd.to_datetime(dataset[feature_name]).map(lambda x: x.year)
# Extrae los meses que han transcurrido desde la  fecha de alta
# hasta la fecha actual.
def diff_month(dataset, d2):
    x=datetime.now()
    dataset[d2] = pd.to_datetime(dataset[d2])
    return ((x.year - dataset[d2].dt.year) * 12 + (x.month - dataset[d2].dt.month)-1).map(lambda x: x)

# Agrupa Categórias
def cut_levels(x, threshold, new_value):
    value_counts = x.value_counts()
    labels = value_counts.index[value_counts < threshold]
    x.loc[np.in1d(x, labels)] = new_value
    return x

for dataset in combined_set:
    dataset['num_lineas']= dataset['num_lineas'].astype(object)
    dataset['InfoNumdt'] = dataset['num_dt'].notnull().astype(int)
    dataset['Month_antig'] = diff_month(dataset, 'antiguedad')
    dataset['CCAA'] = dataset['provincia'].map(Dict)

cat_plot(df_dic, 'num_lineas', '_merge', survival_palette)
cat_plot(df_dic, 'InfoNumdt', '_merge', survival_palette)
plot = pd.crosstab(index=df_dic['_merge'],
            columns=df_dic['InfoNumdt']
                  ).apply(lambda r: r/r.sum() *100,
                          axis=0).plot(kind='bar', stacked=True)
for dataset in combined_set:
    dataset['diff_Month'] = ''
    dataset.loc[dataset['Month_antig'] < 24, 'diff_Month'] = '-2'
    dataset.loc[(dataset['Month_antig'] >= 24) & (dataset['Month_antig'] <= 84), 'diff_Month'] = '2-7'
    dataset.loc[(dataset['Month_antig'] > 84 ) & (dataset['Month_antig'] <= 120), 'diff_Month'] = '7-10'
    dataset.loc[(dataset['Month_antig'] > 120 ) & (dataset['Month_antig'] <= 240), 'diff_Month'] = '10-20'
    dataset.loc[dataset['Month_antig'] > 240, 'diff_Month'] = '20+'
cat_plot(df_dic, 'diff_Month', '_merge', survival_palette)
fig, axis = plt.subplots(1,1,figsize=(28,5))
sns.countplot(x='CCAA', hue='_merge', data=df_dic,
              palette=survival_palette,ax=axis)
axis.set_ylim(0,10000)
plt.show()
print(df_dic['CCAA'].value_counts())
#consu_all_dic = df_dic['mb_datos']-(df_dic['seg_llamad_ent'] + df_dic['seg_llamad_sal'])
#df_dic.insert(loc=len(df_dic.columns), column='consu_all_dic', value=consu_all_dic)

#lista=[]
#for val in df_dic['consu_all_dic']:
 #   if val < 0 :
 #       lista.append("No Ahorro")
 #   else:
  #      lista.append("Ahorro")
#df_dic['ahorro_call_dic']=lista 
df_dic.set_index('id', inplace=True)
df_ene.set_index('id',inplace=True)
df_train= df_dic.drop(['_merge','provincia','antiguedad','num_dt',
                       'Counts','Month_antig'],axis=1)
df_test = df_ene.drop(['provincia','antiguedad','num_dt','Month_antig'],axis=1)

y_train = df_dic['_merge']  # Relocate Survived target feature to y_train
X_train_analysis = df_train.copy()
#Codificación Etiquetas con LabelEncoder
lb_make = LabelEncoder()
#Clientes
X_train_analysis['incidencia'] = X_train_analysis['incidencia'].map({'NO': 0, 'SI': 1}).astype(int)
X_train_analysis['CCAA'] = lb_make.fit_transform(X_train_analysis['CCAA'])
X_train_analysis['diff_Month'] = lb_make.fit_transform(X_train_analysis['diff_Month'])
X_train_analysis['num_lineas'] = lb_make.fit_transform(X_train_analysis['num_lineas'])
#Productos
X_train_analysis['TV'] = X_train_analysis['TV'].map({'tv-futbol': 0, 'tv-familiar': 1, 'tv-total': 2}).astype(int)
X_train_analysis['conexion'] = X_train_analysis['conexion'].map({'ADSL': 0, 'FIBRA': 1}).astype(int)
X_train_analysis['vel_conexion'] = lb_make.fit_transform(X_train_analysis['vel_conexion'])

#Financiación
X_train_analysis['financiacion'] = X_train_analysis['financiacion'].map({'SI': 0, 'NO': 1}).astype(int)
X_train_analysis['descuentos'] = X_train_analysis['descuentos'].map({'SI': 0, 'NO': 1}).astype(int)

X_test_analysis = df_test.copy()
#Codificación Etiquetas con LabelEncoder
lb_make = LabelEncoder()
#Clientes
X_test_analysis['incidencia'] = X_test_analysis['incidencia'].map({'NO': 0, 'SI': 1}).astype(int)
X_test_analysis['CCAA'] = lb_make.fit_transform(X_test_analysis['CCAA'])
X_test_analysis['diff_Month'] = lb_make.fit_transform(X_test_analysis['diff_Month'])
X_test_analysis['num_lineas'] = lb_make.fit_transform(X_test_analysis['num_lineas'])
#Productos
X_test_analysis['TV'] = X_test_analysis['TV'].map({'tv-futbol': 0, 'tv-familiar': 1, 'tv-total': 2}).astype(int)
X_test_analysis['conexion'] = X_test_analysis['conexion'].map({'ADSL': 0, 'FIBRA': 1}).astype(int)
X_test_analysis['vel_conexion'] = lb_make.fit_transform(X_test_analysis['vel_conexion'])

#Financiación
X_test_analysis['financiacion'] = X_test_analysis['financiacion'].map({'SI': 0, 'NO': 1}).astype(int)
X_test_analysis['descuentos'] = X_test_analysis['descuentos'].map({'SI': 0, 'NO': 1}).astype(int)

# Generando copias algunos algoritmo su rendimiento es mejor con algún tipo
# de codificación particular

data_train = X_train_analysis.copy()
data_test = X_test_analysis.copy()
colormap = plt.cm.viridis
plt.figure(figsize=(14,14))
plt.title('Correlation between Features', y=1.05, size = 30)
sns.heatmap(X_train_analysis.corr(),
            linewidths=0.2, 
            vmax=2.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
features_num_train = X_train_analysis
def calculateVIF(features_num):
    features = list(features_num.columns)
    num_features = len(features)
    
    model = LinearRegression()
    
    result = pd.DataFrame(index = ['VIF'], columns = features)
    result = result.fillna(0)
    
    for ite in range(num_features):
        x_features = features[:]
        y_featue = features[ite]
        x_features.remove(y_featue)
        
        x = features_num[x_features]
        y = features_num[y_featue]
        
        model.fit(features_num[x_features],features_num[y_featue])
        
        result[y_featue] = 1/(1 - model.score(features_num[x_features],features_num[y_featue]))
    
    return result

num_vif = features_num_train.copy(deep = True)
features = list(num_vif.columns)
num_vif = num_vif[features]

calculateVIF(num_vif)
X_test_analysis.head(7)
X_train_analysis.head(7)
def automatic_selection(clas,dat_train):
    importances=clas.feature_importances_
    std = np.std([clas.feature_importances_ for tree in clas.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    sorted_important_features=[]
    predictors=dat_train.columns
    for i in indices: 
        sorted_important_features.append(predictors[i])
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(np.size(predictors)), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

    plt.xlim([-1, np.size(predictors)])
    return plt
rforest_checker = RandomForestClassifier(random_state = 0)
rforest_checker.fit(X_train_analysis, y_train)
importances_df = pd.DataFrame(rforest_checker.feature_importances_, columns=['Feature_Importance'],
                              index=X_train_analysis.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)
automatic_selection(rforest_checker,X_train_analysis)
rtree_checker=ExtraTreesClassifier(random_state = 0,n_jobs=1)
rtree_checker.fit(X_train_analysis, y_train)
importances_df = pd.DataFrame(rtree_checker.feature_importances_, columns=['Feature_Importance'],
                              index=X_train_analysis.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)
automatic_selection(rtree_checker,X_train_analysis)
#my_imp_dict = {'Feature Importance RandomForest' : pd.Series([0.361392,0.287002,0.111253,0.024982,0.023951,0.023640,0.022931,0.022705,
#                                                0.022627,0.019297,0.018609,0.013516,0.012645,0.011338,0.010214,
 #                                               0.006487,0.004787,0.002625],
  #           index=['InfoNumdt', 'incidencia', 'descuentos', 'mb_datos','seg_llamad_sal', 'seg_llamad_ent',
   #                 'facturacion', 'financiacion','num_llamad_ent','num_llamad_sal','edad','Month_antig',
    #                'CCAA','vel_conexion','RangeImpFinanc','num_lineas','TV','conexion'])}
#my_imp_df = pd.DataFrame(my_imp_dict)
#print(my_imp_df)
## métricas
def saca_metricas(y1, y2):
    print('matriz de confusión')
    print(confusion_matrix(y1, y2))
    print('accuracy')
    print(accuracy_score(y1, y2))
    print('precision')
    print(precision_score(y1, y2))
    print('recall')
    print(recall_score(y1, y2))
    print('f1')
    print(f1_score(y1, y2))
    false_positive_rate, recall, thresholds = roc_curve(y1, y2)
    roc_auc = auc(false_positive_rate, recall)
    print('AUC')
    print(roc_auc)
    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('AUC = %0.2f' % roc_auc)
    
    
 #funcion para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8,8))
    sns.heatmap(conf_matrix, xticklabels=True, yticklabels=True, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))
    
def draw_confusion_matrices(confusion_matricies,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        sns.heatmap(cm, xticklabels=True, yticklabels=True, annot=True,fmt="d");
        plt.title('Confusion matrix for %s' % classifier)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()    
# ya lo había hecho anteriormente pero soló para no olvidar lo comento aquí
target=df_dic['_merge']
target.unique()
target.head()
X_train, X_test, y_train, y_test = train_test_split(X_train_analysis,
                        target,
                        test_size=0.2,
                        random_state=56,
                        stratify = target)
print(X_train.shape,X_test.shape)
varModel=VarianceThreshold(threshold=0) #Estableciendo umbral de variación a 0
varModel.fit(X_train)
constArr=varModel.get_support()
#get_support() retorna True y False value para cada feature.
#True: Not a constant feature
#False: Constant feature
# Contando el número de features constantes y no constantes
collections.Counter(constArr)
#Non Constant feature:17
# Por tanto No hay features constantes
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)
print('Score list:', select_feature.scores_)
print('Feature list:', X_train.columns)
atrib = select_feature.get_support()
atributos = [X_train.columns[i] for i in list(atrib.nonzero()[0])]
atributos
X_train_2 = select_feature.transform(X_train)
X_test_2 = select_feature.transform(X_test)

#LogisticRegression classifier with n_estimators=10 (default)
classifier = RandomForestClassifier()      
classifier = classifier.fit(X_train_2,y_train)
y_pred = classifier.predict(X_test_2)
mostrar_resultados(y_test, y_pred)

##########################################3
#ac_2 = accuracy_score(y_test,classifier.predict(X_test_2))
#print('Accuracy is: ',ac_2)
#cm_2 = confusion_matrix(y_test,classifier.predict(X_test_2))
#sns.heatmap(cm_2,annot=True,fmt="d")
# Create the RFE object and rank each pixel
  
classifier_2 = RandomForestClassifier()      
rfe = RFE(estimator=classifier_2, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, y_train)
print('Chosen best 5 feature by rfe:',X_train.columns[rfe.support_])
X_train_3 = rfe.transform(X_train)
X_test_3 = rfe.transform(X_test)

#LogisticRegression classifier with n_estimators=10 (default)
classifier_2= RandomForestClassifier()    
classifier_2= classifier_2.fit(X_train_3,y_train)
y_pred = classifier_2.predict(X_test_3)
mostrar_resultados(y_test, y_pred)
# La puntuación de "precisión" es proporcional al número de clasificaciones correctas
classifier_3 = RandomForestClassifier() 
rfecv = RFECV(estimator=classifier_3, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])
# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(X_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()    

X_train = X_train.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)
X_test = X_test.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)

X_test_analysis = X_test_analysis.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)

X_train_analysis = X_train_analysis.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)

df_train = df_train.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)
df_test = df_test.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)

##
data_train = data_train.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)
data_test = data_test.drop(['num_llamad_ent','num_llamad_sal','edad','diff_Month',
                       'CCAA','vel_conexion','num_lineas','TV','conexion'], axis=1)
#####
X_train[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = X_train[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)

X_test[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = X_test[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)
## conjunto de validación
X_test_analysis[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = X_test_analysis[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)

X_train_analysis[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = X_train_analysis[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)
#### 
df_train[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = df_train[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)
df_test[['mb_datos','seg_llamad_ent',
          'seg_llamad_sal','facturacion']] = df_test[['mb_datos','seg_llamad_ent',
                                         'seg_llamad_sal','facturacion']].apply(zscore)
X_test_analysis.head(7)
X_train_analysis.head(7)
#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=7)
knn=knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
mostrar_resultados(y_test, y_pred)
#knn = KNeighborsClassifier(n_neighbors=7)
#knn=knn.fit(X_train_ene, y_train_ene)
#y_pred = knn.predict(X_test_ene)
#mostrar_resultados(y_test_ene, y_pred)
#knn.predict_proba(X_test_ene)
x_train = pd.get_dummies(df_train, columns=['incidencia','financiacion','descuentos','InfoNumdt'])
x_test = pd.get_dummies(df_test, columns=['incidencia','financiacion','descuentos','InfoNumdt'])
x_train.columns
x_train = x_train.drop(['incidencia_NO','financiacion_SI','descuentos_SI',
                          'InfoNumdt_0',], axis=1)
x_test = x_test.drop(['incidencia_NO','financiacion_SI','descuentos_SI',
                         'InfoNumdt_0',], axis=1)
x_train.head()
## classifier = LogisticRegression()

# First we need to train the classifier as usual
## classifier.fit(X_train, y_train)

# estimator = the classifier algorithm to use, cv = number of cross validation split
## acc_logreg = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# You can check the accuracy score for each split. In this case, 10 accuracy scores
## print(acc_logreg)

# Get mean of accuracy score of all cross validations
## acc_logreg.mean() 

# Standard deviation = differences of the accuracy score in each cross validations. the less = less variance = the better
## acc_logreg.std() 
logreg = LogisticRegression()
logreg.fit(X_train_analysis,target)
acc_logreg = cross_val_score(estimator = logreg, X = X_train_analysis, y = target, cv = 10)
logreg_acc_mean = acc_logreg.mean()
logreg_std = acc_logreg.std() 
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2 ) 
knn.fit(X_train_analysis,target)
acc_knn = cross_val_score(estimator = knn, X = X_train_analysis, y = target, cv = 10)
knn_acc_mean = acc_knn.mean()
knn_std = acc_knn.std()
ksvm = SVC(kernel = 'rbf', random_state = 0)
ksvm.fit(X_train_analysis,target)
acc_ksvm = cross_val_score(estimator = ksvm, X = X_train_analysis, y = target, cv = 10)
ksvm_acc_mean = acc_ksvm.mean()
ksvm_std = acc_ksvm.std()
naive = GaussianNB()
naive.fit(X_train_analysis,target)
acc_naive = cross_val_score(estimator = naive, X = X_train, y = target, cv = 10)
naive_acc_mean = acc_naive.mean()
naive_std = acc_naive.std()
dtree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dtree.fit(X_train_analysis,target)
acc_dtree = cross_val_score(estimator = dtree, X = X_train_analysis, y = target, cv = 10)
dtree_acc_mean = acc_dtree.mean()
dtree_std = acc_dtree.std()
rforest = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
rforest.fit(X_train_analysis,target)
acc_rforest = cross_val_score(estimator = rforest, X = X_train_analysis, y = target, cv = 10)
rforest_acc_mean = acc_rforest.mean()
rforest_std = acc_rforest.std()
xgb = XGBClassifier()
xgb.fit(X_train_analysis,target)
acc_xgb = cross_val_score(estimator = xgb, X = X_train_analysis, y = target, cv = 10)
xgb_acc_mean = acc_xgb.mean()
xgb_std = acc_xgb.std()
x_labels = ('Accuracy','Deviation')
y_labels = ('Logistic Regression','K-Nearest Neighbors','Kernel SVM','Naive Bayes'
            ,'Decision Tree','Random Forest','XGBoost')
score_array = np.array([[logreg_acc_mean, logreg_std],
                        [knn_acc_mean, knn_std],
                        [ksvm_acc_mean, ksvm_std],
                        [naive_acc_mean, naive_std],
                        [dtree_acc_mean, dtree_std],
                        [rforest_acc_mean, rforest_std],
                        [xgb_acc_mean, xgb_std]])  
fig = plt.figure(1)
fig.subplots_adjust(left=0.2,top=0.8, wspace=1)
ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)
score_table = ax.table(cellText=score_array,
                       rowLabels=y_labels,
                       colLabels=x_labels,
                       loc='upper center')
score_table.set_fontsize(14)
ax.axis("off") # Hide plot axis
fig.set_size_inches(w=18, h=10)
plt.show()
params_logreg = [{'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1','l2']}]
grid_logreg = GridSearchCV(estimator = LogisticRegression(),
                           param_grid = params_logreg,
                           scoring = 'accuracy',
                           cv = 10)
grid_logreg = grid_logreg.fit(X_train_analysis,target)
best_acc_logreg = grid_logreg.best_score_
best_params_logreg = grid_logreg.best_params_
params_knn = [{'n_neighbors': [5,7,8,9,10,11], 'metric': ['minkowski','hamming']}]
grid_knn = GridSearchCV(estimator = KNeighborsClassifier(),
                        param_grid = params_knn,
                        scoring = 'accuracy',
                        cv = 10)
grid_knn = grid_knn.fit(x_train, y_train)
best_acc_knn = grid_knn.best_score_
best_params_knn = grid_knn.best_params_
X_train_norm = x_train.copy()
X_train_norm[['incidencia_SI',
              'financiacion_NO',
              'descuentos_NO',
              'InfoNumdt_1']] = X_train_norm[['incidencia_SI', 
                                              'financiacion_NO',
                                              'descuentos_NO', 
                                              'InfoNumdt_1']].apply(zscore)
X_train_norm.head()
params_ksvm = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
               {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]},
               {'C': [0.1, 1, 10, 100], 'kernel': ['poly'],
                'degree': [1, 2, 3],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grid_ksvm = GridSearchCV(estimator = SVC(random_state = 0),
                         param_grid = params_ksvm,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs=-1)
grid_ksvm = grid_ksvm.fit(X_train_norm, y_train)  # Replace X_train with normalized version here
best_acc_ksvm = grid_ksvm.best_score_
best_params_ksvm = grid_ksvm.best_params_
params_dtree = [{'min_samples_split': [5, 10, 15, 20],
                 'min_samples_leaf': [1, 2, 3],
                 'max_features': ['auto', 'log2']}]
grid_dtree = GridSearchCV(estimator = DecisionTreeClassifier(criterion = 'gini', 
                                                             random_state = 0),
                            param_grid = params_dtree,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_dtree = grid_dtree.fit(X_train, y_train)
best_acc_dtree = grid_dtree.best_score_
best_params_dtree = grid_dtree.best_params_
params_rforest = [{'n_estimators': [200, 300],
                   'max_depth': [5, 7, 10],
                   'min_samples_split': [2, 4]}]
grid_rforest = GridSearchCV(estimator = RandomForestClassifier(criterion = 'gini', 
                                                               random_state = 0,
                                                               n_jobs=-1),
                            param_grid = params_rforest,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_rforest = grid_rforest.fit(X_train, y_train)
best_acc_rforest = grid_rforest.best_score_
best_params_rforest = grid_rforest.best_params_
grid_score_dict = {'1. Grid Search Score': [best_acc_logreg,best_acc_knn,best_acc_ksvm,'-',
                                            best_acc_dtree,best_acc_rforest,'(add later)'],
                   '2. Previous Score': [logreg_acc_mean,knn_acc_mean,ksvm_acc_mean,naive_acc_mean,
                                         dtree_acc_mean,rforest_acc_mean,xgb_acc_mean],
                   '3. Optimized Parameters': [best_params_logreg,best_params_knn,best_params_ksvm,'-',
                                               best_params_dtree,best_params_rforest,'(add later)'],
                  }
pd.DataFrame(grid_score_dict, index=['Logistic Regression','K-Nearest Neighbors','Kernel SVM','Naive Bayes',
                                     'Decision Tree','Random Forest','XGBoost'])
grid_score_dict = {'Best Score': [best_acc_logreg,
                                  best_acc_dtree,
                                  best_acc_rforest],
                   'Optimized Parameters': [best_params_logreg,
                                            best_params_dtree,
                                            best_params_rforest],
                  }
pd.DataFrame(grid_score_dict, index=['Logistic Regression',
                                     'Decision Tree',
                                     'Random Forest'])
""" params_rforest = [{'n_estimators': [100, 200, 500, 800], 
                   'min_samples_split': [5, 10, 15, 20],
                   'min_samples_leaf': [1, 2, 3],
                   'max_features': ['auto', 'log2']}] """
best_params_logreg 
best_params_dtree
best_params_rforest
logreg = LogisticRegression(C = 10, penalty = 'l1')
logreg.fit(X_train, y_train)
y_pred_train_logreg = cross_val_predict(logreg, X_train, y_train)
y_pred_test_logreg = logreg.predict(X_test)
ksvm = SVC(C = 1, gamma = 0.2, kernel = 'rbf', random_state = 0)
ksvm.fit(X_train, y_train)   # Replace X_train with X_train_norm here if you need
y_pred_train_ksvm = cross_val_predict(ksvm, X_train, y_train)
y_pred_test_ksvm = ksvm.predict(X_test)
dtree = DecisionTreeClassifier(criterion = 'gini', max_features='auto', min_samples_leaf=1, min_samples_split=5, random_state = 0)
dtree.fit(X_train, y_train)
y_pred_train_dtree = cross_val_predict(dtree, X_train, y_train)
y_pred_test_dtree = dtree.predict(X_test)
rforest = RandomForestClassifier(max_depth = 7, min_samples_split=2, n_estimators = 200, random_state = 0) # Grid Search best parameters
rforest.fit(X_train, y_train)
y_pred_train_rforest = cross_val_predict(rforest, X_train, y_train)
y_pred_test_rforest = rforest.predict(X_test)
y_test = np.array(y_test)
class_names = np.unique(y_test)
confusion_matrices = [
       ("R Logística",confusion_matrix(y_test,y_pred_test_logreg)),
       ("Tree",confusion_matrix(y_test, y_pred_test_dtree)),
       ("Random Forest",confusion_matrix(y_test, y_pred_test_rforest))
]

# Pyplot code not included to reduce clutter
draw_confusion_matrices(confusion_matrices,class_names)
logreg = LogisticRegression(C = 1, penalty = 'l1')
logreg.fit(X_train_analysis,target)
y_pred_train_logreg = cross_val_predict(logreg,X_train_analysis,target)
y_pred_test_logreg = logreg.predict_proba(X_test_analysis)
knn = KNeighborsClassifier(n_neighbors=7)
knn=knn.fit(X_train_analysis,target)
y_pred_train_knn = cross_val_predict(knn,X_train_analysis, target)
y_pred_test_knn = knn.predict_proba(X_test_analysis)
len(y_pred_test_logreg[0][predict_logreg]*100)
pd.value_counts(true_prob)

predict_logreg  = logreg.predict(X_test_analysis)
# Use 10 estimators so predictions are all multiples of 0.1
pred_prob =y_pred_test_logreg 
pred_churn = pred_prob[:,1]
is_churn = predict_logreg == 1

# Number of times a predicted probability is assigned to an observation
counts =pd.value_counts(pred_churn)

# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    Class_Predict = pd.Series(true_prob)

#true_prob = true_prob[true_prob == 1.0]
counts = pd.concat([counts,Class_Predict], axis=1).reset_index()
counts.columns = ['Pred_prob','count','Class_Predict']
X=X_test_analysis.reset_index()
table_prob = pd.concat((X,counts),axis=1)
table_prob.set_index('id',inplace=True)
x =table_prob[table_prob['Class_Predict']==1.0]
table_Noper = x[['Pred_prob','Class_Predict']]
x_labels = ('Pred_proba','Class_Predict')
y_labels = table_Noper.index
score_array = np.array(table_Noper) 
fig = plt.figure(1)
fig.subplots_adjust(left=0.2,top=0.8, wspace=1)
ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)
score_table = ax.table(cellText=score_array,
                       rowLabels=y_labels,
                       colLabels=x_labels,
                       loc='upper center')
score_table.set_fontsize(14)
ax.axis("off") # Hide plot axis
fig.set_size_inches(w=18, h=10)
plt.title('Probabilidades Fuga de Clientes Enero \n LogisticRegression', fontdict = {'fontsize' : 20})
plt.show()
predict_logreg  = logreg.predict(X_test_analysis)
# Use 10 estimators so predictions are all multiples of 0.1
pred_prob1 = y_pred_test_logreg 
pred_churn = pred_prob1[:,1]
pred_churn = np.round(pred_churn,1)
is_churn = predict_logreg  == 1

# Number of times a predicted probability is assigned to an observation
counts =pd.value_counts(pred_churn)

# calculate true probabilities 
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    Class_Predict = pd.Series(true_prob)

#true_prob = true_prob[true_prob == 1.0]
counts = pd.concat([counts,Class_Predict], axis=1).reset_index()
counts.columns = ['Pred_prob','Grouped_Customers','Class_Predict']

x_labels = ('Pred_proba','Grouped_Customers','Class_Predict')
#y_labels = counts.index
score_array = np.array(counts) 
fig = plt.figure(1)
fig.subplots_adjust(left=0.2,top=0.8, wspace=1)
ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)
score_table = ax.table(cellText=score_array,
                       #rowLabels=y_labels,
                       colLabels=x_labels,
                    loc='upper center')
score_table.set_fontsize(14)
ax.axis("off") # Hide plot axis
fig.set_size_inches(w=18, h=10)
plt.title('Cantidad de Clientes y Probabilidad de Abandono \n LogisticRegression', fontdict = {'fontsize' : 20})
plt.show()
### Calibración y discriminación
second_layer_train = pd.DataFrame( {'Logistic Regression': y_pred_train_logreg.ravel(),
                                    'Kernel SVM': y_pred_train_ksvm.ravel(),
                                    'Decision Tree': y_pred_train_dtree.ravel(),
                                    'Random Forest': y_pred_train_rforest.ravel()
                                    } )
second_layer_train.head()

X_train_second = np.concatenate(( y_pred_train_logreg.reshape(-1, 1), y_pred_train_ksvm.reshape(-1, 1), 
                                  y_pred_train_dtree.reshape(-1, 1), y_pred_train_rforest.reshape(-1, 1)),
                                  axis=1)
X_test_second = np.concatenate(( y_pred_test_logreg.reshape(-1, 1), y_pred_test_ksvm.reshape(-1, 1), 
                                 y_pred_test_dtree.reshape(-1, 1), y_pred_test_rforest.reshape(-1, 1)),
                                 axis=1)

xgb = XGBClassifier(
        n_estimators= 800,
        max_depth= 4,
        min_child_weight= 2,
        gamma=0.9,                        
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(X_train_second, y_train)

y_pred = xgb.predict(X_test_second)

def run_model_balanced(X_Train,y_Train):
    clf = LogisticRegression(C=0.5,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    clf.fit(X_train_analysis,target)
    return clf

model_one = run_model_balanced(X_train_analysis,target)
y_pred_proba = model_one.predict_proba(X_test_analysis)
predict_model_one  = model_one.predict(X_test_analysis)
# Use 10 estimators so predictions are all multiples of 0.1
pred_prob_one=y_pred_proba
pred_churn = pred_prob_one[:,1]
pred_churn = np.round(pred_churn,1)
is_churn = predict_model_one == 1

# Number of times a predicted probability is assigned to an observation
counts =pd.value_counts(pred_churn)

# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    Class_Predict = pd.Series(true_prob)

#true_prob = true_prob[true_prob == 1.0]
counts = pd.concat([counts,Class_Predict], axis=1).reset_index()
counts.columns = ['Pred_prob','count','Class_Predict']
counts
us = NearMiss(ratio=0.5, n_neighbors=3, version=2, random_state=1)
X_train_res, y_train_res = us.fit_sample(X_train_analysis,target)
 
print ("Distribution before resampling {}".format(Counter(target)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))
 
model_two= run_model(X_train_res,y_train_res)
y_pred = model.predict(X_test_analysis)
os =  RandomOverSampler(ratio=0.5)
X_train_res, y_train_res = os.fit_sample(X_train_analysis,target)
 
print ("Distribution before resampling {}".format(Counter(target)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))
 
model_three = run_model(X_train_res,y_train_res)
y_pred = model.predict(X_test_analysis)
os_us = SMOTETomek(ratio=0.5)
X_train_res, y_train_res = os_us.fit_sample(X_train_analysis,target)
 
print ("Distribution before resampling {}".format(Counter(target)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))
 
model = run_model(X_train_res,y_train_res)
y_pred = model.predict(X_test_analysis)
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
 
#Train the classifier.
bbc.fit(X_train_analysis,target)
y_pred = bbc.predict(X_test_analysis)