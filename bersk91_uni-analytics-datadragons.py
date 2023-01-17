import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt #visualización
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix,roc_curve,auc,roc_auc_score

from sklearn.linear_model import LogisticRegression

#Importando la data de prueba
ds = pd.read_csv("../input/Churn_Modelling_Sample.txt", sep = "\t")
ds.head()
ds.columns.values
df = ds[['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']]
######################################################
# 1. Análisis estadísticos de los datos
######################################################

# 1.1 Análisis Univariado

# a. Generar los inputs para el analisis univariado

v=pd.DataFrame({"variable": df.columns.values})
#Seteo de Variables Categoricas
df.Geography = df.Geography.astype('category')
df.Gender = df.Gender.astype('category')
df.HasCrCard = df.HasCrCard.astype('category')
df.IsActiveMember = df.IsActiveMember.astype('category')
df.Exited = df.Exited.astype('category')

df.dtypes

t=pd.DataFrame({"tipo": df.dtypes.values})
meta = pd.concat([v, t], axis=1)
# b. GENERACIÓN DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=df[v].value_counts() 
        fr=fa/len(df[v]) 
        #Barras
        plt.subplot(1,2,1)
        plt.bar(fa.index,fa)
        plt.xticks(fa.index)
        plt.title(v)
        #Pie
        plt.subplot(1,2,2)
        plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(fr.index,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(v)
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(df[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(df[v])
        plt.title(v)
    plt.show()
# c. GENERACION DE INDICADORES
import scipy.stats as sc 

for i in range(len(meta)) :
    v=meta.iloc[i].variable 
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        x=pd.DataFrame({"var":v,"mean": ".",
                        "median": ".",
                        "mode": ".",
                        "min": ".",
                        "max": ".",
                        "sd": ".",
                        "cv": ".",
                        "k": ".",
                        "Q1": ".",
                        "Q3": ".",
                        "Nmiss": "."
                        },index=[i])
    else:
        x=pd.DataFrame({"var":v,"mean": df[v].mean(),
                        "median": df[v].median(),
                        "mode": df[v].mode(),
                        "min": df[v].min(),
                        "max": df[v].max(),
                        "sd": df[v].std(),
                        "cv": df[v].std()/df[v].mean(),
                        "k": sc.kurtosis(df[v]),
                        "Q1": np.percentile(df[v],q=25),
                        "Q3": np.percentile(df[v],q=75),
                        "Nmiss": df[v].isnull().sum()
                        }, index=[i])
    if(i==0):
        x1=x
    else:
        x1 = pd.concat([x1, x]) #x1.append(x)

x1
df.describe()
######################################################
# 1. Análisis estadísticos de los datos
######################################################

# 1.2 Análisis Bivariado

# a. GENERACION DE GRAFICOS

#target:
y="Exited"
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if v==y: break
    print(v)
    if (t.__class__.__name__=="CategoricalDtype"):        
        g=df.groupby([df[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado')
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    else:
        d=pd.qcut(df[v], 10, duplicates='drop',labels=False)     
        g=df.groupby(['Exited', d]).size().unstack(0)   
        N = len(g)
        menMeans = g[0]
        womenMeans = g[1]
        tf= g[1]/(g[0]+g[1])
        ind = np.arange(N)    # the x locations for the groups

        width = 0.9       # the width of the bars: can also be len(x) sequence        
        p1 = plt.bar(ind, menMeans, width)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans)
        
        plt.ylabel('Freq')
        plt.xlabel("Deciles " + v)
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(ind, np.arange(1,10,1))
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    plt.show()
plt.figure(figsize=(14,12))
sns.heatmap(df.corr(), vmin = -1.0, vmax=1.0, annot=True)
plt.show()
######################################################
# 2. Tratamiento de datos
######################################################

#Luego de evaluar los resultados en el bivariado y univariado, se debe realizar 
#la seleccion de variable categórica, así como el tratamiento de los outliers.

# 2.1 Segmentación

#1era Segmentación (BiVar 1) : Geography : (France, Spain) y (Germany)

df_PB=df[df["Geography"]!="Germany"]
df_PA=df[df["Geography"]=="Germany"]
# a. GENERACIÓN DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=df_PB[v].value_counts() 
        fr=fa/len(df_PB[v]) 
        #Barras
        plt.subplot(1,2,1)
        plt.bar(fa.index,fa)
        plt.xticks(fa.index)
        plt.title(v)
        #Pie
        plt.subplot(1,2,2)
        plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(fr.index,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(v)
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(df_PB[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(df_PB[v])
        plt.title(v)
    plt.show()
# 2.2 Tratamiento de datos
 
######Segmento PB#######
#Llamar boxplot por boxplot
plt.boxplot(df_PB["Age"])
plt.boxplot(df_PB["CreditScore"])
plt.boxplot(df_PB["NumOfProducts"])

#CreditScore outlier
np.percentile(df_PB['CreditScore'],q=1) # P1: 431
df_PB.loc[df_PB["CreditScore"]<=431,"CreditScore"]=431

#Age outlier
np.percentile(df_PB['Age'],q=94) # P94: 59
df_PB.loc[df_PB["Age"]>=59,"Age"]=59

#NumOfProducts outlier
np.percentile(df_PB['NumOfProducts'],q=99) # P99: 3
df_PB.loc[df_PB["NumOfProducts"]>=3,"NumOfProducts"]=3

plt.boxplot(df_PB["Age"])
plt.boxplot(df_PB["CreditScore"])
plt.boxplot(df_PB["NumOfProducts"])
##Una vez realizada la categorización, repetir el analisis del paso 1

# 1.1 Análisis Univariado

ds1=df_PB

v=pd.DataFrame({"variable": ds1.columns.values})
#Seteo de Variables Categoricas
ds1.Geography=ds1.Geography.astype('category')
ds1.Gender=ds1.Gender.astype('category')
ds1.HasCrCard=ds1.HasCrCard.astype('category')
ds1.IsActiveMember=ds1.IsActiveMember.astype('category')
ds1.Exited=ds1.Exited.astype('category')

t=pd.DataFrame({"tipo": ds1.dtypes.values})
meta = pd.concat([v, t], axis=1)
#GENERACION DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=ds1[v].value_counts() 
        fr=fa/len(ds1[v]) 
        #Barras
        plt.subplot(1,2,1)
        plt.bar(fa.index,fa)
        plt.xticks(fa.index)
        plt.title(v)
        #Pie
        plt.subplot(1,2,2)
        plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(fr.index,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(v)
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(ds1[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(ds1[v])
        plt.title(v)
    plt.show()
#GENERACION DE INDICADORES 
import scipy.stats as sc 
for i in range(len(meta)) :
    v=meta.iloc[i].variable 
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        x=pd.DataFrame({"var":v,"mean": ".",
                        "median": ".",
                        "mode": ".",
                        "min": ".",
                        "max": ".",
                        "sd": ".",
                        "cv": ".",
                        "k": ".",
                        "Q1": ".",
                        "Q3": ".",
                        "Nmiss": "."
                        },index=[i])
    else:
        P25=np.percentile(ds1[v],q=25)
        P75=np.percentile(ds1[v],q=75)
        IQR=P75-P25
        liV=P25-1.5*IQR # Mimimo Viable
        lsV=P75+1.5*IQR # Maximo Viable
        x=pd.DataFrame({"var":v,"mean": ds1[v].mean(),
                        "median": ds1[v].median(),
                        "mode": ds1[v].mode(),
                        "min": ds1[v].min(),
                        "max": ds1[v].max(),
                        "sd": ds1[v].std(),
                        "cv": ds1[v].std()/ds1[v].mean(),
                        "k": sc.kurtosis(ds1[v]),
                        "Q1": np.percentile(ds1[v],q=25),
                        "Q3": np.percentile(ds1[v],q=75),
                        "Nmiss": ds1[v].isnull().sum()/len(ds1)
                        }, index=[i])
    if(i==0):
        x1=x
    else:
       x1=pd.concat([x1, x])  # x1.append(x)
    
x1  
# 1.1 Análisis Bivariado

#target:
y="Exited"
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(11,6))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if v==y: break
    print(v)
    if (t.__class__.__name__=="CategoricalDtype"):        
        g=ds1.groupby([ds1[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    else:
        d=pd.qcut(ds1[v], 10, duplicates='drop',labels=False)     
        g=ds1.groupby(['Exited', d]).size().unstack(0)   
        N = len(g)
        menMeans = g[0]
        womenMeans = g[1]
        tf= g[1]/(g[0]+g[1])
        ind = np.arange(N)    # the x locations for the groups

        width = 0.9       # the width of the bars: can also be len(x) sequence        
        p1 = plt.bar(ind, menMeans, width)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans)
        
        plt.ylabel('Freq')
        plt.xlabel("Deciles " + v)
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(ind, np.arange(1,10,1))
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    plt.show()
#######################################################
######Segmento PA#######

# a. GENERACIÓN DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=df_PA[v].value_counts() 
        fr=fa/len(df_PA[v]) 
        #Barras
        plt.subplot(1,2,1)
        plt.bar(fa.index,fa)
        plt.xticks(fa.index)
        plt.title(v)
        #Pie
        plt.subplot(1,2,2)
        plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(fr.index,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(v)
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(df_PA[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(df_PA[v])
        plt.title(v)
    plt.show()
# 2.2 Tratamiento de datos
 
######Segmento PA#######
#Llamar boxplot por boxplot
plt.boxplot(df_PA["Age"])
plt.boxplot(df_PA["CreditScore"])
plt.boxplot(df_PA["NumOfProducts"])
plt.boxplot(df_PA["Balance"])

#CreditScore outlier
np.percentile(df_PA['CreditScore'],q=0.5) # P0.5: 416
df_PA.loc[df_PA["CreditScore"]<=416,"CreditScore"]=416

#Age outlier
np.percentile(df_PA['Age'],q=97) # P97: 65
df_PA.loc[df_PA["Age"]>=65,"Age"]=65

#NumOfProducts outlier
np.percentile(df_PA['NumOfProducts'],q=99) # P99: 3
df_PA.loc[df_PA["NumOfProducts"]>=3,"NumOfProducts"]=3

#Balance outlier
np.percentile(df_PA['Balance'],q=99) # P99: 185170
df_PA.loc[df_PA["Balance"]>=185170,"Balance"]=185170
np.percentile(df_PA['Balance'],q=0.8) # P0.8: 54000
df_PA.loc[df_PA["Balance"]<=54000,"Balance"]=54000

plt.boxplot(df_PA["Age"])
plt.boxplot(df_PA["CreditScore"])
plt.boxplot(df_PA["NumOfProducts"])
plt.boxplot(df_PA["Balance"])
# 1.1 Análisis Univariado

ds2=df_PA

v=pd.DataFrame({"variable": ds2.columns.values})
#Seteo de Variables Categoricas
ds2.Geography=ds2.Geography.astype('category')
ds2.Gender=ds2.Gender.astype('category')
ds2.HasCrCard=ds2.HasCrCard.astype('category')
ds2.IsActiveMember=ds2.IsActiveMember.astype('category')
ds2.Exited=ds2.Exited.astype('category')

t=pd.DataFrame({"tipo": ds2.dtypes.values})
meta = pd.concat([v, t], axis=1)
#GENERACION DE INDICADORES 
import scipy.stats as sc 
for i in range(len(meta)) :
    v=meta.iloc[i].variable 
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        x=pd.DataFrame({"var":v,"mean": ".",
                        "median": ".",
                        "mode": ".",
                        "min": ".",
                        "max": ".",
                        "sd": ".",
                        "cv": ".",
                        "k": ".",
                        "Q1": ".",
                        "Q3": ".",
                        "Nmiss": "."
                        },index=[i])
    else:
        P25=np.percentile(ds2[v],q=25)
        P75=np.percentile(ds2[v],q=75)
        IQR=P75-P25
        liV=P25-1.5*IQR # Mimimo Viable
        lsV=P75+1.5*IQR # Maximo Viable
        x=pd.DataFrame({"var":v,"mean": ds2[v].mean(),
                        "median": ds2[v].median(),
                        "mode": ds2[v].mode(),
                        "min": ds2[v].min(),
                        "max": ds2[v].max(),
                        "sd": ds2[v].std(),
                        "cv": ds2[v].std()/ds2[v].mean(),
                        "k": sc.kurtosis(ds2[v]),
                        "Q1": np.percentile(ds2[v],q=25),
                        "Q3": np.percentile(ds2[v],q=75),
                        "Nmiss": ds2[v].isnull().sum()/len(ds2)
                        }, index=[i])
    if(i==0):
        x1=x
    else:
       x1=pd.concat([x1, x])  # x1.append(x)
    
x1
# 1.1 Análisis Bivariado

#target:
y="Exited"
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(11,6))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if v==y: break
    print(v)
    if (t.__class__.__name__=="CategoricalDtype"):        
        g=ds2.groupby([ds2[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    else:
        d=pd.qcut(ds2[v], 10, duplicates='drop',labels=False)     
        g=ds2.groupby(['Exited', d]).size().unstack(0)   
        N = len(g)
        menMeans = g[0]
        womenMeans = g[1]
        tf= g[1]/(g[0]+g[1])
        ind = np.arange(N)    # the x locations for the groups

        width = 0.9       # the width of the bars: can also be len(x) sequence        
        p1 = plt.bar(ind, menMeans, width)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans)
        
        plt.ylabel('Freq')
        plt.xlabel("Deciles " + v)
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(ind, np.arange(1,10,1))
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
    plt.show()
###############################
#Dummies para países Bajos
###############################
dummies = pd.get_dummies(ds1[['Geography','Gender']])
ds1 = ds1.join(dummies).copy()
ds1.drop(['Geography','Gender','Gender_Female','Geography_Germany','Geography_Spain'],axis=1,inplace= True)
ds1.head()
###############################
#Dummies para países Bajos
###############################
dummies = pd.get_dummies(ds2[['Geography','Gender']])
ds2 = ds2.join(dummies).copy()
ds2.drop(['Geography','Gender','Gender_Female','Geography_France','Geography_Germany','Geography_Spain'],axis=1,inplace= True)
ds2.head()
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

y = ds1.Exited.copy()
X = ds1.drop('Exited',axis=1).copy()
y = y.astype('int64')
X.IsActiveMember = X.IsActiveMember.astype('int64')
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

parameters = {'max_depth':range(3,20)}
scoring = ['accuracy','balanced_accuracy','roc_auc','f1','precision','recall']

for i in range(X.shape[1]-2):
    
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring= scoring, refit='roc_auc', return_train_score =True)
    clf.fit(X=X, y=y)
    
    
    idx = clf.cv_results_['params'].index(clf.best_params_)
    resultado =  clf.best_params_
    resultado.update({'{}'.format(s) : clf.cv_results_['mean_test_{}'.format(s)][idx] for s in scoring})
    
    if i==0:
        df_resultado = pd.Series(resultado,name = 'todos_incluidos').to_frame()
        modelos = {'todos_incluidos':clf}
        col_variables = {'todos_incluidos':X.columns}
    else:
        df_resultado = df_resultado.join(pd.Series(resultado,name = 'sin_{}'.format(drop_col))).copy()
        modelos.update({'sin_{}'.format(drop_col):clf })
        col_variables.update({'sin_{}'.format(drop_col):X.columns })
    
    drop_col = X.columns[np.argmin(clf.best_estimator_.feature_importances_)]
    X.drop(drop_col,axis=1,inplace=True)
    
df_resultado
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

pd.Series(modelos['todos_incluidos'].best_estimator_.feature_importances_*100,\
          index=col_variables['todos_incluidos']).sort_values().plot.barh(figsize=(15,10))
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

ax = pd.DataFrame({'Prueba':        modelos['sin_EstimatedSalary'].cv_results_['mean_test_roc_auc'],\
                   'Entrenamiento': modelos['sin_EstimatedSalary'].cv_results_['mean_train_roc_auc']},\
                  index=[i['max_depth'] for i in modelos['sin_EstimatedSalary'].cv_results_['params']]).plot(figsize=(10,10))
ax.set_xlabel('Max_Depth')

ax.set_ylabel('ROC_AUC')
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

dot_data = tree.export_graphviz(modelos['sin_EstimatedSalary'].best_estimator_, out_file=None, 
                     feature_names=col_variables['sin_EstimatedSalary'],  
                     class_names=np.array(['No Exited', 'Exited']),  
                     filled=True, rounded=True,  
                     special_characters=True)  
graphviz.Source(dot_data)  
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

cm = confusion_matrix(y,modelos['sin_EstimatedSalary'].best_estimator_.predict(ds1[col_variables['sin_EstimatedSalary']]))
ax = sns.heatmap(cm,annot=True, fmt='.0f',cbar=False)
ax.set(xlabel='Condición Predecida', ylabel='Condición Actual')
plt.show()
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

print('Sensibilidad: {:.1f}%\nEspecificidad: {:.1f}%'.format(cm[1,1]/cm[1].sum()*100,cm[0,0]/cm[0].sum()*100))
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES BAJOS #####################

from sklearn import metrics
probs = modelos['sin_EstimatedSalary'].best_estimator_.predict_proba(ds1[col_variables['sin_EstimatedSalary']])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

y = ds2.Exited.copy()
X = ds2.drop('Exited',axis=1).copy()
y = y.astype('int64')
X.IsActiveMember = X.IsActiveMember.astype('int64')
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

parameters = {'max_depth':range(3,20)}
scoring = ['accuracy','balanced_accuracy','roc_auc','f1','precision','recall']

for i in range(X.shape[1]-2):
    
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring= scoring, refit='roc_auc', return_train_score =True)
    clf.fit(X=X, y=y)
    
    
    idx = clf.cv_results_['params'].index(clf.best_params_)
    resultado =  clf.best_params_
    resultado.update({'{}'.format(s) : clf.cv_results_['mean_test_{}'.format(s)][idx] for s in scoring})
    
    if i==0:
        df_resultado = pd.Series(resultado,name = 'todos_incluidos').to_frame()
        modelos = {'todos_incluidos':clf}
        col_variables = {'todos_incluidos':X.columns}
    else:
        df_resultado = df_resultado.join(pd.Series(resultado,name = 'sin_{}'.format(drop_col))).copy()
        modelos.update({'sin_{}'.format(drop_col):clf })
        col_variables.update({'sin_{}'.format(drop_col):X.columns })
    
    drop_col = X.columns[np.argmin(clf.best_estimator_.feature_importances_)]
    X.drop(drop_col,axis=1,inplace=True)
    
df_resultado
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

pd.Series(modelos['todos_incluidos'].best_estimator_.feature_importances_*100,\
          index=col_variables['todos_incluidos']).sort_values().plot.barh(figsize=(15,10))
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

ax = pd.DataFrame({'Prueba':        modelos['sin_EstimatedSalary'].cv_results_['mean_test_roc_auc'],\
                   'Entrenamiento': modelos['sin_EstimatedSalary'].cv_results_['mean_train_roc_auc']},\
                  index=[i['max_depth'] for i in modelos['sin_EstimatedSalary'].cv_results_['params']]).plot(figsize=(10,10))
ax.set_xlabel('Max_Depth')

ax.set_ylabel('ROC_AUC')
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

dot_data = tree.export_graphviz(modelos['sin_EstimatedSalary'].best_estimator_, out_file=None, 
                     feature_names=col_variables['sin_EstimatedSalary'],  
                     class_names=np.array(['No Exited', 'Exited']),  
                     filled=True, rounded=True,  
                     special_characters=True)  
graphviz.Source(dot_data)  
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

cm = confusion_matrix(y,modelos['sin_EstimatedSalary'].best_estimator_.predict(ds2[col_variables['sin_EstimatedSalary']]))
ax = sns.heatmap(cm,annot=True, fmt='.0f',cbar=False)
ax.set(xlabel='Condición Predecida', ylabel='Condición Actual')
plt.show()
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

print('Sensibilidad: {:.1f}%\nEspecificidad: {:.1f}%'.format(cm[1,1]/cm[1].sum()*100,cm[0,0]/cm[0].sum()*100))
############### ARBOL DE DECISIÓN #####################
############### ANALISIS PARA DATA DE PAISES ALTOS #####################

from sklearn import metrics
probs = modelos['sin_EstimatedSalary'].best_estimator_.predict_proba(ds2[col_variables['sin_EstimatedSalary']])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Desde este punto, hacer el análisis para regresión logística

##########################################################
# 3.1 Múltiples Entrenamientos
##########################################################

