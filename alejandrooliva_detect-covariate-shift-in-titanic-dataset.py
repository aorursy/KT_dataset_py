# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


"""
Función encargada de categorizar la característica tiquet
"""
def process_ticket(data):
    ticket = []
    for x in list(data.Ticket):
        if x.isdigit():#unicamente para valores numericos
            ticket.append('N')
        else: #resto de tickets que contienen caracteres 
            ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])
    
    #remplazar valores para la caracteristica ticket
    data.Ticket = ticket
    
    #reducir las categorias, mantener solo el primer caracter de las diferentes categorias
    data.Ticket = data.Ticket.apply(lambda x : x[0])
    
    return data
#cargar dataset
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#añadir nueva caracteristica Oringin; 0='production' 1='train'
train_df['Origin'] = 0
test_df['Origin'] = 1


#Comprobar valores a NaN en el dataset
#print(train_df.isna().sum(),test_df.isna().sum())
train_df.Age.fillna(value = -1, inplace = True) #Cambiar NaN a valor X
test_df.Age.fillna(value = -1, inplace = True) #Cambiar NaN a valor X

#Procesar nombre por MR. Sr. y categorizar en 6 categorias 
train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.')
test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.')
train_df.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)
train_df.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)
train_df.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
test_df.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)
test_df.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)
test_df.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
train_df['Title'] = train_df['Title'].astype('category').cat.codes
test_df['Title'] = test_df['Title'].astype('category').cat.codes

# eliminar características (nombre,id,survived(train))
train_df=train_df.drop("PassengerId", axis=1)
train_df=train_df.drop("Name", axis=1)
train_df=train_df.drop("Survived", axis=1)
test_df=test_df.drop("PassengerId", axis=1)
test_df=test_df.drop("Name", axis=1)

# cambiar valor para la característica 'embarked'
train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 0
train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 1
train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 2
test_df.loc[test_df['Embarked'] == 'S', 'Embarked'] = 0
test_df.loc[test_df['Embarked'] == 'C', 'Embarked'] = 1
test_df.loc[test_df['Embarked'] == 'Q', 'Embarked'] = 2

#pasamos a categorias el sexo del pasajero (male:1 female:0)
train_df['Sex'] = train_df['Sex'].astype('category').cat.codes
test_df['Sex'] = test_df['Sex'].astype('category').cat.codes


#pasamos a categorias los códigos alfanuméricos de la cabin
#train_df.Cabin.fillna(value = 'X', inplace = True) #Cambiar NaN a valor X
train_df.Cabin = train_df.Cabin.apply( lambda x : x[0] if(pd.notnull(x)) else x) #Manterner primer caracter para categorizar

#test_df.Cabin.fillna(value = 'X', inplace = True) #Cambiar NaN a valor X
test_df.Cabin = test_df.Cabin.apply( lambda x : x[0]  if(pd.notnull(x)) else x) #Manterner primer caracter para categorizar

train_df['Cabin'] = train_df['Cabin'].astype('category').cat.codes
test_df['Cabin'] = test_df['Cabin'].astype('category').cat.codes

print(train_df.Cabin.unique())
print(test_df.Cabin.unique())

#train_df = train_df.drop(['Cabin'], axis=1)
#test_df = test_df.drop(['Cabin'], axis=1)

#Revisar
train_df=process_ticket(train_df)
test_df=process_ticket(test_df)
train_df['Ticket'] = train_df['Ticket'].astype('category').cat.codes
test_df['Ticket'] = test_df['Ticket'].astype('category').cat.codes

#ver distribucion de la cabina
combine_df=pd.concat([train_df,test_df], axis=0, ignore_index=True)
plt.hist(combine_df.loc[combine_df['Origin'] == 0,"Cabin"], color="skyblue", label="Origin=0")
plt.hist(combine_df.loc[combine_df['Origin'] == 1,"Cabin"],color="red",alpha=.75, label="Origin=1")
plt.legend()
plt.show()

#crear nuevo conjunto de train a partir del 80% de train_df y 80% de test
train_c=pd.concat([train_df[:8*len(train_df)//10], test_df[:8*len(test_df)//10]], sort = False)
#20% de train y test corresponde a test_df
test_c=pd.concat([train_df[8*len(train_df)//10:], test_df[8*len(test_df)//10:]], sort = False)

#cambiar NaN a -1
train_c=train_c.fillna(-1)
test_c=test_c.fillna(-1)

print(train_c['Origin'].groupby(train_c['Cabin']).count())
print(test_c['Origin'].groupby(test_c['Cabin']).count())

#Matriz de correlación
correlation = train_c.loc[:, ['Pclass','Sex','Age','SibSp','Parch',
 'Fare','Embarked','Ticket','Origin','Cabin','Title']]
correlation = correlation.agg(LabelEncoder().fit_transform)
correlation['Origin'] = train_c.Origin 
correlation = correlation.set_index('Origin').reset_index() # Move Age at index 0.

plt.figure(figsize = (20,7))
sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
plt.title('Variables Correlated with Origin', fontsize = 18)
plt.show()
combine_df=pd.concat([train_c,test_c], axis=0, ignore_index=True)
plt.hist(combine_df.loc[combine_df['Origin'] == 0,"Cabin"], color="skyblue",alpha=.75, label="Origin=0")
plt.hist(combine_df.loc[combine_df['Origin'] == 1,"Cabin"],color="red",alpha=.75, label="Origin=1")
plt.legend()
plt.show()
X_train_o = train_c.drop("Origin", axis=1)
Y_train_o = train_c["Origin"]
X_test_o  = test_c.drop("Origin", axis=1)
y_test_o = test_c["Origin"]
"""
Calculo de métricas Precisión, Recall, F1-Score
"""
from sklearn.metrics import roc_auc_score
from sklearn import metrics
def metricas(y_t,y_p):
    p=metrics.precision_score(y_t,y_p,average='weighted')
    r=metrics.recall_score(y_t,y_p,average='weighted')
    f1=metrics.f1_score(y_t,y_p,average='weighted')
    return p,r,f1
def plot_curve(y_t,y_p):   
    fpr, tpr, umbral = roc_curve(y_t, y_p)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()
def b_ratio(probs):
    probs=probs+1e-20 #suavizar valores a 0
    b=(1./probs)-1 #calcular pesos
    b/= np.mean(b) #normalizar pesos
    sns.distplot(b, kde=False)
    plt.show()
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_o, Y_train_o)
Y_pred_o = random_forest.predict(X_test_o)
acc_random_forest = round(random_forest.score(X_test_o, y_test_o) * 100, 2)

probs = random_forest.predict_proba(X_test_o)[:, 0] #prob por muestra P(train|x)
b_ratio(probs)

print("Acc: ",acc_random_forest)
print("Phi Coefficiente: ",matthews_corrcoef(y_test_o, Y_pred_o))
print("ROC-AUC: ",roc_auc_score(y_test_o, Y_pred_o))
plot_curve(y_test_o,Y_pred_o)
print('---------------------------------------------------------')
comb_df=pd.concat([train_df,test_df],sort=False)
comb_df=comb_df.sample(frac=1)

train_df=comb_df[:8*len(comb_df)//10]
train_df['Origin']=0

test_df=comb_df[8*len(comb_df)//10:]
test_df['Origin']=1

#crear nuevo conjunto de train a partir del 80% de train_df y 80% de test
train_c=pd.concat([train_df[:8*len(train_df)//10], test_df[:8*len(test_df)//10]], sort = False)
test_c=pd.concat([train_df[8*len(train_df)//10:], test_df[8*len(test_df)//10:]], sort = False)

#cambiar NaN a -1
train_c=train_c.fillna(-1)
test_c=test_c.fillna(-1)

combine_df=pd.concat([train_c,test_c], axis=0, ignore_index=True)

plt.hist(combine_df.loc[combine_df['Origin'] == 0,"Cabin"], color="skyblue",alpha=.75, label="Origin=0")
plt.hist(combine_df.loc[combine_df['Origin'] == 1,"Cabin"],color="red",alpha=.75, label="Origin=1")
plt.legend()
plt.show()

X_train_o = train_c.drop("Origin", axis=1)
Y_train_o = train_c["Origin"]
X_test_o  = test_c.drop("Origin", axis=1)
y_test_o = test_c["Origin"]
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_o, Y_train_o)
Y_pred_o = random_forest.predict(X_test_o)
acc_random_forest = round(random_forest.score(X_test_o, y_test_o) * 100, 2)

probs = random_forest.predict_proba(X_test_o)[:, 0] #prob por muestra P(train|x)
b_ratio(probs)

print("Acc: ",acc_random_forest)
p,r,f1=metricas(y_test_o,Y_pred_o)
print("Phi Coefficiente: ",matthews_corrcoef(y_test_o, Y_pred_o))
print("ROC-AUC: ",roc_auc_score(y_test_o, Y_pred_o))
plot_curve(y_test_o,Y_pred_o)
print('---------------------------------------------------------')
#Crear training con las muestra de sex:1 == Male

ma_train=train_c.loc[train_c['Sex'] == 1]
ma_test=test_c.loc[test_c['Sex'] == 1]
train_m=pd.concat([ma_train, ma_test], sort = False)
train_m["Origin"]= 0

#Crear teste con las muestras de sex:0 == Female
fe_train=train_c.loc[train_c['Sex'] == 0]
fe_test=test_c.loc[test_c['Sex'] == 0]
test_m=pd.concat([fe_train,fe_test], sort = False)
test_m["Origin"]= 1

#crear nuevo conjunto de train a partir del 80% de train_df y 80% de test
train_mf=pd.concat([train_m[:8*len(train_m)//10], test_m[:8*len(test_m)//10]], sort = False)
test_mf=pd.concat([train_m[8*len(train_m)//10:], test_m[8*len(test_m)//10:]], sort = False)

combine_df=pd.concat([train_mf,test_mf], axis=0, ignore_index=True)
plt.hist(combine_df.loc[combine_df['Origin'] == 0,"Sex"], color="skyblue",alpha=.75, label="Origin=0")
plt.hist(combine_df.loc[combine_df['Origin'] == 1,"Sex"],color="red",alpha=.75, label="Origin=1")
plt.legend()
plt.show()

def correlation_matrix(data, label, pd_label,text):
    #Matriz de correlación
    #correlation = train_c.loc[:, ['Pclass','Sex','Age','SibSp','Parch',
    # 'Fare','Embarked','Origin']]
    correlation = data.loc[:, ['Pclass','Age','SibSp','Parch',
     'Fare','Embarked','Ticket','Cabin','Origin','Title']]
    correlation = correlation.agg(LabelEncoder().fit_transform)
    correlation[label] = pd_label 
    correlation = correlation.set_index(label).reset_index() # Move Age at index 0.

    '''Now create the heatmap correlation.'''
    plt.figure(figsize = (10,7))
    sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
    plt.title('Variables Correlated with Origin '+text, fontsize = 18)
    plt.show()
#correlation_matrix(train_mf,'Origin', train_mf.Origin, "Male (Training)")
#correlation_matrix(test_mf,'Origin', test_mf.Origin, "Female (Test)")
X_train = train_mf.drop("Origin", axis=1)
Y_train = train_mf["Origin"]

X_test  = test_mf.drop("Origin", axis=1)
y_test = test_mf["Origin"]
X_train.shape, Y_train.shape, X_test.shape, y_test.shape
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
probs = random_forest.predict_proba(X_test)[:, 0] #prob por muestra P(train|x)
b_ratio(probs)

print("Acc: ",acc_random_forest)
p,r,f1=metricas(y_test,Y_pred)
print("Phi Coefficient: ",matthews_corrcoef(y_test, Y_pred))
print("ROC-AUC: ",roc_auc_score(y_test, Y_pred))
plot_curve(y_test,Y_pred)

print('---------------------------------------------------------')
X_train = train_mf.drop(["Origin","Sex"], axis=1)
Y_train = train_mf["Origin"]

X_test  = test_mf.drop(["Origin","Sex"], axis=1)
y_test = test_mf["Origin"]
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
probs = random_forest.predict_proba(X_test)[:, 0] #prob por muestra P(train|x)
b_ratio(probs)

print("Acc: ",acc_random_forest)
p,r,f1=metricas(y_test,Y_pred)
print("Phi Coefficient: ",matthews_corrcoef(y_test, Y_pred))
print("ROC-AUC: ",roc_auc_score(y_test, Y_pred))
plot_curve(y_test,Y_pred)
print('---------------------------------------------------------')
X_train = train_mf.drop(["Origin","Sex","Title"], axis=1)
Y_train = train_mf["Origin"]

X_test  = test_mf.drop(["Origin","Sex","Title"], axis=1)
y_test = test_mf["Origin"]
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
probs = random_forest.predict_proba(X_test)[:, 0] #prob por muestra P(train|x)
b_ratio(probs)

print("Acc: ",acc_random_forest)
p,r,f1=metricas(y_test,Y_pred)
print("Phi Coefficient: ",matthews_corrcoef(y_test, Y_pred))
print("ROC-AUC: ",roc_auc_score(y_test, Y_pred))
plot_curve(y_test,Y_pred)
print('---------------------------------------------------------')

