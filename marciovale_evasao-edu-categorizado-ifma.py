from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#plt.style.use('ggplot')
#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
#Carregando nosso dataset
#evasao_data = pd.read_csv('../input/baseeducategorizacaosocial2klimredux/BaseEduCatRedux_Limit.csv')
evasao_data = pd.read_csv('../input/base-edu-cat-social-1-para-1b/BaseEduCat1p1.csv')

#Mostrando as 5 primeiras linhas de nosso dataset
evasao_data.head()
## mostra informações como tipo de dados, colunas, valores null encontrados, memoria usada etc
## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
evasao_data.info(verbose=True)

## estatísticas básicas detalhadas sobre o dado (observe que apenas colunas numéricas seriam exibidas aqui, a menos que o parâmetro include="all")
## for reference: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
evasao_data.describe()

## Veja também :
## para retornar colunas de um tipo específico: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
evasao_data.describe().T
evasao_data_copy = evasao_data.copy(deep = True)
##diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
evasao_data_copy[['ira_novo','renda_per_capita']] = evasao_data_copy[['ira_novo','renda_per_capita']].replace(0,np.NaN)

## showing the count of Nans
print(evasao_data_copy.isnull().sum())
#### Para preencher esses valores Nan, a distribuição de dados precisa ser entendida
evasao_data_copy.head()
evasao_data_copy.describe().T
p = evasao_data.hist(figsize = (20,20))
## diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

evasao_data_copy['ira_novo'].fillna(evasao_data_copy['ira_novo'].median(), inplace = True)
evasao_data_copy['renda_per_capita'].fillna(evasao_data_copy['renda_per_capita'].median(), inplace = True)


p = evasao_data_copy.hist(figsize = (20,20))
## observing the shape of the data
evasao_data_copy.shape
## data type analysis
#plt.figure(figsize=(5,5))
#sns.set(font_scale=2)
sns.countplot(y=evasao_data.dtypes ,data=evasao_data)
plt.xlabel("contagem de cada tipo")
plt.ylabel("tipos de dados")
plt.show()
## null count analysis
import missingno as msno
p=msno.bar(evasao_data_copy)

## checando o balanceamento dos dados para plotar a contagem de resultados por seus valores
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = evasao_data_copy["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(evasao_data_copy.Outcome.value_counts())
p=evasao_data_copy.Outcome.value_counts().plot(kind="bar")

from pandas.tools.plotting import scatter_matrix
p=scatter_matrix(evasao_data,figsize=(25, 25))
p=sns.pairplot(evasao_data_copy, hue = 'Outcome')
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(evasao_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(evasao_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
##X =  pd.DataFrame(sc_X.fit_transform(evasao_data_copy.drop(["Outcome"],axis = 1),),
##        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
##       'BMI', 'DiabetesPedigreeFunction', 'Age'])
##        columns=['ira_novo', 'ativo', 'curso_campus_id', 'turno_id', 'situacao_id', 
##                 'periodo_atual', 'periodo_letivo', 'ano_letivo_id', 'ocorrencias', 
##                 'raca_id', 'estado_civil_id', 'qtd_filhos', 'pai_nivel_escolaridade_id', 
##                 'mae_nivel_escolaridade_id', 'possui_necessidade_especial', 
##                 'ficou_tempo_sem_estudar', 'qtd_pessoas_domicilio_antes_ifma', 
##                 'tipo_moradia_id', 'renda_per_capita', 'tipo_imovel_residencial_id', 
##                 'tipo_area_residencial_id', 'possui_conhecimento_idiomas', 'possui_conhecimento_informatica'])


X =  pd.DataFrame(sc_X.fit_transform(evasao_data_copy.drop(["Outcome"],axis = 1),),
        columns=['ira_novo', 'ativo', 'curso_campus_id', 'turno_id', 'situacao_id', 
                 'periodo_atual', 'periodo_letivo', 'ano_letivo_id', 'ocorrencias', 
                 'raca_id', 'estado_civil_id', 'qtd_filhos', 'pai_nivel_escolaridade_id', 
                 'mae_nivel_escolaridade_id', 'possui_necessidade_especial', 
                 'ficou_tempo_sem_estudar', 'qtd_pessoas_domicilio_antes_ifma', 
                 'tipo_moradia_id', 'renda_per_capita', 'tipo_imovel_residencial_id', 
                 'tipo_area_residencial_id', 'possui_conhecimento_idiomas', 
                 'possui_conhecimento_informatica'])
X.head()
#X = diabetes_data.drop("Outcome",axis = 1)
y = evasao_data_copy.Outcome
#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(1)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
## trying to plot decision boundary 
value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2, 
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value, 
                                             8: value, 9: value, 10: value, 11: value, 12: value, 13: value, 
                                             14: value, 15: value, 16: value, 17: value, 18: value, 19: value, 
                                             20: value, 21: value, 22: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width, 
                                             8: width, 9: width, 10: width, 11: width, 12: width, 13: width, 
                                             14: width, 15: width, 16: width, 17: width, 18: width, 19: width, 
                                             20: width, 21: width, 22: width},
                      X_highlight=X_test.values)

# Adding axes annotations
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
plt.title('KNN com Dados de Evasão')
plt.show()
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Matriz de Confusão', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=1) ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
