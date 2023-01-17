!pip install pandas==0.23.4

!pip install seaborn==0.9.0

#!kaggle datasets download -d edalrami/19000-spotify-songs
import os

import pandas as pd

import numpy as np

import seaborn as sns; sns.set_style("whitegrid")

from matplotlib import pyplot as plt



import warnings

warnings.filterwarnings('ignore')





sns.__version__
import os



print(os.chdir("../input"))

df_data=pd.read_csv("song_data.csv",sep=",",engine="python")



df_info=pd.read_csv("song_info.csv",sep=",",engine="python")
df_data.head(5)
df_data.info()
df_data.describe().T
#Well behaved dataset - no NA





df_data.isna().sum()
#Some unnecessary repetitions observed



df_data.sort_values("song_popularity",ascending=False).head(10)
#Original database shape

df_data.shape
#After duplicate elimination dataframe shape

df_data[df_data.duplicated(subset="song_popularity",keep="first")].shape
df_data=df_data[df_data.duplicated(subset="song_popularity",keep="first")].reset_index(drop="index")



df_data.head(3)
#2 different types of variables - categoricals (param_cat) and numerical(param)



param_cat=["key","audio_mode","time_signature"]



param=["song_duration_ms",

"acousticness",

"danceability",

"energy",

"instrumentalness",

"liveness",

"loudness",

"speechiness",

"tempo",

"audio_valence"]
#Distribuição geral dos dados



for feature in param:

  g = sns.FacetGrid(df_data,col="audio_mode") 

  g.map(sns.distplot, feature) 

  

plt.plot()
#Mapa de correlação com matrix sem tratamento de outliers



plt.figure(figsize=(9, 6))  # Aumenta o tamanho da figura

ax=sns.heatmap(

    pd.concat([df_data[param],df_data["song_popularity"]],axis=1).corr(),

    vmin=-1, vmax=1, annot=True, fmt='.2f')

plt.show()
# Scatter plot e regressão com matrix sem tratamento de outliers para variáveis categoricas



#Nota-se que time signature possui indicação de correlação positiva com popularidade do som - última linha



plt.figure(figsize=(9, 6))

g = sns.pairplot(

    pd.concat([df_data[param_cat],df_data["song_popularity"]],axis=1).sample(n=3000),

    kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.05}}

)

# Scatter plot e regressão com matrix sem tratamento de outliers para variáveis numéricas



#Nota-se que dançabilidade e barulheira possui indicação de correlação positiva com popularidade do som - última linha



plt.figure(figsize=(9, 6))

g = sns.pairplot(

    pd.concat([df_data[param],df_data["song_popularity"]],axis=1).sample(n=3000),

    kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.03}}

)

df_info.head(2)
#Word cloud



# fig, ax = plt.subplots(figsize = (16, 12))

# plt.subplot(1, 2, 1)

# text_cat = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Name'].fillna('').values)



# wordcloud = WordCloud(max_font_size=None, background_color='white',

#                       width=1200, height=1000).generate(text_cat)

# plt.imshow(wordcloud)

# plt.title('Top cat names')

# plt.axis("off")



# plt.subplot(1, 2, 2)

# text_dog = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Name'].fillna('').values)

# wordcloud = WordCloud(max_font_size=None, background_color='white',

#                       width=1200, height=1000).generate(text_dog)

# plt.imshow(wordcloud)

# plt.title('Top dog names')

# plt.axis("off")



# plt.show()
#Boxplot univariado com as variáveis numéricas



for position,feature in enumerate(param):  

  plt.figure(figsize=(3, 3))

  plt.figure(position)  

  plt.title(feature)

  sns.boxplot(data=df_data,y=feature)

  plt.plot()

  



  

#plt.tight_layout()

#plt.show()

            

#y="song_duration_ms",hue="song_popularity") 
#Corte dos valores que estão fora do invervalo de confiança 66%,95% e 99% - sendo que será utilizado o intervalo de 99% para este exercício



df_data_66=df_data[df_data[param].apply(lambda x: np.abs(x - x.mean()) / x.std() < 1).all(axis=1)]

df_data_95=df_data[df_data[param].apply(lambda x: np.abs(x - x.mean()) / x.std() < 2).all(axis=1)]

df_data_99=df_data[df_data[param].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
#Análise de boxplot com uso de confidence interval of 99%



for position,feature in enumerate(param):  

  plt.figure(figsize=(3, 3))

  plt.figure(position)  

  plt.title(feature)

  sns.boxplot(data=df_data_95,y=feature)

  plt.plot()
#Mesmo empregando intervalos diferentes, nota-se que que as correlações positivas e negativas mais fortes permanecem



plt.figure(figsize=(12, 9))



plt.subplot(3,1,1)



plt.title("Original matrix")



#plt.figure(figsize=(9, 6))  # matrix original

ax=sns.heatmap(

    pd.concat([df_data[param],df_data["song_popularity"]],axis=1).corr(),

    vmin=-1, vmax=1, annot=True, fmt='.2f')



plt.title("66 % matrix")



plt.subplot(3,1,2)

#matrix 66

ax=sns.heatmap(

    pd.concat([df_data_66[param],df_data_66["song_popularity"]],axis=1).corr(),

    vmin=-1, vmax=1, annot=True, fmt='.2f')





plt.title("99% matrix")

plt.subplot(3,1,3)

#matrix 99

#plt.figure(figsize=(9, 6))  # Aumenta o tamanho da figura

ax=sns.heatmap(

    pd.concat([df_data_99[param],df_data_99["song_popularity"]],axis=1).corr(),

    vmin=-1, vmax=1, annot=True, fmt='.2f')



plt.show()


#Histograma de corte para análise - top 80 até 100 versus 0 até 20



plt.figure(figsize=(5, 3))



df_data_99["song_popularity"].hist()



plt.axvline(x=20,color="red")



plt.axvline(x=80,color="red")
#Corte da base de dados considerando apenas TOP E HIT

df_data_versus=df_data_99.loc[(df_data_99["song_popularity"]>=80) | (df_data_99["song_popularity"]<=20)].reset_index(drop="index")
#Criação de categoria TOP E HIT

df_data_versus["top"] = df_data_versus["song_popularity"].apply(lambda x : "Hit" if x >=80 else "Flop")



df_data_versus.head(5)
#Scatter plot com regressão entre as duas categorias para diversas variáveis 



#Nota-se que quase todas as variáveis apresentaram Slope diferente de zero - seja positivo ou negativo, mostrando possíveis diferenças de valor 



plt.figure(figsize=(9, 6))

for feature in param:

  #sns.factorplot(data=df_data_versus,x="song_popularity",y=feature,hue="top",kind="strip")

  sns.lmplot(data=df_data_versus,x="song_popularity",y=feature,hue="top",fit_reg=True, markers=["o", "x"])

  

  #sns.(data=df_data_versus,x="song_popularity",y=feature,hue="top")
#Teste estatístico de média T-test para todas as variáveis numéricas - 

#todas elas apresentaram diferença de média estatísticamente diferente (99%) entre estas duas categorias

#Assim, há indicação de que essas 2 categorias possuem diferenças quanto a todos os atributos utilizados



from scipy import stats





for feature in param:



  rvs1=df_data_versus.loc[(df_data_versus["top"]=="Hit"),feature]



  rvs2=df_data_versus.loc[(df_data_versus["top"]=="Flop"),feature]



  t,p=stats.ttest_ind(rvs1,rvs2, equal_var = False)

  

  print("The T-test between Hit and Flop using mean value of {} is {:03.3f} and p-value of {:03.3f}".format(feature,t,p))
df_data_99.head(2)
#dummy transformation das variáveis categoricas



df_data_99 = pd.get_dummies(data=df_data_99,columns=["key","audio_mode","time_signature"])
# Padronização dos valores numéricos via standard scale



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



for column in pd.concat([df_data_99["song_duration_ms"],df_data_99.select_dtypes("float64")],axis=1).columns:

  

  scaler.fit(df_data_99[column].values.reshape(-1, 1))

  

  df_data_99[column]=scaler.transform(df_data_99[column].values.reshape(-1, 1))
#Cópia dos dados em 2 outras bases diferentes - uma utilizada para regressao e outra para classificacao de song_popularity



df_data_99_class=df_data_99.copy()



df_data_99_regress=df_data_99.copy()
#Classificação da popularidade em 4 etiquetas sendo A - melhor e D - pior

df_data_99_class["song_popularity"] = df_data_99_class["song_popularity"].apply(lambda x : "A" if x>=75 else ("B" if x>=50 and x<=75 else ("C" if x>=25 and x<=50 else "D")))
#X and y selection



y=df_data_99_class.song_popularity



X=df_data_99_class.iloc[:,2:]
#split



from sklearn.model_selection import train_test_split, GridSearchCV



train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=0.2, random_state=1)
#Modelo possui precisão média de 39 mas ainda há bastante espaço 

#para melhora do modelo



#classifier Naives



from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.metrics import classification_report





M_mult=MultinomialNB()



#M_mult.get_params



tuned_parameters = {

    'alpha': [1, 1e-1, 1e-2]

}



clf=GridSearchCV(M_mult,tuned_parameters,cv=10)



clf.fit(abs(train_X),train_y)



print(classification_report(test_y, clf.predict(test_X), digits=4))
#classifier KNeighborsClassifier

#Apresentou performance um pouco acima da modelo multinomial de Naives utilizando 2-3 clusters de análise



from sklearn.neighbors import KNeighborsClassifier



kmeans=KNeighborsClassifier()



test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(train_X,train_y)

    

    train_scores.append(knn.score(train_X,train_y))

    test_scores.append(knn.score(test_X,test_y))



plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
df_data_99_regress.head(2)
#X and y usando base de regressão



y=df_data_99_regress.song_popularity



X=df_data_99_regress.iloc[:,2:]
#Split de dados

from sklearn.model_selection import train_test_split, GridSearchCV



train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=0.2, random_state=1)
#Funcao para erro medio ao quadrado para funcoes linear normalizados como Ridge e Lasso



from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
#Otimização de parâmetros



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75,100,200,300,400,500,600]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
#O valor 300 de alpha é o ponto ótimo para diminuição do erro da função mas ainda assim não há grandes ganhos

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Ridge")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
#Uso de modelo Lasso



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_X, train_y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = train_X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
#Variáveis mais importantes e as menos importantes para explicar popularidade



plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#Deep learning



from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
model = Sequential()



model.add(Dense(256, activation="relu", input_dim = train_X.shape[1]))

model.add(Dense(1, input_dim = train_X.shape[1], W_regularizer=l1(0.001)))



model.compile(loss = "mse", optimizer = "adam")
model.summary()
hist = model.fit(train_X, train_y, validation_data = (test_X, test_y),epochs=50)
#Valores de previsão

pd.Series(model.predict(test_X)[:,0]).hist()