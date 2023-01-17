# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as smf

import statsmodels.api as sm



from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 14, 6   # set default figure size



# adjust the Jupyter Notebook window width for better presentation

# you don't need to understand this. It's just handy to use. 

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
df=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

df.head()

df.dtypes
df=df.drop(columns='Serial No.')

df.head()
df=df.rename(columns={"GRE Score":"GRE_Score","TOEFL Score":"TOEFL_Score","University Rating":"Univer_Rating","SOP":"Statement_Of_Purpose","LOR ":"Letter_Of_Recom","Research":"Research_Experience",

"Chance of Admit ":"Chance_of_Admit"})

df.head()
df.GRE_Score= df.GRE_Score.astype(float)

df.TOEFL_Score= df.TOEFL_Score.astype(float)

df.Univer_Rating= df.Univer_Rating.astype(float)

df.Research_Experience= df.Research_Experience.astype(float)

df.dtypes
df.head()
cor=df.corr(method = 'pearson').round(3)

cor
fig=plt.figure(figsize=(20,7))

sns.heatmap(cor,cmap = plt.cm.YlGnBu,annot = True,linecolor='white')

plt.show()
fig=plt.figure(figsize=(15,7))

fig = sns.regplot(x="GRE_Score", y="CGPA", data=df)

plt.title("GRE Score vs CGPA")

plt.show()
plt.figure(figsize=(15,7))

sns.countplot(df.GRE_Score,hue=df.Research_Experience)

plt.title('GREScore for Hue Research Show')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

sns.countplot(df.TOEFL_Score,hue=df.Research_Experience)

plt.title('TOEFL for Hue Research Show')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x='Univer_Rating',y='GRE_Score',data=df)

plt.xlabel('University Rating')

plt.ylabel('GRE Score')

plt.title('University Rating Vs GREScore')

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(x=df.GRE_Score.value_counts().index,y=df.GRE_Score.value_counts().values)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

sns.scatterplot(y="CGPA", x="GRE_Score",hue="Univer_Rating",data=df,s =100)

plt.show()
plt.figure(figsize=(15,7))

sns.scatterplot(y="CGPA", x="GRE_Score",hue="Research_Experience",data=df,s =100)

plt.show()


df.dropna(inplace=True)

model_f ='Chance_of_Admit ~ GRE_Score + TOEFL_Score + Univer_Rating + Statement_Of_Purpose + Letter_Of_Recom+CGPA+Research_Experience+GRE_Score*TOEFL_Score+TOEFL_Score*Univer_Rating+Statement_Of_Purpose*Univer_Rating +Statement_Of_Purpose*Letter_Of_Recom+ Letter_Of_Recom*CGPA+ CGPA*Research_Experience+ GRE_Score*Univer_Rating+GRE_Score*Statement_Of_Purpose+GRE_Score*Letter_Of_Recom+GRE_Score*CGPA+GRE_Score*Research_Experience+TOEFL_Score*Statement_Of_Purpose+TOEFL_Score*Letter_Of_Recom+TOEFL_Score*CGPA+TOEFL_Score*Research_Experience+Univer_Rating*Letter_Of_Recom+Univer_Rating*CGPA+Univer_Rating*Research_Experience+Letter_Of_Recom*Research_Experience'

result1=smf.ols(formula=model_f,data= df).fit()

print(result1.summary())
n=len('CGPA ')

MSE3=result1.ssr/n

MSE3=np.round(MSE3,3)

R23= result1.rsquared

R23=np.round(R23,3)

R2adj3=result1.rsquared_adj

R2adj3=np.round(R2adj3,4)

AIC3= result1.aic

AIC3=np.round(AIC3,3)

BIC3= result1.bic

BIC3=np.round(BIC3,3)

print ("MSE:  ", MSE3)

print( 'R2:   ', R23)

print( 'R2adj:',R2adj3)

print( 'AIC:  ',AIC3)

print( 'BIC:  ',BIC3)
df.dropna(inplace=True)

model_f ='Chance_of_Admit~ GRE_Score + TOEFL_Score + Univer_Rating + Statement_Of_Purpose + Letter_Of_Recom+CGPA+Research_Experience'

result2=smf.ols(formula=model_f,data= df).fit()

print(result2.summary())
n=len('CGPA ')

MSE3=result2.ssr/n

MSE3=np.round(MSE3,3)

R23= result2.rsquared

R23=np.round(R23,3)

R2adj3=result2.rsquared_adj

R2adj3=np.round(R2adj3,4)

AIC3= result2.aic

AIC3=np.round(AIC3,3)

BIC3= result2.bic

BIC3=np.round(BIC3,3)

print ("MSE:  ", MSE3)

print( 'R2:   ', R23)

print( 'R2adj:',R2adj3)

print( 'AIC:  ',AIC3)

print( 'BIC:  ',BIC3)
df.dropna(inplace=True)

model_f ='Chance_of_Admit~ GRE_Score + TOEFL_Score + Letter_Of_Recom+Univer_Rating+CGPA+Research_Experience'

result3=smf.ols(formula=model_f,data= df).fit()

print(result3.summary())
n=len('CGPA ')

MSE3=result3.ssr/n

MSE3=np.round(MSE3,3)

R23= result3.rsquared

R23=np.round(R23,3)

R2adj3=result3.rsquared_adj

R2adj3=np.round(R2adj3,4)

AIC3= result3.aic

AIC3=np.round(AIC3,3)

BIC3= result3.bic

BIC3=np.round(BIC3,3)

print ("MSE:  ", MSE3)

print( 'R2:   ', R23)

print( 'R2adj:',R2adj3)

print( 'AIC:  ',AIC3)

print( 'BIC:  ',BIC3)
df.dropna(inplace=True)

model_f ='Chance_of_Admit~ GRE_Score + TOEFL_Score + Letter_Of_Recom+Statement_Of_Purpose+CGPA+Research_Experience'

result4=smf.ols(formula=model_f,data= df).fit()

print(result4.summary())
n=len('CGPA ')

MSE3=result4.ssr/n

MSE3=np.round(MSE3,3)

R23= result4.rsquared

R23=np.round(R23,3)

R2adj3=result4.rsquared_adj

R2adj3=np.round(R2adj3,4)

AIC3= result4.aic

AIC3=np.round(AIC3,3)

BIC3= result4.bic

BIC3=np.round(BIC3,3)

print ("MSE:  ", MSE3)

print( 'R2:   ', R23)

print( 'R2adj:',R2adj3)

print( 'AIC:  ',AIC3)

print( 'BIC:  ',BIC3)
df.dropna(inplace=True)

model_f ='Chance_of_Admit~ GRE_Score + TOEFL_Score + Letter_Of_Recom+CGPA+Research_Experience'

result5=smf.ols(formula=model_f,data= df).fit()

print(result5.summary())
n=len('CGPA ')

MSE3=result5.ssr/n

MSE3=np.round(MSE3,3)

R23= result5.rsquared

R23=np.round(R23,3)

R2adj3=result5.rsquared_adj

R2adj3=np.round(R2adj3,4)

AIC3= result5.aic

AIC3=np.round(AIC3,3)

BIC3= result5.bic

BIC3=np.round(BIC3,3)

print ("MSE:  ", MSE3)

print( 'R2:   ', R23)

print( 'R2adj:',R2adj3)

print( 'AIC:  ',AIC3)

print( 'BIC:  ',BIC3)
X_ads=df[['GRE_Score','TOEFL_Score','Univer_Rating','Statement_Of_Purpose','Letter_Of_Recom','CGPA','Research_Experience']]

X_ads
Y_ads=df.Chance_of_Admit

Y_ads
from sklearn.model_selection import train_test_split

ex_train,ex_test,ey_train,ey_test=train_test_split(X_ads,Y_ads,test_size=0.3,random_state=1)

print(ex_train,ex_test,ey_train,ey_test)

from sklearn import linear_model,metrics

lr=linear_model.LinearRegression()

lr1=lr.fit(ex_train,ey_train)





print('coefficient',lr1.coef_)

print('lr1.intercept',lr1.intercept_)

print('r21_score', metrics.r2_score(ey_test,lr1.predict(ex_test)))

print('MSE1',metrics.mean_squared_error(ey_test,lr1.predict(ex_test)))

X_ads=df[['GRE_Score','TOEFL_Score','Univer_Rating','Letter_Of_Recom','CGPA','Research_Experience']]



lr=linear_model.LinearRegression()

lr2=lr.fit(ex_train,ey_train)

print('coefficient',lr2.coef_)

print('lr2.intercept',lr2.intercept_)

print('r22_score', metrics.r2_score(ey_test,lr2.predict(ex_test)))

print('MSE2',metrics.mean_squared_error(ey_test,lr2.predict(ex_test)))

X_ads=df[['GRE_Score','TOEFL_Score','Univer_Rating','Letter_Of_Recom','CGPA','Research_Experience']]

lr=linear_model.LinearRegression()

lr3=lr.fit(ex_train,ey_train)
print('coefficient',lr3.coef_)

print('lr3.intercept',lr3.intercept_)

print('r23_score', metrics.r2_score(ey_test,lr3.predict(ex_test)))

print('MSE3',metrics.mean_squared_error(ey_test,lr3.predict(ex_test)))

X_ads=df[['GRE_Score','TOEFL_Score','Letter_Of_Recom','CGPA','Research_Experience']]
lr=linear_model.LinearRegression()

lr4=lr.fit(ex_train,ey_train)
print('coefficient',lr4.coef_)

print('lr4.intercept',lr4.intercept_)

print('r24_score', metrics.r2_score(ey_test,lr4.predict(ex_test)))

print('MSE4',metrics.mean_squared_error(ey_test,lr4.predict(ex_test)))


from sklearn.preprocessing import StandardScaler

df_scaled = df.copy()

scaler = StandardScaler()

columns =df.columns[0:8]

df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

df_scaled.head()
plt.figure(figsize=(15,7))

plt.scatter(x='GRE_Score',y='CGPA',data= df_scaled, c=df.Univer_Rating,s= 100,alpha=.3)

plt.show()
from sklearn.cluster import KMeans

plt.figure(figsize=(15,8))

kmeans=KMeans(n_clusters=3,init='random',n_init=1,random_state=8)

kmeans.fit(df_scaled)

labels= kmeans.labels_

centroids=kmeans.cluster_centers_#cluster centroids 

plt.scatter(centroids[:,0],centroids[:,1],c='k',s=200,marker='*')

plt.title('max_iter='+str(kmeans.n_iter_)+'\n'+'inertia='+str(kmeans.inertia_.round(3)),fontsize=14)# title

plt.scatter(x='GRE_Score',y='CGPA',data= df_scaled,c=labels, s= 100,alpha=.2)

plt.show()
from sklearn.cluster import KMeans

inertia=[]

for k in range(1,15):

    kmeans = KMeans(n_clusters=k).fit(df_scaled)

    inertia.append([k,kmeans.inertia_])# attach to bottom of inertia

inertia

plt.plot(pd.DataFrame(inertia)[0],pd.DataFrame(inertia)[1],marker='o')
from sklearn.metrics import silhouette_score

silhouette=[]

for k in range(2,15):# smallest is 2

    kmeans = KMeans(n_clusters=k).fit(df_scaled)

    silhouette.append([k,silhouette_score(df_scaled,kmeans.labels_)])# cluster solutions

plt.plot(pd.DataFrame(silhouette)[0],pd.DataFrame(silhouette)[1],marker='o')
plt.figure(figsize=(15,8))

plt.plot(np.transpose(kmeans.cluster_centers_),marker='o')

plt.xticks(np.arange(3),columns[0:],rotation = 16,fontsize=14)

plt.title('Profiles',fontsize=18)

plt.ylabel('Standardized Feature Value',fontsize=16)

plt.xlabel('Features',fontsize=16)

plt.show()
df_centroids=pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),columns=columns[:])

df_centroids.round({'GRE_Score':1,'TOEFL_Score':0,'Univer_Rating':0,'Statement_Of_Purpose':1, 'Letter_Of_Recom':1,'CGPA':2,'Research_Experience': 0,'Chance_of_Admit ': 2})