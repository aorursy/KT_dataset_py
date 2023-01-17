# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
df=data.copy()
display(df.head())
display(df.tail())
df.info()
df.isnull().sum()
df.Publisher.fillna('Unknown', inplace=True)
df.Year.fillna(df.Year.mode()[0], inplace=True) #we fill Year as mode of year
df.Year=df.Year.astype('int64')
df.info()
#Lets look at values or kategories of variables.
display(df.Platform.unique())
display(df.Genre.unique())
display(df.Year.unique())
df['Platform'].replace('2600', 'Atari', inplace=True)
display(df.Platform.unique())
df["Year"].value_counts()
df=df[df["Year"]<2017]
df.corr()
df.describe()
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# we can see correlation on heatmap.
f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
#Here,we can see with regression line 
sns.pairplot(df,kind="reg")
df1=df.head(100)
trace1 = go.Scatter(
                    x = df1.Rank,
                    y = df1.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),
                    text= df.Name)

trace2 = go.Scatter(
                    x = df1.Rank,
                    y = df1.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),
                    text= df1.Name)
trace3 = go.Scatter(
                    x = df1.Rank,
                    y = df1.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',size=8),
                    text= df.Name)
trace4 = go.Scatter(
                    x = df1.Rank,
                    y = df1.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'lime',size=8),
                    text= df.Name)
                    

data = [trace1, trace2,trace3,trace4]
layout = dict(title = 'North America, Europe, Japan and Other Sales of Top 100 Video Games',
              xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'Sales(In Millions)',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)
#BILGISAYAR KASTI
# plt.figure(figsize=(10,7))
# plt.title('Global Sales (in Millions) throughout the Year', color = 'g', size = 13)
# plt.xlabel('Year', color = "c");
# plt.ylabel('Global_Sales', color = "r");

# plt.bar(df.Year, df.Global_Sales, width=0.7, color="y", edgecolor="r", linewidth=2,
#        yerr=0.01, ecolor="m", hatch="*")
# plt.show()

sns.jointplot(df.Year,df.Global_Sales,size=8, ratio=9, color="blue")
plt.show()
# Scatter Plot 
df.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red')
plt.xlabel('NA_Sales')              # label = name of label
plt.ylabel('EU_Sales')
plt.title('NA_Sales EU_Sales Scatter Plot')            # title = title of plot
plt.show()
sns.barplot(x="Year",y = 'Global_Sales',data=df);
plt.xticks(rotation=30);
#Games according to Genre
labels=df.Genre.value_counts().index
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = df.Genre.value_counts().values
# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette('Set2'), autopct='%1.1f%%')
plt.title('Games According to Genre',fontsize = 17,color = 'green');
# Count of Platforms
plt.subplots(figsize = (15,8))
sns.countplot(df.Platform)
plt.title("Platform",color = 'blue',fontsize=20);
style.use('seaborn-poster')

f, ax = plt.subplots()
platform_releases = df['Platform'].value_counts()

sns.barplot(x=platform_releases.values, y=platform_releases.index, ec='Black')
ax.set_title('Platforms with the Most Releases', fontweight='bold', fontsize=23)
ax.set_xlabel('Releases', fontsize=18)
ax.set_xlim(0, max(platform_releases.values)+130)
ax.set_ylabel('Platform', fontsize=18)
plt.show()
genre_global_sales = df.groupby(['Genre'])['Global_Sales'].sum().sort_values(ascending=False)
print(genre_global_sales)
sns.barplot(x=genre_global_sales.index, y=genre_global_sales.values, ec='Black', palette='twilight')
plt.xticks(rotation=20, fontsize=12)
plt.xlabel('Genre', fontsize=18)
plt.ylabel('Global Sales (in Millions)', fontsize=18)
plt.title('Global Sales of Genres from 1980-2016', fontweight='bold', fontsize=22)
plt.tight_layout()
plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

stopwords = set(STOPWORDS)

for x in df.Genre.unique():
    wc = WordCloud(background_color="white", max_words=2000, 
                   stopwords=stopwords, max_font_size=40, random_state=42)
    wc.generate(df.Name[df.Genre == x].to_string())
    plt.imshow(wc)
    plt.title(x)
    plt.axis("off")
    plt.show()
fig,ax = plt.subplots(figsize=(8,5))
df['Genre'].value_counts(sort=False).plot(kind='bar',ax=ax,rot =90)
plt.title('Genre Distribution',fontsize=15)
plt.xlabel('Genre',fontsize=15)
plt.ylabel('Number of sales',fontsize=15)
from collections import Counter
genre = Counter(df['Genre'].dropna().tolist()).most_common(10)   #top 10 genre
genre_name = [name[0] for name in genre]
genre_counts = [name[1] for name in genre]

fig,ax = plt.subplots(figsize=(8,5))
sns.barplot(x=genre_name,y=genre_counts,ax=ax)
plt.title('Top ten Genre',fontsize=15)
plt.xlabel('Genre',fontsize=15)
plt.ylabel('Number of genre',fontsize=15)
ticks = plt.setp(ax.get_xticklabels(),fontsize=15,rotation=60)
platform = Counter(df['Platform'].dropna().tolist()).most_common(10)  #ten top platform
platform_name = [name[0] for name in platform]
platform_count = [name[1] for name in platform]

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x=platform_name,y=platform_count,ax=ax)
plt.title('Top ten platform',fontsize=15)
plt.ylabel('Number of platform',fontsize=15)
plt.xlabel('Platform',fontsize=15)
ticks = plt.setp(ax.get_xticklabels(),fontsize=15,rotation=60)
publisher = Counter(df['Publisher'].dropna().tolist()).most_common(10) #ten top publisher
publisher_name = [name[0] for name in publisher]
publisher_count = [name[1] for name in publisher]

fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(x=publisher_name,y=publisher_count,ax=ax)
plt.title('Top ten publisher',fontsize=15)
plt.ylabel('number of publisher',fontsize=15)
plt.xlabel('publisher',fontsize=15)
ticks = plt.setp(ax.get_xticklabels(),fontsize=15,rotation=90)
#preparing data frame
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
df1 = df.iloc[:100,:]

#import graph objects as "go"
import plotly.graph_objs as go

#creating trace1
trace1 = go.Scatter(
                    x = df1.Rank,
                    y = df1.NA_Sales,
                    mode = "lines",
                    name = "NA Sales",
                    marker = dict(color="rgba(166,11,2,0.8)"),
                    text = df1.Name)
#creating trace2
trace2 = go.Scatter(
                    x = df1.Rank,
                    y = df1.EU_Sales,
                    mode = "lines+markers",
                    name = "EU Sales",
                    marker = dict(color = "rgba(80,12,160,0.5)"),
                    text = df.Name)
data = [trace1,trace2]
layout = dict(title = "Global Sales of Top 100 Games",
                xaxis = dict(title="Rank",ticklen= 5, zeroline=False)
             )
fig = dict(data = data, layout = layout)
py.offline.iplot(fig)
x2016 = df.Genre[df.Year == 2016]      #Number of genres 2016 to 2010
x2006 = df.Genre[df.Year == 2010]

trace1 = go.Histogram(
                        x = x2016,
                        opacity = 0.75,
                        name = "2016",
                        marker = dict(color="rgba(162,50,70,0.9)"))
trace2 = go.Histogram(
                        x = x2006,
                        opacity = 0.75,
                        name = "2010",
                        marker = dict(color="rgba(24,68,200,0.6)"))

data = [trace1,trace2]
layout = go.Layout(barmode = "overlay",
                    title = "Number of Genres in 2016 and 2010 ",
                  xaxis = dict(title="Genre"),
                  yaxis = dict(title = "Count"),
                  )
fig = go.Figure(data=data,layout=layout)
py.offline.iplot(fig)
# Plot for games' count on each platform (classic bar plot)
x=df.Platform.unique()
y=df.Platform.value_counts()
plt.figure(figsize=(24,12))
sns.barplot(x=x,y=y,edgecolor="black")
plt.xlabel("Platforms")
plt.ylabel("Games count")
plt.show()
# Sequential bar plot for games' genre count
x=df.Genre.unique()
y=df.Genre.value_counts().sort_values(ascending=True)
plt.figure(figsize=(16,9))
sns.barplot(x=x,y=y,palette="rocket")
plt.xlabel("Genres")
plt.ylabel("Games count")
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
# label encoding of categorical variables
lbe = LabelEncoder()
df['Genre_Cat'] = lbe.fit_transform(df['Genre'])
df['Platform_Cat'] = lbe.fit_transform(df['Platform'])
df['Publisher_Cat'] = lbe.fit_transform(df['Publisher'])
df.sample(3)
df1 = df.loc[:,'Global_Sales':]
df1.head()
y = df1.Global_Sales.values
x_dat = df1.drop(['Global_Sales'], axis = 1)
x=(x_dat-np.min(x_dat))/(np.max(x_dat)-np.min(x_dat)).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
import statsmodels.api as sm  
x = sm.add_constant(x)

x=x.drop(["const"],axis=1)
x[0:5]
y[0:5]
lm = sm.OLS(y,x)
model = lm.fit()
model.summary()
model.params
model.summary().tables[1]
model.conf_int()  #confident intervaL
model.f_pvalue
print("f_pvalue: ", "%.4f" % model.f_pvalue)
print("fvalue: ", "%.2f" % model.fvalue)
print("tvalue: ", "%.2f" % model.tvalues[0:1])
model.rsquared_adj
model.fittedvalues[0:5]  #prediction values
y[0:5] #Real values
mse = mean_squared_error(y, model.fittedvalues)
mse
rmse = np.sqrt(mse)
rmse
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
lm = LinearRegression()
model_multi = lm.fit(x_train, y_train)
# Coefficients
model_multi.coef_
model_multi.intercept_
y_multipred=model_multi.predict(x_test)[0:10]
y_multipred
multi_reg_rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
multi_reg_rmse  # it is 1.4104046948894693
np.sqrt(mean_squared_error(y_train, model.predict(x_train))) #TRAIN ERROR IS 1.4104046948894693
np.sqrt(mean_squared_error(y_test, model.predict(x_test))) #TEST ERROR IS 2.0467221073976525
model_multi.score(x_train, y_train) #R2 VALUE IS 0.0008389037527712916
-cross_val_score(model_multi, x_train, y_train, cv = 10, scoring = "r2").mean()
multi_tuned_rmse_train=np.sqrt(-cross_val_score(model_multi, 
                x_train, 
                y_train, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
multi_tuned_rmse_train
multi_tuned_rmse_test=np.sqrt(-cross_val_score(model_multi, 
                x_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
multi_tuned_rmse_test #RMSE_MULTI_TUNED=1.759670817591785

r2_score(y_test, y_pred) #R2SCORE_MULTI_TUNED=0.0011145467577804435
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=42)
pca = PCA()
x_reduced_train = pca.fit_transform(scale(x_train))
x_reduced_train[0:1,:]
np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:2] #We can see succesful of choices
lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train, y_train)
pcr_model.intercept_
pcr_model.coef_
y_pred = pcr_model.predict(x_reduced_train)
y_pred[0:5]
rmse_pcr=np.sqrt(mean_squared_error(y_train, y_pred))
rmse_pcr
df["Global_Sales"].mean()
r2score_pcr=r2_score(y_train, y_pred)
r2score_pcr
pca2 = PCA()
x_reduced_test = pca2.fit_transform(scale(x_test))
y_pred = pcr_model.predict(x_reduced_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train[:,0:10], y_train)  #burdaki 0:10 bilesen sayisi degistikce hata da degisiyor
y_pred = pcr_model.predict(x_reduced_test[:,0:10])
print(np.sqrt(mean_squared_error(y_test, y_pred)))
from sklearn import model_selection  #cross validation icin uygun bilesen saysisini bulma islemi
cv_10 = model_selection.KFold(n_splits = 10,  #10 katli cross validation
                             shuffle = True,  #gruplarin karistirilip karistirilmayacagi
                             random_state = 1)
cv_10
lm = LinearRegression()
RMSE = []
for i in np.arange(1, x_reduced_train.shape[1] + 1):
    
    score = np.sqrt(-1*model_selection.cross_val_score(lm, 
                                                       x_reduced_train[:,:i], 
                                                       y_train.ravel(), 
                                                       cv=cv_10, 
                                                       scoring='neg_mean_squared_error').mean())
    RMSE.append(score)
#Her bir bilesen sayisi icin 10 katli cross validation uygulayip hatalara bakip hangi bilesen sayisinda daha az hata var onu buluyoruz
plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Prediction for PCR Model Tuning');

lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train[:,0:2], y_train)  #en dusuk hata bilesen sayisi 2 oldugunda cikti
y_pred = pcr_model.predict(x_reduced_train[:,0:2])
print(np.sqrt(mean_squared_error(y_train, y_pred)))
y_pred = pcr_model.predict(x_reduced_test[:,0:2])
print(np.sqrt(mean_squared_error(y_test, y_pred))) #rmse_pcr_tuned=2.0488329468861872
r2_score(y_test, y_pred) #R2SCORE_PCR_TUNED=0.0011145467577804435
from sklearn.cross_decomposition import PLSRegression, PLSSVD
pls_model = PLSRegression().fit(x_train, y_train)
pls_model.coef_  #bunlar her zaman degisken sayisi kadar cikar zaten ,biz bunlari carpistirip indirgeme yapiyoruz PCR da da ayni
x_train.head()
pls_model.predict(x_train)[0:10]
y_pred = pls_model.predict(x_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # RMSE_PLS_TRAIN=1.4037266722286532
r2_score(y_train, y_pred)  #R2SCORE_PLS=0.0008389010941547426
y_pred = pls_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_PLS_TEST=2.048013316263683
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
RMSE = []
for i in np.arange(1, x_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, x_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, x_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Global_Sales');
pls_model = PLSRegression(n_components = 3).fit(x_train, y_train) #3 IS BETER FOR N_COMPONENTS
y_pred = pls_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_PLS_TUNED=2.048013783230342
r2_score(y_test, y_pred) #R2SCORE__PLS_TUNED=0.0019131351293203425
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 0.1).fit(x_train, y_train)  #alfa dedigimiz sey lambda.
ridge_model
ridge_model.coef_
10**np.linspace(10,-2,100)*0.5
lambdalar = 10**np.linspace(10,-2,100)*0.5 

ridge_model = Ridge()
katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(x_train, y_train) 
    katsayilar.append(ridge_model.coef_) 
    

    
ax = plt.gca()
ax.plot(lambdalar, katsayilar) 
ax.set_xscale('log') 

plt.xlabel('Lambda(Alpha) Values')
plt.ylabel('Coefficients / weights')
plt.title('Ridge Coefficients as a Function of Regularization');
y_pred = ridge_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_RIDGE=2.04801379183374
#R2SCORE_RIDGE= 0.001913126743694149
r2_score(y_test, y_pred)
lambdalar = 10**np.linspace(10,-2,100)*0.5 
lambdalar[0:5]
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas = lambdalar, 
                   scoring = "neg_mean_squared_error",
                   normalize = True)
ridge_cv.fit(x_train, y_train)
ridge_cv.alpha_
#RMSE_RIDGE_TUNED=2.048436910707652
ridge_tuned = Ridge(alpha = ridge_cv.alpha_, 
                   normalize = True).fit(x_train,y_train)
np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(x_test)))  #icindeki y_pred degeri,ayni sey( ridge_tuned.predict(x_test)=y_pred)
r2_score(y_test, ridge_tuned.predict(x_test)) #R2SCORE_RIDGE_TUNED=0.0015006754017719004
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 0.1).fit(x_train, y_train)
lasso_model
lasso_model.coef_
lasso = Lasso()
lambdalar = 10**np.linspace(10,-2,100)*0.5 
katsayilar = []

for i in lambdalar:
    lasso.set_params(alpha=i)
    lasso.fit(x_train, y_train)
    katsayilar.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(lambdalar*2, katsayilar)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
lasso_model.predict(x_test)
y_pred = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_LASSO=2.050443578753031
#R2SCORE_LASSO=-0.00045656143377237335
r2_score(y_test, y_pred)
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas = None, 
                         cv = 10, 
                         max_iter = 10000, 
                         normalize = True)
lasso_cv_model.fit(x_train,y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(x_train, y_train)
y_pred = lasso_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_LASSO_TUNED=2.04801384684936
r2_score(y_test, y_pred) #R2_LASSO_TUNED=0.0019130731206485896
from sklearn.linear_model import ElasticNet
enet_model = ElasticNet().fit(x_train, y_train)
enet_model
enet_model.coef_
enet_model.intercept_
enet_model.predict(x_test)
y_pred = enet_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE_ELASTICNET=2.050443578753031
r2_score(y_test, y_pred) # R2SCORE_ELSTICNET=-0.00045656143377237335
from sklearn.linear_model import ElasticNetCV
enet_cv_model = ElasticNetCV(cv = 10, random_state = 0).fit(x_train, y_train)
enet_cv_model
enet_cv_model.alpha_
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(x_train,y_train)
y_pred = enet_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE_ELASTICNET_TUNED=2.0480524706030367
# R2SCORE_ELASTICNET_TUNED=0.001875426669009972
r2_score(y_test, y_pred)
