import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="darkgrid")
sns.axes_style("darkgrid")
import plotly.express as px
from scipy.stats.mstats import winsorize
from scipy.stats import stats
from scipy.stats import zscore
from scipy.stats import jarque_bera
from scipy.stats import normaltest
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale 

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
sns.set(style="whitegrid")

title_font= {"family": "arial", "weight": "bold", "color": "darkred", "size": 15}
label_font= {"family": "arial", "weight": "bold", "color": "darkblue", "size": 10}

df_life= pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")
df_all= pd.read_csv("../input/country-mapping-iso-continent-region/continents2.csv")
df_all["Country"]= df_all["name"] 
del df_all["name"]
df_all.head()
df_all.info()
merged= pd.DataFrame()
merged= pd.merge(df_life, df_all, on="Country",  how= "left")
merged.head()
usefull_col= ["alpha-3", "sub-region"]
for i in merged.columns:
    if i not in df_life.columns | usefull_col:
        del merged[i]
merged.head()
merged.info()
df= merged.copy()
df.head().T
cols= df.columns.tolist()
cols = cols[-2:]+ cols[:-2]
df= df[cols]
df.head()
df.info()  
df.columns= ['alpha_3', 'sub_region', 'country', 'year', 'status',
       'life_expectancy', 'adult_mortality', 'infant_deaths', 'alcohol',
       'percentage_expenditure', 'hepatitis_b', 'measles', 'bmı',
       'under-five deaths ', 'Polio', 'total_expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'gdp', 'population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'schooling']
df["country"]= df["country"].astype("category")
df["status"]= df["status"].astype("category")
df["alpha_3"]= df["alpha_3"].astype("category")
df["sub_region"]= df["sub_region"].astype("category")
df.nunique()
df.info() # Size decreased from 573 to 513 kb. 
df.isnull().sum()*100/df.shape[0]
df.dropna(subset=["alpha_3", "sub_region"], inplace=True)
df.isnull().sum()*100/df.shape[0]
df[(df["schooling"]==0)] # There are some zero values, we wil transform them to null.
for i in df.columns:
    df[i].replace(0, np.nan, inplace=True)
df[(df["schooling"]==0)]
df["alcohol"]= df.groupby("country")["alcohol"].transform(lambda x: x.interpolate(limit_direction="both",
                                                                                  method= "linear"))
df["alcohol"]= df.groupby("sub_region")["alcohol"].transform(lambda x: x.fillna(x.mean()))

df["schooling"]= df.groupby("sub_region")["schooling"].transform(lambda x: x.interpolate(method="linear",
                                                                                         limit_direction="both"))
df["schooling"]= df.groupby("sub_region")["schooling"].transform(lambda x: x.fillna(x.mean()))

df["bmı"]= df.groupby("country")["bmı"].transform(lambda x: x.interpolate(method= "linear", 
                                                                          limit_direction="both"))
df["bmı"]= df.groupby("sub_region")["bmı"].transform(lambda x: x.fillna(x.mean()))

df["gdp"]= df.groupby("country")["gdp"].transform(lambda x: x.interpolate(method="linear",
                                                                          limit_direction="both"))
df["gdp"]= df.groupby("sub_region")["gdp"].transform(lambda x: x.fillna(x.mean()))

df["hepatitis_b"]= df.groupby("country")["hepatitis_b"].transform(lambda x: x.interpolate(method="linear",
                                                                                          limit_direction= "both"))
df["hepatitis_b"]= df.groupby("sub_region")["hepatitis_b"].transform(lambda x: x.fillna(x.mean()))

df["life_expectancy"]= df.groupby("country")["hepatitis_b"].transform(lambda x: x.interpolate(method="linear", 
                                                                                               limit_direction= "both"))
df["life_expectancy"]= df.groupby("sub_region")["hepatitis_b"].transform(lambda x: x.fillna(x.mean()))

df["percentage_expenditure"]= df.groupby("country")["percentage_expenditure"].transform(lambda x: x.interpolate(method="linear"
                                                                                                                ,limit_direction="both"))
df["percentage_expenditure"]= df.groupby("sub_region")["percentage_expenditure"].transform(lambda x: x.fillna(x.mean()))
                                                                
df.isnull().sum()*100/df.shape[0]
plt.figure(figsize=(20,14))
column_names= ["life_expectancy","percentage_expenditure", "alcohol", "schooling", "hepatitis_b", "bmı", "gdp"]
for i in range(7):
    plt.subplot(3,3,i+1)
    plt.boxplot(df[column_names[i]])
    plt.title("{} Box Graph".format(column_names[i], fontdict=title_font))
plt.show()
plt.figure(figsize=(20,14))
column_names= ["life_expectancy","percentage_expenditure", "alcohol", "schooling", "hepatitis_b", "bmı", "gdp"]
for i in range(7):
    plt.subplot(3,3,i+1)
    plt.hist(df[column_names[i]], bins=50)
    plt.title("{} Histogram Graph".format(column_names[i]), fontdict=title_font)
plt.show()
column_names= ["life_expectancy","percentage_expenditure", "alcohol", "schooling", "hepatitis_b", "bmı", "gdp"]
plt.figure(figsize=(20,14))
for i in range(7):
    plt.subplot(3,3,i+1)
    plt.boxplot(np.log(df[column_names[i]]+1))
    plt.title("{} Box Graph with Log".format(column_names[i]), fontdict=title_font)
plt.show()   
column_names= ["life_expectancy","percentage_expenditure", "alcohol", "schooling", "hepatitis_b", "bmı", "gdp"]
for name in range(0,7):
    zscorelist= []
    zscores= zscore(df[column_names[name]])
    for thereshold in np.arange(1,5,0.1):
        zscorelist.append((thereshold, len(np.where(zscores>thereshold)[0])))
        df_outliers= pd.DataFrame(zscorelist, columns= ["thereshold", "outliers"])
    plt.figure(figsize=(8,4))
    plt.plot(df_outliers.thereshold, df_outliers.outliers)
    plt.title("{} outliers".format(column_names[name]), fontdict=title_font)
    plt.show()
        

column_names= ["life_expectancy","percentage_expenditure", "alcohol", "schooling", "hepatitis_b", "bmı", "gdp"]
for name in column_names:
    q75, q25= np.percentile(df[name], [75,25])
    caa= q75-q25
    q75_log, q25_log= np.percentile(np.log(df[name]+1), [75,25])
    caa_log= q75_log, q25_log
    comparation= pd.DataFrame()
    for thereshold in np.arange(0,5,0.5):
        max_value= q75+ (caa*thereshold)
        min_value= q25- (caa*thereshold)
        max_value_log= q75+ (caa*thereshold)
        min_value_log= q25- (caa*thereshold)
        outliers= len((np.where((df[name]>max_value) | (df[name]<min_value))[0]))
        outliers_log= len((np.where((np.log(df[name]+1)>max_value_log) | (np.log(df[name]+1)<min_value_log))[0]))
        comparation= comparation.append({"thereshold": thereshold, "outliers {}".format(name): outliers,
                                         "outliers_log": outliers_log}, ignore_index=True)
    display(comparation)
df["winsorize_life_expectancy"]= winsorize(df["life_expectancy"], (0.06,0.06))
df["winsorize_schooling"]= winsorize(df["schooling"], (0.155,0.155))
df["winsorize_hepatitis_b"]= winsorize(df["hepatitis_b"], (0,0.8))
df["winsorize_gdp"]= winsorize(df["gdp"], (0.05,0.05))
winsorize_columns= ["life_expectancy", "winsorize_life_expectancy", "schooling", "winsorize_schooling",
                    "hepatitis_b", "winsorize_hepatitis_b", "gdp", "winsorize_gdp"]
plt.figure(figsize=(14,10))
for i in range(0,len(winsorize_columns)):
    plt.subplot(4,2,i+1)
    plt.boxplot(df[winsorize_columns[i]])
    plt.title(winsorize_columns[i], fontdict=title_font)
    
df["log_percentage_expenditure"]= np.log(df["percentage_expenditure"]+1)
df["log_alcohol"]= np.log(df["alcohol"]+1)
log_columns= ["percentage_expenditure", "log_percentage_expenditure", "alcohol", "log_alcohol"]
plt.figure(figsize=(12,7))
for i in range(0,len(log_columns)):
    plt.subplot(2,2,i+1)
    plt.boxplot(df[log_columns[i]])
    plt.title(log_columns[i], fontdict=title_font)
plt.show()   
log_columns= ["percentage_expenditure", "log_percentage_expenditure", "alcohol", "log_alcohol"]
plt.figure(figsize=(14,6))
for i in range(len(log_columns)):
    plt.subplot(2,2,i+1)
    plt.hist(df[log_columns[i]], bins=50)
    plt.title(log_columns[i], fontdict=title_font)
plt.show()  #log dönüşümü sonrası aykırı değer değşiminin gösterimi
pd.options.display.float_format= "{:.5f}".format
df.describe()
df_status= pd.DataFrame(df.groupby("status").mean())
df_status.drop(["year"], inplace=True, axis=1)
df_status.head()
plt.figure(figsize=(12,6))
sns.barplot(df["status"], df["winsorize_life_expectancy"])
plt.title("Status & Life Expectancy", fontdict=title_font)
plt.xlabel("Status", fontdict=label_font)
plt.ylabel("Life Expectancy", fontdict=label_font)
plt.show()
statu= df.status.unique()
comparasion= pd.DataFrame(columns=["group_1", "group_2", "statistics", "p_value"])
pd.options.display.float_format= "{:.6f}".format
for i in range(0,len(statu)):
    for j in range(i+1, len(statu)):
        ttest= stats.ttest_ind(df[df["status"]==statu[i]]["winsorize_life_expectancy"],
                               df[df["status"]==statu[j]]["winsorize_life_expectancy"])
        group_1= statu[i]
        group_2= statu[j]
        statistics= ttest[0]
        p_value= ttest[1]
        comparasion= comparasion.append({"group_1": group_1, "group_2": group_2, "statistics": statistics,
                                         "p_value": p_value}, ignore_index=True)
        display(comparasion)
column_names = ["log_percentage_expenditure", "log_alcohol", "winsorize_schooling", "winsorize_hepatitis_b", "bmı"]
plt.figure(figsize=(20,14))
for i in range(0,len(column_names)):
    plt.subplot(2,3,i+1)
    plt.scatter(df[column_names[i]], df["winsorize_life_expectancy"])
    plt.title("{} & Life Expectancy".format(column_names[i]), fontdict=title_font)
    plt.xlabel(column_names[i], fontdict=label_font)
    plt.ylabel("Life Expectancy", fontdict=label_font)

    # ayrılacak renklendirilecek. seaborn hue 
plt.show()

df.columns
column_names = ["log_percentage_expenditure", "log_alcohol", "winsorize_schooling", "winsorize_hepatitis_b", "bmı"]
plt.figure(figsize=(20,14))
for i in range(len(column_names)):
    plt.subplot(3,2,i+1)
    sns.scatterplot(data=df, x=column_names[i], y="winsorize_life_expectancy", hue="status") # region ile denencek
    plt.title(column_names[i], fontdict=title_font)
plt.show()
plt.figure(figsize=(20,12))
sns.scatterplot(data=df, x="alcohol", y="winsorize_life_expectancy", hue="sub_region")
plt.title("Life Expectancy & Hepatitis B with Region", fontdict=title_font)
plt.xlabel("hepatitis_b", fontdict=label_font)
plt.ylabel("winsorize_life_expectancy", fontdict=label_font)
plt.show()  # bu şekilde hepsini göstermeli miyim?
df1= df.copy()
column_names= ["winsorize_life_expectancy","log_percentage_expenditure", "log_alcohol", 
               "winsorize_schooling", "winsorize_hepatitis_b", "bmı", "country", "year"]
for col in df1.columns:
    if col not in column_names:
        df1.drop([col], inplace=True, axis=1)
df1.head()
df1_matrix= df1.corr()
display(df1_matrix)
plt.figure(figsize=(12,7))
sns.heatmap(df1_matrix, square=True, annot=True, linewidths=.5, vmin=0, vmax=1, cmap="viridis")
plt.title("Korelasyon Matrisi", fontdict=title_font)
plt.show() # döngü ile ayrı ayrı çizilebilir.
category_col= ["sub_region", "status"]
drop_col= ["alpha_3", "country", "country", "sub_region", "status", "year"]
df2= pd.concat([df, pd.get_dummies(df[category_col])], axis=1)
df2.drop(drop_col, inplace=True, axis=1)
df2.head()
df2.columns
# nomalleştirme
df2["norm_bmı"]= normalize(np.array(df2["bmı"]).reshape(1,-1)).reshape(-1,1)
df2["norm_winsorize_life_expectancy"]= normalize(np.array(df2["winsorize_life_expectancy"]).reshape(1,-1)).reshape(-1,1)
df2["norm_log_percentage_expenditure"]= normalize(np.array(df2["log_percentage_expenditure"]).reshape(1,-1)).reshape(-1,1)
df2["norm_log_alcohol"]= normalize(np.array(df2["log_alcohol"]).reshape(1,-1)).reshape(-1,1)
df2["norm_winsorize_schooling"]= normalize(np.array(df2["winsorize_schooling"]).reshape(1,-1)).reshape(-1,1)
df2["norm_winsorize_hepatitis_b"]= normalize(np.array(df2["winsorize_hepatitis_b"]).reshape(1,-1)).reshape(-1,1)
df2["norm_winsorize_gdp"]= normalize(np.array(df2["winsorize_gdp"]).reshape(1,-1)).reshape(-1,1)

normal_features= ["norm_bmı", "norm_winsorize_life_expectancy", "norm_log_percentage_expenditure", "norm_log_alcohol",
                  "norm_winsorize_schooling", "norm_winsorize_hepatitis_b", "norm_winsorize_gdp"]
print("Minimum Values\n----------------")
print(df2[normal_features].min())
print("Maximum Values\n----------------")
print(df2[normal_features].max()) 
# standartlaştırma
df2["scale_bmı"]= scale(df2["bmı"])
df2["scale_winsorize_life_expectancy"]= scale(df2["winsorize_life_expectancy"])
df2["scale_log_percentage_expenditure"]= scale(df2["log_percentage_expenditure"])
df2["scale_log_alcohol"]= scale(df2["log_alcohol"])
df2["scale_winsorize_schooling"]= scale(df2["winsorize_schooling"])
df2["scale_winsorize_hepatitis_b"]= scale(df2["winsorize_hepatitis_b"])
df2["scale_winsorize_gdp"]= scale(df2["winsorize_gdp"])

scale_features= ["scale_bmı", "scale_winsorize_life_expectancy", "scale_log_percentage_expenditure", "scale_log_alcohol",
                "scale_winsorize_schooling", "scale_winsorize_hepatitis_b", "scale_winsorize_gdp"]
print("Standart Deviations\n---------")
print(df2[scale_features].std())
print("Mean Values\n--------")
print(df2[scale_features].mean())
from sklearn import linear_model
import statsmodels.api as sm
df3= df2.copy()
df3.drop(['life_expectancy', 'adult_mortality', 'infant_deaths',
       'alcohol', 'percentage_expenditure', 'hepatitis_b', 'measles',
       'under-five deaths ', 'Polio', 'total_expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'gdp', 'population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'schooling'], inplace=True, axis=1)
df3.columns #regresyon analizimi yapacağım dataframe bu.
df_r= df3.copy()

df_r.drop(['norm_bmı', 'norm_winsorize_life_expectancy',
       'norm_log_percentage_expenditure', 'norm_log_alcohol',
       'norm_winsorize_schooling', 'norm_winsorize_hepatitis_b',
       'norm_winsorize_gdp', 'scale_bmı', 'scale_winsorize_life_expectancy',
       'scale_log_percentage_expenditure', 'scale_log_alcohol',
       'scale_winsorize_schooling', 'scale_winsorize_hepatitis_b',
       'scale_winsorize_gdp'], inplace=True, axis=1)
df_r.columns
request_col= []
for i in df_r.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y= df_r["winsorize_life_expectancy"]
X= df_r[request_col]
X= sm.add_constant(X)
model_1= sm.OLS(Y,X).fit()
model_1.summary()
request_col= ["bmı", "winsorize_life_expectancy", "log_percentage_expenditure", "log_alcohol",
                "winsorize_schooling", "winsorize_hepatitis_b", "winsorize_gdp"]
df_r= df3.copy()
for i in df3.columns:
    if i not in request_col:
        df_r.drop([i], inplace=True, axis=1)
df_r.columns
request_col= []
for i in df_r.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y= df_r["winsorize_life_expectancy"]
X= df_r[request_col]
X= sm.add_constant(X)
model_2= sm.OLS(Y,X).fit()
model_2.summary()
request_col= ["norm_bmı", "norm_winsorize_life_expectancy", "norm_log_percentage_expenditure", "norm_log_alcohol",
                "norm_winsorize_schooling", "norm_winsorize_hepatitis_b", "norm_winsorize_gdp",'sub_region_Central Asia', 'sub_region_Eastern Asia',
           'sub_region_Eastern Europe',
           'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
           'sub_region_Micronesia', 'sub_region_Northern Africa',
           'sub_region_Northern America', 'sub_region_Northern Europe',
           'sub_region_Polynesia', 'sub_region_South-eastern Asia',
           'sub_region_Southern Asia', 'sub_region_Southern Europe',
           'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
           'sub_region_Western Europe', 'status_Developed', 'status_Developing']
df_r= df3.copy()
for i in df3.columns:
    if i not in request_col:
        df_r.drop([i], inplace=True, axis=1)
df_r.columns
request_col= []
refuse_col= ["norm_winsorize_life_expectancy"]
for i in df_r.columns:
    if i not in refuse_col:
        request_col.append(i)
request_col
Y= df_r["norm_winsorize_life_expectancy"]
X= df_r[request_col]
X= sm.add_constant(X)
model_3= sm.OLS(Y,X).fit()
model_3.summary()
df3.columns
df_r= df3.copy()
request_col= ['scale_bmı', 'scale_winsorize_life_expectancy',
       'scale_log_percentage_expenditure', 'scale_log_alcohol',
       'scale_winsorize_schooling', 'scale_winsorize_hepatitis_b',
       'scale_winsorize_gdp', 'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing']
for i in df_r.columns:
    if i not in request_col:
        df_r.drop([i], inplace=True, axis=1)
df_r.columns
request_col= []
for i in df_r.columns:
    if i != "scale_winsorize_life_expectancy":
        request_col.append(i)
request_col
Y=df_r["scale_winsorize_life_expectancy"]
X=df_r[request_col]
X= sm.add_constant(X)
model_4= sm.OLS(Y,X).fit()
model_4.summary()
df_r= df3.copy()
request_col= ["winsorize_life_expectancy",'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing']
for i in df_r.columns:
    if i not in request_col:
        df_r.drop([i], inplace=True, axis=1)
df_r.columns
request_col= []
for i in df_r.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y=df_r["winsorize_life_expectancy"]
X=df_r[request_col]
X= sm.add_constant(X)
model_5= sm.OLS(Y,X).fit()
model_5.summary()
df_r= df3.copy()
request_col= ['winsorize_life_expectancy','winsorize_hepatitis_b',
              'log_percentage_expenditure', 'log_alcohol', 'bmı']
for i in df_r.columns:
    if i not in request_col:
        df_r.drop([i], inplace=True, axis=1)
df_r.columns
request_col= []
for i in df_r.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y=df_r["winsorize_life_expectancy"]
X= df_r[request_col]
X= sm.add_constant(X)
model_6= sm.OLS(Y,X).fit()
model_6.summary()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
df_t= df3.copy()
request_col= ['bmı', 'winsorize_life_expectancy', 'winsorize_schooling',
       'winsorize_hepatitis_b', 'winsorize_gdp', 'log_percentage_expenditure',
       'log_alcohol', 'sub_region_Australia and New Zealand',
       'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing']
for i in df3.columns:
    if i not in request_col:
        df_t.drop([i], inplace=True, axis=1)
df_t.columns
request_col= []
for i in df_t.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2)

print("Eğitim kümesindeki eleman sayısı: {}".format(X_train.shape[0]))
print("Test kümesindeki eleman sayısı: {}".format(X_test.shape[0]))

X_train = sm.add_constant(X_train)
model_1= sm.OLS(Y_train, X_train).fit()
model_1.summary()
X_test= sm.add_constant(X_test)
Y_preds= model_1.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(0,100)
plt.title("Life Expectancy Gerçek Ve Tahmin Değerleri", fontdict=title_font) 
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmin Değerleri", fontdict=label_font)
plt.show()

print("Ortalama Mutlak Hata (MAE): {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Ortalama Kare Hata (MSE): {}".format(mse(Y_test, Y_preds)))
print("Kök Ortalama Kare Error (RMSE): {}".format(rmse(Y_test, Y_preds)))
print("Ortalama Mutlak Yüzde Hata (MAPE): {}".format(np.mean(np.abs((Y_test-Y_preds) / Y_test))*100))
df_t= df3.copy()
request_col= ['winsorize_life_expectancy','sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing']
for i in df_t.columns:
    if i not in request_col:
        df_t.drop([i], inplace=True, axis=1)
request_col
request_col= []
for i in df_t.columns:
    if i != "winsorize_life_expectancy":
        request_col.append(i)
request_col
Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print("Eğitim kümesi eleman sayısı: {}".format(X_train.shape[0]))
print("Test kümesi eleman sayısı: {}".format(X_test.shape[0]))


X_train= sm.add_constant(X_train)
model_5.summary()
X_test= sm.add_constant(X_test)
Y_preds= model_5.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x= Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(30,100)
plt.title("Gerçek Ve Tahmini Değerler", fontdict=title_font)
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmini Değerler", fontdict=label_font)
plt.show()

print("Ortalama Mutlak Hata (MAE): {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Ortalama Kare Hata (MSE): {}".format(mse(Y_test, Y_preds)))
print("Kök Ortalama Kare Hata (RMSE): {}".format(rmse(Y_test, Y_preds)))
print("Ortalama Mutlak Yüzde Hata (MAPE): {}".format(np.mean(np.abs((Y_test-Y_preds) / Y_test))*100))
df3.columns
df_t= df3.copy()
request_col= ['bmı', 'winsorize_schooling',
       'winsorize_hepatitis_b', 'winsorize_gdp', 'log_percentage_expenditure',
       'log_alcohol','norm_bmı' ] 
Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train= sm.add_constant(X_train)
model_2= sm.OLS(Y_train, X_train).fit()
model_2.summary()#

X_test= sm.add_constant(X_test)
Y_preds= model_2.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(50,100)
plt.title("Gerçek Ve Tahmini Değerler", fontdict=title_font)
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmini Değerler", fontdict=label_font)
plt.show()

print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Mean Square Error: {}".format(mse(Y_test, Y_preds)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_preds)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_preds) / Y_test))*100))

df_t= df3.copy()
request_col= ['bmı', 'sub_region_Australia and New Zealand',
       'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing',
       'scale_log_percentage_expenditure', 'scale_log_alcohol',
       'scale_winsorize_schooling', 'scale_winsorize_hepatitis_b',
       'scale_winsorize_gdp']

Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train= sm.add_constant(X_train)
model_4= sm.OLS(Y_train, X_train).fit()
model_4.summary()
X_test= sm.add_constant(X_test)
y_preds= model_4.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue= df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(40,100)
plt.title("Geçrek Ve Tahmin Değerleri ", fontdict=title_font)
plt.xlabel("Geçrek Değerler", fontdict=label_font)
plt.ylabel("Tahmin Değerleri", fontdict=label_font)
plt.show()

print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Mean Square Error: {}".format(mse(Y_test, Y_preds)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_preds)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_preds) / Y_test))*100))

df_t= df2.copy()
request_col= ['percentage_expenditure', 'alcohol','winsorize_schooling',
       'winsorize_hepatitis_b', 'winsorize_gdp', 'sub_region_Australia and New Zealand',
       'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing']
Y=df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train= sm.add_constant(X_train)
model_7= sm.OLS(Y_train, X_train).fit()
model_7.summary()
X_test= sm.add_constant(X_test)
Y_preds= model_7.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(40,100)
plt.title("Gerçek Ve Tahmin Değerleri",fontdict=title_font)
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmini Değerler", fontdict=label_font)
plt.show()

print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Mean Square Error: {}".format(mse(Y_test, Y_preds)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_preds)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_preds)/ Y_test))*100))

df_t= df3.copy()
request_col= ['sub_region_Australia and New Zealand',
       'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing',
       'norm_bmı',
       'norm_log_percentage_expenditure', 'norm_log_alcohol',
       'norm_winsorize_schooling', 'norm_winsorize_hepatitis_b',
       'norm_winsorize_gdp']


Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)

X_train= sm.add_constant(X_train)
model_3= sm.OLS(Y_train, X_train).fit()
model_3.summary()
X_test= sm.add_constant(X_test)
Y_preds= model_3.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.xlim(20,100)
plt.title("Gerçek Ve Tahmini Değeler", fontdict=title_font)
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmini Değerler", fontdict=label_font)
plt.show()

print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Mean Square Error: {}".format(mse(Y_test, Y_preds)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_preds)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test- Y_preds) / Y_test))*100))
df_t= df3.copy()
request_col= ['winsorize_hepatitis_b',
              'log_percentage_expenditure', 'log_alcohol', 'bmı']
Y= df_t["winsorize_life_expectancy"]
X= df_t[request_col]

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)

X_train= sm.add_constant(X_train)
model_6= sm.OLS(Y_train, X_train).fit()
model_6.summary()
X_test= sm.add_constant(X_test)
Y_preds= model_6.predict(X_test)

plt.figure(dpi=100)
sns.scatterplot(x=Y_test, y=Y_preds, hue=df["status"])
sns.lineplot(x=Y_test, y=Y_test, color="black")
plt.title("Gerçek Ve Tahmin Değerleri", fontdict=title_font)
plt.xlim(20,120)
plt.xlabel("Gerçek Değerler", fontdict=label_font)
plt.ylabel("Tahmin Değerleri", fontdict=label_font)
plt.show()

print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_preds)))
print("Mean Square Error: {}".format(mse(Y_test, Y_preds)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_preds)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_preds)/ Y_test))*100))



from sklearn.linear_model import Ridge 
request_col= ['sub_region_Australia and New Zealand',
       'sub_region_Central Asia', 'sub_region_Eastern Asia',
       'sub_region_Eastern Europe',
       'sub_region_Latin America and the Caribbean', 'sub_region_Melanesia',
       'sub_region_Micronesia', 'sub_region_Northern Africa',
       'sub_region_Northern America', 'sub_region_Northern Europe',
       'sub_region_Polynesia', 'sub_region_South-eastern Asia',
       'sub_region_Southern Asia', 'sub_region_Southern Europe',
       'sub_region_Sub-Saharan Africa', 'sub_region_Western Asia',
       'sub_region_Western Europe', 'status_Developed', 'status_Developing',
       'norm_bmı', 'norm_winsorize_life_expectancy',
       'norm_log_percentage_expenditure', 'norm_log_alcohol',
       'norm_winsorize_schooling', 'norm_winsorize_hepatitis_b',
       'norm_winsorize_gdp']

Y= df3["winsorize_life_expectancy"]
X= df3[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

for i in range(1,10):
    ridgereg= Ridge(alpha=10**i)
    ridgereg.fit(X_train, Y_train)

    Y_test_tahmini= ridgereg.predict(X_test)
    
    print("************")
    print("Alpha değeri 10 üzeri {} için: ".format(i))
    print("Eğitim kümesi gözlem sayısı: {}".format(Y_train.shape[0]))
    print("Test kümesi gözlem sayısı: {}".format(Y_test.shape[0]))
    print("Eğitim kümesindeki R-kare değeri: {}".format(ridgereg.score(X_train, Y_train)))
    print("-------Test kümesi istatistikleri---------")
    print("Test kümesindeki R-kare değeri: {}".format(ridgereg.score(X_test, Y_test)))
    print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_test_tahmini)))
    print("Mean Square Error: {}".format(mse(Y_test, Y_test_tahmini)))
    print("Root Mean Square Error: {}".format(rmse(Y_test, Y_test_tahmini)))
    print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_test_tahmini)/ Y_test))*100))
Y= df3["winsorize_life_expectancy"]
X= df3[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

for i in range(1,40):
    ridgereg= Ridge(alpha=10**i)
    ridgereg.fit(X_train, Y_train)
    
    Y_test_tahmini= ridgereg.predict(X_test)
    if ridgereg.score(X_train, Y_train)<1.0:
         print("Eğitim kümesi R kare değerleri: {}".format(ridgereg.score(X_train, Y_train)))
    elif ridgereg.score(X_train, Y_train)>0.5:
                 print("Eğitim kümesi R kare değerleri: {}".format(ridgereg.score(X_train, Y_train)))

Y= df3["winsorize_life_expectancy"]
X= df3[request_col]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


ridgereg= Ridge(alpha=10*37)
ridgereg.fit(X_train, Y_train)

Y_test_tahmini= ridgereg.predict(X_test)

print("Eğitim kümesi gözlem sayısı: {}".format(Y_train.shape[0]))
print("Test kümesi gözlem sayısı: {}".format(Y_test.shape[0]))

print("Eğitim kümesindeki R-kare değeri: {}".format(ridgereg.score(X_train, Y_train)))
print("-------Test kümesi istatistikleri---------")
print("Test kümesindeki R-kare değeri: {}".format(ridgereg.score(X_test, Y_test)))
print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_test_tahmini)))
print("Mean Square Error: {}".format(mse(Y_test, Y_test_tahmini)))
print("Root Mean Square Error: {}".format(rmse(Y_test, Y_test_tahmini)))
print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_test_tahmini)/ Y_test))*100))
from sklearn.linear_model import Lasso
for i in range(1,30):
    print("******************")
    print("Alpha değeri {} için:".format(i))
    lassoreg= Lasso(alpha=10**i)
    lassoreg.fit(X_train, Y_train)

    Y_test_tahmini= lassoreg.predict(X_test)
    
    print("Eğitim kümesi R-kare değeri: {}".format(lassoreg.score(X_train, Y_train)))
    print("----Test Kümesi Hakkında İstatistikler-------")
    print("Test kümesi R-kare değeri: {}".format(lassoreg.score(X_test, Y_test)))
    print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_test_tahmini)))
    print("Mean Square Error: {}".format(mse(Y_test, Y_test_tahmini)))
    print("Root Mean Square Error: {}".format(rmse(Y_test, Y_test_tahmini)))
    print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_test_tahmini)/ Y_test))*100))

from sklearn.linear_model import ElasticNet
for i in range(1,30):
    elasticreg= ElasticNet(alpha=10**i, l1_ratio=0.5)
    elasticreg.fit(X_train, Y_train)

    Y_test_tahmini= elasticreg.predict(X_test)
    print("*********************")
    print("Alpha değeri 10 üzeri {} için: ".format(i))
    print("Eğitim kümesi R-kare değeri: {}".format(elasticreg.score(X_train, Y_train)))
    print("----Test Kümesi Hakkında İstatistikler-------")
    print("Test kümesi R-kare değeri: {}".format(elasticreg.score(X_test, Y_test)))
    print("Mean Absolute Error: {}".format(mean_absolute_error(Y_test, Y_test_tahmini)))
    print("Mean Square Error: {}".format(mse(Y_test, Y_test_tahmini)))
    print("Root Mean Square Error: {}".format(rmse(Y_test, Y_test_tahmini)))
    print("Mean Absolute Percentage Error: {}".format(np.mean(np.abs((Y_test-Y_test_tahmini)/ Y_test))*100))