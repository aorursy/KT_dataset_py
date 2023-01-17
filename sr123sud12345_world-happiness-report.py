from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import statistics
df=pd.read_csv("2015.csv")
df.tail(2)
def cntry(c,df):
    for i in np.arange(len(df["Country"])):
        if df["Country"][i]==c:
            ind=(df["Happiness Rank"][i])
            break
        else:
            ind="not obtained"
    print("The happyness index for",c,'was',ind)
cntry("India",df)
df["score"]=np.arange((len(df["Country"])),0,-1)
df1=DataFrame([df["Economy (GDP per Capita)"],df["Family"],df["Health (Life Expectancy)"],df["Freedom"],df["Trust (Government Corruption)"],df["score"]]).T

df1.head()
df2=df1.corr()
df2
sns.pairplot(df1)
sns.heatmap(df2,annot=True)
df_16=pd.read_csv("2016.csv")
df_17=pd.read_csv("2017.csv")
df_18=pd.read_csv("WorldHappiness2018_Data.csv")
df_19=pd.read_csv("world-happiness-report-2019.csv")
df_16["score"]=np.arange((len(df_16["Country"])),0,-1)
df_17["score"]=np.arange((len(df_17["Country"])),0,-1)
df_18["score"]=np.arange((len(df_18["Country"])),0,-1)
df_19["score"]=np.arange((len(df_19["Country (region)"])),0,-1)
df_19.columns
df1_16=DataFrame([df_16["Economy (GDP per Capita)"],df_16["Family"],df_16["Health (Life Expectancy)"],df_16["Freedom"],df_16["Trust (Government Corruption)"],df_16["score"]]).T
df1_17=DataFrame([df_17["Economy..GDP.per.Capita."],df_17["Family"],df_17["Health..Life.Expectancy."],df_17["Freedom"],df_17["Trust..Government.Corruption."],df_17["score"]]).T
df1_18=DataFrame([df_18["GDP_Per_Capita"],df_18["Social_Support"],df_18["Healthy_Life_Expectancy"],df_18["Freedom_To_Make_Life_Choices"],df_18["score"]]).T
df1_19=DataFrame([df_19['Log of GDP\nper capita'],df_19["Social support"],df_19["Healthy life\nexpectancy"],df_19["Freedom"],df_19["score"]]).T
sns.heatmap(df1_19.corr(),annot=True)
sns.heatmap(df1_16.corr(),annot=True)
#country_wise_study_nature of factors_associated_with_their_rank
def info_15(c):
    for i in np.arange(len(df["Country"])):
        if df["Country"][i]==c:
            rank=df["Happiness Rank"][i]
            eco=df["Economy (GDP per Capita)"][i]
            fam=df["Family"][i]
            hel=df["Health (Life Expectancy)"][i]
            fre=df["Freedom"][i]
            break
        else:
            rank=0
            eco=0
            fam=0
            hel=0
            fre=0
    v=[rank,eco,fam,hel,fre]
    return (v)
def info_16(c):
    for i in np.arange(len(df_16["Country"])):
        if df_16["Country"][i]==c:
            rank=df_16["Happiness Rank"][i]
            eco=df_16["Economy (GDP per Capita)"][i]
            fam=df_16["Family"][i]
            hel=df_16["Health (Life Expectancy)"][i]
            fre=df_16["Freedom"][i]
            break
        else:
            rank=0
            eco=0
            fam=0
            hel=0
            fre=0
    v=[rank,eco,fam,hel,fre]
    return (v)
def info_17(c):
    for i in np.arange(len(df_17["Country"])):
        if df_17["Country"][i]==c:
            rank=df_17["Happiness.Rank"][i]
            eco=df_17["Economy..GDP.per.Capita."][i]
            fam=df_17["Family"][i]
            hel=df_17["Health..Life.Expectancy."][i]
            fre=df_17["Freedom"][i]
            break
        else:
            rank=0
            eco=0
            fam=0
            hel=0
            fre=0
    v=[rank,eco,fam,hel,fre]
    return (v)
def info_18(c):
    for i in np.arange(len(df_18["Country"])):
        if df_18["Country"][i]==c:
            rank=df_18["Rank"][i]
            eco=df_18["GDP_Per_Capita"][i]
            fam=df_18["Social_Support"][i]
            hel=df_18["Healthy_Life_Expectancy"][i]
            fre=df_18["Freedom_To_Make_Life_Choices"][i]
            break
        else:
            rank=0
            eco=0
            fam=0
            hel=0
            fre=0
    v=[rank,eco,fam,hel,fre]
    return (v)
def info_19(c):
    for i in np.arange(len(df_19["Country (region)"])):
        if df_19["Country (region)"][i]==c:
            rank=df_19["Ladder"][i]
            eco=df_19["Log of GDP\nper capita"][i]
            fam=df_19["Social support"][i]
            hel=df_19["Healthy life\nexpectancy"][i]
            fre=df_19["Freedom"][i]
            break
        else:
            rank=0
            eco=0
            fam=0
            hel=0
            fre=0
    v=[rank,eco,fam,hel,fre]
    return (v)
info_19("Finland")
def info(c):
    a_15=info_15(c)
    a_16=info_16(c)
    a_17=info_17(c)
    a_18=info_18(c)
    a_19=info_19(c)
    a_df=DataFrame([a_15,a_16,a_17,a_18,a_19],index=[2015,2016,2017,2018,2019],columns=["Rank","GDP","Family","Life_expectency","Freedom"])
    return(a_df)
info("India")["Rank"].plot()
info("Pakistan")["Rank"].plot()
plt.title("Rank of country corrosponding to year",fontsize=20)
plt.xlabel("year",fontsize=20)
plt.ylabel("Happiness Rank",fontsize=20)
plt.legend(["India","Pakistan"])
info("India")["Family"][:-1].plot()
info("Pakistan")["Family"][:-1].plot()
info("Switzerland")["Rank"].plot()
info("Iceland")["Rank"].plot()
info("Denmark")["Rank"].plot()
info("Norway")["Rank"].plot()
plt.title("Rank of country corrosponding to year",fontsize=20)
plt.xlabel("year",fontsize=20)
plt.ylabel("Happiness Rank",fontsize=20)
plt.legend(["Switzerland","Iceland","Denmark","Norway"])
info("United Kingdom")["Rank"].plot()
info("United States")["Rank"].plot()
plt.legend(["UK","US"])
info("Switzerland")
#now i'm gonna predict the factors like gdp,family and based upon those predictions u have to predict the  rank you can use 2019 as test data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
def reg_le(c,fac):
    #model_le=LinearRegression()
    model_le=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=18, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
    info_df=info(c)
    x=info_df.index.values.reshape(-1,1)[:-1]
    y=info_df[fac].values.reshape(-1,1)[:-1]
    model_le.fit(x,y)
    y_test=np.array([2019,2020])
    y_pre=model_le.predict(y_test.reshape(-1,1))
    return(y_pre)
reg_le("Switzerland","Family")[1]
(np.append(info("Switzerland")["Life_expectency"][:-1],reg_le("Switzerland","Life_expectency")))
factors=info("India").columns[1:]
df_20=DataFrame([],index=factors)
v2=[]
for i in df["Country"]:
    v1=[]
    for j in factors:
        v1=np.append(v1,reg_le(i,j)[1])
    df_20[i]=v1
df_20=df_20.T
x_test=df_20
x_test.head()
df1=df1.drop("Trust (Government Corruption)",1)
df1_16=df1_16.drop("Trust (Government Corruption)",1)
df1_17=df1_17.drop("Trust..Government.Corruption.",1)
df_train1=DataFrame(df1.values).append(DataFrame(df1_16.values)).append(DataFrame(df1_17.values))
df_train=df_train1.append(DataFrame(df1_18.values))
df_train.columns=df1.columns
y_train=df_train["score"].values.reshape(-1,1)
x_train=(df_train.drop("score",1)).values
model_le=LinearRegression()
model_le.fit(x_train,y_train)
y_pre=model_le.predict(x_test.values)
import xgboost as xgb
model_xg=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=18, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
model_xg.fit(x_train,y_train)
y_prexg=model_xg.predict(x_test.values)
y_pre1=[]
for i in (y_pre):
    y_pre1=np.append(y_pre1,np.round(i))
y_pre1
x_test["score"]=y_pre1
df1_20=x_test.sort_values(by="score",axis=0,ascending=False)
df1_20["rank"]=np.arange(1,len(df1_20["score"])+1)
df1_20       #this is the predicted happines report for 2020
info("India")["Rank"][::-1].plot()
def rank20(c):
    for i in np.arange(len(df1_20["rank"])):
        if df1_20.index[i]==c:
            print("The rank of ",c,"in 2020 would be",df1_20["rank"][i])
rank20("India")
r19=[]
r20=[]
c_name=[]
for i in np.arange(len(df_19["Ladder"])):
    for j in np.arange(len(df1_20["rank"])):
        if df_19["Country (region)"][i]==df1_20.index[j]:
            c_name=np.append(c_name,df_19["Country (region)"][i])
            r19=np.append(r19,df_19["Ladder"][i])
            r20=np.append(r20,df1_20["rank"][j])
df_1920=DataFrame([r19,r20],columns=c_name,index=["rank_2019","rank_2020le"]).T
df_1920.head()
df_1920["rank_2019"].plot(figsize=(20,6),kind="line")
df_1920["rank_2020le"].plot(figsize=(20,6))
plt.title("Rank of country corrosponding to year",fontsize=20)
plt.xlabel("country_name",fontsize=20)
plt.ylabel("Happiness Rank",fontsize=20)
plt.legend(["rank in year 2019","predicted rank for year 2020"])
y_pre11=[]
for i in y_prexg:
    y_pre11=np.append(y_pre11,int(i))
x_test["scorexg"]=y_pre11
df2_20=df1_20
df2_20=x_test.sort_values(by="scorexg",axis=0,ascending=False)
df2_20["rank"]=np.arange(1,len(df2_20["scorexg"])+1)
df2_20[10:20]    #this is the predicted happines report for 2020
r19=[]
r20=[]
c_name=[]
for i in np.arange(len(df_19["Ladder"])):
    for j in np.arange(len(df1_20["rank"])):
        if df_19["Country (region)"][i]==df2_20.index[j]:
            c_name=np.append(c_name,df_19["Country (region)"][i])
            r19=np.append(r19,df_19["Ladder"][i])
            r20=np.append(r20,df2_20["rank"][j])
df2_1920=DataFrame([r19,r20],columns=c_name,index=["rank_2019","rank_2020xg"]).T
df2_1920["rank_2019"].plot(figsize=(20,6),kind="line")
df2_1920["rank_2020xg"].plot(figsize=(20,6))
plt.title("Rank of country corrosponding to year",fontsize=20)
plt.xlabel("country_name",fontsize=20)
plt.ylabel("Happiness Rank",fontsize=20)
plt.legend(["rank in year 2019","predicted rank for year 2020"])
df_m=pd.merge(df_1920,df2_1920)
df_m.index=df_1920.index
sns.jointplot(df_m["rank_2019"].values,df_m["rank_2020le"],kind="reg")
sns.jointplot(df_m["rank_2019"].values,df_m["rank_2020xg"],kind="reg")
sns.lmplot("rank_2019","rank_2020le",df_m,order=2,
           scatter_kws={"marker":"o","color":"red"},
              line_kws={"linewidth":1,"color":"blue"})
sns.lmplot("rank_2019","rank_2020xg",df_m,order=2,
           scatter_kws={"marker":"o","color":"green"},
              line_kws={"linewidth":2,"color":"blue"})
df_m.corr()
sns.heatmap(df_m.corr(),annot=True)
df_m



