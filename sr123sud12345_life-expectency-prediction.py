from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("ahs-mort-odisha-sundargarh.csv")
df1=df[["rural","stratum_code","deceased_sex",
    "age_of_death_above_one_year","treatment_source","place_of_death","month_of_death","year_of_death","member_identity",
    "social_group_code","marital_status","year_of_marriage","highest_qualification","occupation_status","disability_status",
    "chew","smoke","alcohol","house_structure","drinking_water_source","household_have_electricity","lighting_source",
    "cooking_fuel","no_of_dwelling_rooms"]]
df1["year_of_death"].unique()
#df1.to_csv("anual_health_survey.csv")
df1.head(2)
def cat(cl):
    c=pd.get_dummies(df1[cl]).columns
    n=(len(df1[cl].values))
    ar=np.zeros(n)
    for i in np.arange(len(c)):
        for j in np.arange(n):
            if c[i]==df1[cl][j]:
                ar[j]=i
    return(ar)
def replacer_mean(dff):
    dff1=dff
    r0=np.mean(dff)
    r1=r0.index
    r2=r0.values
    for i in np.arange(len(r1)):
          ri=r1[i]
          rv=r2[i]
          dff1[ri].fillna(value=rv)
          dff1[ri]=(nan_remover(dff[ri].values,rv))
    return(dff1)
def nan_remover(v,vm):
    vr=[]
    for i in np.arange(len(v)):
        if str(v[i])=="nan":
            vr=np.append(vr,vm)
        else:
            vr=np.append(vr,v[i])
    return(vr)
def numriser(a):
    a1=[]
    for i in np.arange(len(a)):
        a1=np.append(a1,round(a[i]))
    return(a1)

def substi(ar,s):
    n=len(ar)
    sum1=0
    for i in np.arange(n):
        j=ar[i]
        df_c1["interval"][j]=s
        #sum1=sum1+ar[i]
    return(df_c1)
def inret(y,m):
    return(np.where((df_c1["year_of_death"]==y) & (df_c1["month_of_death"]==m))[0])
(cat("rural"))
df1.columns
df1["cooking_fuel"][0:1]
txt=["rural","stratum_code","deceased_sex","treatment_source","place_of_death","social_group_code","marital_status",
       "highest_qualification","occupation_status","disability_status", "chew","smoke","alcohol","house_structure","drinking_water_source",
    "household_have_electricity","lighting_source","cooking_fuel"]
for j in (txt):
    df1[j]=cat(j)
r0=(np.mean(df1))
r1=r0.index
r2=r0.values
df2=df1
df1=replacer_mean(df2)
y=df1["age_of_death_above_one_year"].values
df1=df1.drop(["age_of_death_above_one_year","year_of_death","month_of_death"],axis=1)
for i in np.arange(len(df1.columns)):
    sc=StandardScaler()
    cl=df1.columns[i]
    sc.fit(df1[cl].values.reshape(-1,1))
    df2[cl]=sc.transform(df2[cl].values.reshape(-1,1))
x=df1.values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
model_le=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=.2, gamma=0.2,
       importance_type='gain', learning_rate=0.3, max_delta_step=0,
       max_depth=15, min_child_weight=3, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
model_le.fit(x_train,y_train)
y_pre=model_le.predict(x_train)
accuracy_score(numriser(y_pre),numriser(y_train))
y_pre1=model_le.predict(x_test)
accuracy_score(numriser(y_pre1),numriser(y_test))
df_pre=DataFrame([y_pre1,y_test],index=["predicted","observed"]).T
df_pre["predicted"].plot(kind="kde",label=True,figsize=(15,5),alpha=1)
df_pre["observed"].plot(kind="kde",label=True,figsize=(15,5),alpha=1)
plt.title("predicted label of happiness",fontsize=20)
plt.xlabel("labels of happiness ",fontsize=20)
plt.ylabel("frequency",fontsize=20)
plt.legend(["predicted","observed"])
df_pre["predicted"][1:100].plot(figsize=(20,5))
df_pre["observed"][1:100].plot(figsize=(20,5))
plt.title("predicted age and observerd age classification",fontsize=20)
plt.xlabel("people ",fontsize=20)
plt.ylabel("age",fontsize=20)
plt.legend(["predicted","observed"])
(mean_squared_error(y_pre1,y_test))
df_c=df2[["month_of_death","year_of_death","age_of_death_above_one_year"]]
df_c["interval"]=np.zeros(len(df_c["month_of_death"]))
df_c.head(2)
mon=np.unique(df_c["month_of_death"])
year=np.unique(df_c["year_of_death"])
df_c1=df_c
s1=1
for i in year:
    for j in mon[1:]:
        indx=inret(i,j)
        df_c=substi(indx,s1)
        s1=s1+1
        #print(s1)
sum1=[]
for i in np.arange(len(df_c["interval"])):
    if df_c["interval"][i]==0:
        sum1=np.append(sum1,i)
df_c=df_c.drop(sum1)
d=[]
for i in np.arange(1,len(np.unique(df_c["interval"]))+1):
                d=np.append(d,sum(pd.get_dummies(df_c["interval"])[i]))
s=len(df_c["interval"].values)
s
s=len(df_c["interval"].values)
cs=[s]
for i in np.arange(1,49):
    cs1=s-d[i-1]
    cs=np.append(cs,cs1)
    s=cs1
clt=DataFrame([np.arange(1,49),cs,d],index=["age interval","cum_sum","deaths"]).T
qt=clt["deaths"]/clt["cum_sum"]
pt=(np.ones(len(qt))-qt)
clt["qt"]=qt
clt["pt"]=pt
clt["Pt"]=np.ones(len(qt))
for i in np.arange(0,47):
    clt["Pt"][i+1]=(clt["pt"][i]*clt["Pt"][i])
clt=clt.drop(48)
clt["Pt"].plot(figsize=(12,5))
plt.title("survival_rate",fontsize=20)
plt.xlabel("months ",fontsize=20)
plt.ylabel("prob_of_remission",fontsize=20)
plt.legend(["survival_odisha"])
df_p=pd.read_csv("ahs-mort-bihar-patna.csv")
df_cb=df_p[["month_of_death","year_of_death","age_of_death_above_one_year"]]
r0=(np.mean(df1))
r1=r0.index
r2=r0.values
df2=df_cb
df_cb=replacer_mean(df2)
mon1=np.unique(df_cb["month_of_death"])
year1=np.unique(df_cb["year_of_death"])
def substi_b(ar,s):
    n=len(ar)
    sum1=0
    for i in np.arange(n):
        j=ar[i]
        df_cb1["interval"][j]=s
        #sum1=sum1+ar[i]
    return(df_c1)
def inret_b(y,m):
    return(np.where((df_cb1["year_of_death"]==y) & (df_cb1["month_of_death"]==m))[0])
df_cb["interval"]=np.zeros(len(df_cb["month_of_death"]))

mon1=np.unique(df_cb["month_of_death"])
year1=np.unique(df_cb["year_of_death"])

df_cb1=df_cb

s1=1
for i in year:
    for j in mon[1:]:
        indx=inret_b(i,j)
        df_cb=substi_b(indx,s1)
        s1=s1+1
sum1=[]
for i in np.arange(len(df_c1["interval"])):
    if df_cb1["interval"][i]==0:
        sum1=np.append(sum1,i)
df_cb1=df_cb1.drop(sum1)
d=[]
for i in np.arange(1,len(np.unique(df_cb1["interval"]))):
                d=np.append(d,sum(pd.get_dummies(df_cb1["interval"])[i]))
s=len(df_cb1["interval"].values)
cs=[s]
for i in np.arange(1,49):
    cs1=s-d[i-1]
    cs=np.append(cs,cs1)
    s=cs1

clt_b=DataFrame([np.arange(1,49),cs,d],index=["age interval","cum_sum","deaths"]).T

qt_b=clt_b["deaths"]/clt_b["cum_sum"]
pt_b=(np.ones(len(qt_b))-qt_b)

clt_b["qt"]=qt_b
clt_b["pt"]=pt_b

clt_b["Pt"]=np.ones(len(qt_b))

for i in np.arange(0,47):
    clt_b["Pt"][i+1]=(clt_b["pt"][i]*clt_b["Pt"][i])

clt_b=clt_b.drop(48)

clt_b["Pt"].plot(figsize=(12,5))
clt["Pt"].plot(figsize=(12,5))
plt.title("survival_rate",fontsize=20)
plt.xlabel("months ",fontsize=20)
plt.ylabel("prob._of_remission",fontsize=20)
plt.legend(["survival_Bihar(patna)","survival_odisha(sundargadh)"])
clt_ob=clt_b
clt_ob["o"]=clt["Pt"].values
clt_ob.head(2)
#univariate_analysis
df1.columns
df1["no_of_dwelling_rooms"].plot(kind="hist",figsize=(9,5),color="red")
g=[sum(pd.get_dummies(df["smoke"])["Ex - Smoker"]),sum(pd.get_dummies(df["smoke"])["Never smoked"]),sum(pd.get_dummies(df["smoke"])["Not known"]),
                                          sum(pd.get_dummies(df["smoke"])["Occasional smoker"]),
                                          sum(pd.get_dummies(df["smoke"])["Usual smoker"])]
labels=["EX_smokers","Never smoked","not known","Occasional smoker","Occasional smoker"]
pie=plt.pie(g,radius=1.5,shadow=True,autopct='%1.1f%%')
plt.legend(pie[0], labels, loc="best")
df1["chew"].plot(kind="hist",figsize=(9,5),color="brown")
df_anova=(pd.get_dummies(df["chew"]))
#anova
import statsmodels.api as sm
from statsmodels.formula.api import ol
from statsmodels.formula.api import ols
df2.columns
lm=ols(" age_of_death_above_one_year ~ chew",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
lm=ols(" age_of_death_above_one_year ~ smoke",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
lm=ols(" age_of_death_above_one_year ~ cooking_fuel",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
lm=ols(" age_of_death_above_one_year ~ alcohol",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
lm=ols(" age_of_death_above_one_year ~ social_group_code",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
#df1["smoke"]
df2=df1
ty=np.unique(df2["smoke"])

ty=np.unique(df2["smoke"])
ex_smoker=[]
never_smoke=[]
not_known=[]
occ_smoker=[]
usual_smoker=[]
for j in np.arange(len(df1["smoke"])):
    if int(ty[0])==int(df2["smoke"][j]):
        ex_smoker=np.append(ex_smoker,y[j])
    if int(ty[1])==int(df2["smoke"][j]):
        never_smoke=np.append(never_smoke,y[j])
    if int(ty[2])==int(df2["smoke"][j]):
        not_known=np.append(not_known,y[j])
    if int(ty[3])==int(df2["smoke"][j]):
        occ_smoker=np.append(occ_smoker,y[j])
    if int(ty[4])==int(df2["smoke"][j]):
        usual_smoker=np.append(usual_smoker,y[j])
stats.ttest_ind(usual_smoker,never_smoke,equal_var=False)
[np.mean(usual_smoker),np.mean(never_smoke)]
stats.ttest_ind(ex_smoker,occ_smoker,equal_var=False)




