import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df_coron=pd.read_csv("../input/wind-coron/coron_all.csv",parse_dates=["time"]).set_index("time")
df_vigo=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",
                           usecols=['mod_o','wind_gust_o','mod_4K',"mod_36K","datetime","metar_o"],index_col="datetime",parse_dates=True)
#mean speed ten minutes around exact hour
df={}
df["left"]=df_coron["spd_o"].resample("20min").mean()
df["right"]=df_coron["spd_o"].resample("20min",label='right',closed="right").mean()
df_coron["spd_o_mean"]=(df["left"]+df["right"])/2

#median speed one before
df_coron["spd_o_median"]=df_coron["spd_o"].resample("h").quantile(.5)

#display correlations and new columns
df_fil={}

(df_coron[df_coron.mod_p>12])[["spd_o",'mod_p', "spd_o_median","spd_o_mean"]].dropna().describe()



df_coron[["spd_o",'mod_p', "spd_o_median","spd_o_mean"]][df_coron.mod_p>12].describe()

for speed in range (0,10,2):
    
    print("wind speed predicted > {} m/s".format(speed))
    print((df_coron[["spd_o",'mod_p', "spd_o_median","spd_o_mean"]][df_coron.mod_p>speed]).corr().mod_p)   
    print("**************************")
    

for speed in range (0,10,2):
    
    print("wind speed predicted > {} m/s".format(speed))
    print(((df_coron[df_coron.mod_p>speed])[["spd_o",'mod_p', "spd_o_median","spd_o_mean"]]).corr().mod_p)   
    print("**************************")
fig = px.scatter(df_coron.dropna(), x='mod_p',y='spd_o', trendline="ols",)
fig.show()
fig = px.density_contour(df_coron, x="mod_p", y="spd_o")
fig.show()
fig = px.scatter(df_coron.dropna(), x='mod_p',y='spd_o_median', trendline="ols")
fig.show()

g=df_coron.spd_o.plot(kind="hist", density=True,grid=True)
df_coron[["mod_p","spd_o","spd_o_median","spd_o_mean"]].describe()
formula='spd_o ~ mod_p'
model=sm.ols(formula=formula,data=df_coron)
fitted = model.fit()
print(fitted.summary())

res_anal=pd.DataFrame(data={"residuals":fitted.resid.values,"std_residuals":StandardScaler().fit_transform(fitted.resid.values.reshape(-1,1)).flatten()
                  ,"pred_spd":fitted.predict(),})
g1=res_anal[["residuals","std_residuals"]].plot(kind="kde",grid=True, title="Coron residuals")

g=res_anal.std_residuals.plot(grid=True)
df_coron["dif_spd_o"]=df_coron["mod_p"]-df_coron["spd_o"]
df_coron["dif_spd_median"]=df_coron["mod_p"]-df_coron["spd_o_median"]
df_coron["dif_spd_mean"]=df_coron["mod_p"]-df_coron["spd_o_mean"]

df_coron["residuals_linear_model"]=fitted.resid
g=df_coron[["dif_spd_o","dif_spd_median","dif_spd_mean","residuals_linear_model"]].plot(kind="box",figsize=(10,10),grid=True,notch=5)
df_coron[["dif_spd_o","dif_spd_median","dif_spd_mean","residuals_linear_model"]].describe()
fig = px.scatter(res_anal,x="pred_spd", y="std_residuals", trendline="ols",)
fig.show()
df_vigo.describe()
df_vigo[["mod_4K","mod_36K","mod_o"]].corr()
g=sns.jointplot(x="mod_4K", y="mod_o", data=df_vigo, kind="reg", )
formula='mod_o ~ mod_4K'
model=sm.ols(formula=formula,data=df_vigo)
fitted = model.fit()
print(fitted.summary())
res_anal=pd.DataFrame(data={"residuals":fitted.resid.values,"std_residuals":StandardScaler().fit_transform(fitted.resid.values.reshape(-1,1)).flatten()
                  ,"pred_spd":fitted.predict(),})
g1=res_anal[["residuals","std_residuals"]].plot(kind="kde",grid=True,title="Vigo residuals")
g=res_anal.std_residuals.plot(grid=True,title="Vigo residuals_std 4Km model")
df_vigo["dif_4K"]=df_vigo["mod_o"]-df_vigo["mod_4K"]
df_vigo["dif_36K"]=df_vigo["mod_o"]-df_vigo["mod_36K"]
df_vigo["residuals_linear_model_4K"]=fitted.resid
g=df_vigo[["dif_4K","dif_36K","residuals_linear_model_4K"]].plot(kind="box",grid=True,title="Vigo differences")
df_vigo[["dif_4K","dif_36K","residuals_linear_model_4K"]].describe()