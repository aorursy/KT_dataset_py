import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
sub_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")
train_data.head()
plt.figure(figsize=(20,20))
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter("%m")
# Firstly getting knowledge of what the unique countries we have.
unique_countries = train_data["Country_Region"].unique()
# So there are 187 total unqiue countries we have . we will take into consideration all of these and prepare a dataframe 
# of has three columns viz (country name ,confirmed , fatalities) . 
fig,ax = plt.subplots()
fig.figsize = (20,20)
# Making a dictionary to hold all countries as  keys .
Confirmed = [c for i,c in enumerate(train_data["TargetValue"]) if i%2==0]
Fatalities  =  [c for i,c in enumerate(train_data["TargetValue"]) if i%2!=0]
countries = [c for i,c in enumerate(train_data["Country_Region"]) if i%2==0]
dates = [c for i,c in enumerate(train_data["Date"]) if i%2==0]

# DataFrame Made such that we don't have to apply transformations in further visualizations.
df_ccf = pd.DataFrame(columns=["Country","Confirmed","Fatalities","Date"])
df_ccf["Country"] = countries
df_ccf["Confirmed"] = Confirmed
df_ccf["Fatalities"] = Fatalities
df_ccf["Date"] = dates

df_ccf = df_ccf.set_index("Country")
# Visualization of Confirmed Cases of 10 Countries.

for i,uc in enumerate(unique_countries):
    if i<5:
        ax.plot(df_ccf.loc[uc,"Date"],df_ccf.loc[uc,"Confirmed"],label=uc)
    
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
plt.legend()
plt.show()

# As we can see we are not able to find any pattern in the data so lets some another visualizations.
# Just taking top 5 Countries with the Confirmed Cases and Fatalities being High
top20_c = {e:np.sum(df_ccf.loc[e,"Confirmed"]) for e in unique_countries}
top20_c = sorted(top20_c.items(),key=lambda x:x[1],reverse=True)
df_ccf.groupby("Country").sum().loc["US"]
top20_f = {e:np.sum(df_ccf.loc[e,"Fatalities"]) for e in unique_countries}
for e in top20_f:
    print(e)
top20_f = sorted(top20_f.items(),key=lambda x:x[1],reverse=True)
top20_f = [v for (k,v) in top20_f[:20]]
reduced_df = df_ccf.groupby("Country").sum().sort_values(by="Confirmed",ascending=False)[:20]
plt.figure(figsize=(20,8))
reduced_df["Confirmed"].plot(kind="bar",color="green",edgecolor="black",label="Confirmed")
reduced_df["Fatalities"].plot(kind="bar",color="darkred",edgecolor="red",label="Fatalities")
plt.suptitle("Confirmed Cases And Fatalities Top-20 Countries",fontsize=30,color="brown",fontfamily="italics")
plt.legend()
plt.show()
plt.figure(figsize=(10,7))
color = plt.cm.get_cmap("viridis",20)
color = color.colors
for i,(k,v) in enumerate(top20_c):
    if i<20:
        plt.bar(i,height=v,label=k,color=color[i])
plt.plot(top20_f,c="darkred",marker="s")
plt.legend()
plt.title("Top-20 Country When Confirmed Cases",fontsize=28,fontstyle="oblique",color="steelblue",fontfamily="cursive")
plt.show()
plt.plot(range(20),top20_f,marker="o")
# Here we can Visualize that #Deaths are following the same order with Country meaning more the Confirmed Cases
# in the Country more the Deaths hence they should be directly Proportional To each other.
# Lets Visualize the Confirmed/Fatalities ratio of top-20 country to support our statement,
ratio = [c[1]/f for c,f in zip(top20_c,top20_f) if f!=0]
plt.plot(range(len(ratio)),ratio,linewidth=2,marker="s") # Improve COloring Scheme
plt.show()

# here our assumption of Fatalities changing as Confirmed Cases Fails because when that would be the case then
# we could have got the ratio to be constant or almost constant but here it varies drastically.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Pipeline and GridSearch
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import dates
features = train.drop(["TargetValue"],axis=1)
features["Date"] = dates.datestr2num(features["Date"])
labels = train["TargetValue"]
rfr_new = RandomForestRegressor(n_estimators=300,max_features="sqrt")
X_train,X_test,Y_train,Y_test = train_test_split(features,labels,test_size=0.2,random_state=42)
X_train
rfr_new.fit(X_train,Y_train)
rfr_new.predict(X_test)
Y_train_pred = rfr_new.predict(X_train)
print(f"R2 Score: {round(r2_score(Y_train,Y_train_pred),2)*100} %")
print(f"MSE: {mean_squared_error(Y_train,Y_train_pred)}")
Y_test_pred = rfr_new.predict(X_test)
print(f"R2 Score: {round(r2_score(Y_test,Y_test_pred),2)*100} %")
print(f"MSE: {mean_squared_error(Y_test,Y_test_pred)}")
# Getting predictions from current model for test.csv
test = test_data.drop(["ForecastId","Country_Region","Target","Province_State","County"],axis=1)
test["Date"] = pd.to_datetime(test["Date"]).dt.strftime("%Y%m%d").astype(int)
preds = pipeline.predict(test)
preds = [int(e) for e in preds]
outputs = pd.DataFrame({"Id":test.index,"TargetValue":preds})
a=outputs.groupby(["Id"])["TargetValue"].quantile(q=0.05).reset_index()
b= outputs.groupby(["Id"])["TargetValue"].quantile(q=0.5).reset_index()
c = outputs.groupby(["Id"])["TargetValue"].quantile(q=0.95).reset_index()
a.columns = ["Id","q0.05"]
b.columns = ["Id","q0.5"]
c.columns = ["Id","q0.95"]
a = pd.concat([a,b["q0.5"],c["q0.95"]],1)
a["q0.05"] = a["q0.05"].clip(0,10000)
a["q0.5"] = a["q0.5"].clip(0,10000)
a["q0.95"] = a["q0.95"].clip(0,10000)
a["Id"] = a["Id"]+1
submission = pd.melt(a,id_vars=["Id"],value_vars=["q0.05","q0.5","q0.95"])
submission["variable"] = submission["variable"].str.replace("q","",regex=False)
submission["ForecastId_Quantile"] = submission["Id"].astype(str)+"_"+submission["variable"]
submission["TargetValue"] = submission["value"]
submission = submission[["ForecastId_Quantile","TargetValue"]]
submission.reset_index(drop=True,inplace=True)
submission.head()
submission.to_csv("submission.csv",index=False)
