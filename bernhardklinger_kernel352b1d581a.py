import numpy as np

import pandas as pd

import plotly.express as px

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from datetime import datetime



pd.set_option('display.max_rows', None)



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

train.columns = ["Id","Prov","Ctry","Date","Cases","Death"]

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

test.columns = ["Id","Prov","Ctry","Date"]

test["Cases"]=0

test["Death"]=0

train["Date"]= pd.to_datetime(train.Date,infer_datetime_format=True)

test["Date"]= pd.to_datetime(test.Date,infer_datetime_format=True)

world = pd.read_csv("/kaggle/input/wb0904/WorldBankData.csv")

for col in world.columns[1:]:

    avgcol= world[col].mean()

    world.loc[world[col]==0,col]=avgcol



pop = pd.read_csv("/kaggle/input/pop1204/Population11.04.2020.csv")

sample_sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

old_cols = sample_sub.columns

sample_sub.columns=["Id","Case","Death"]

mysub=sample_sub.set_index('Id')



train["Test"]=0

test["Test"]=1



X_full = pd.concat((train[train.Date < "2020-04-02"], test[test.Date >= "2020-04-02"]),sort=True).reset_index(drop=True)



X_full["Reg"]=X_full["Ctry"]+X_full["Prov"].fillna("None")

pop["Reg"]=pop["Ctry"]+pop["Prov"].fillna("None")



X_full= X_full.merge(world, on=["Ctry"],how="left")

X_full= X_full.merge(pop[["Pop","Reg"]], on=["Reg"],how="left")



X_full.loc[:,"GDPPerc"]= X_full.GDPPerc.astype("float")

X_full.loc[:,"GDPperCapita"]= X_full.GDPperCapita.astype("float")

X_full.loc[X_full.GDPperCapita == 0,"GDPperCapita"]=10000

    

X_full.fillna(0,inplace=True)
train.head(100)
pd.set_option('display.max_rows', None)

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor





def model_fit(model,X,target_col,folds=3):

    

    kf = KFold(folds, shuffle=True, random_state=4991)

    

    drop_cols = set(X.columns).intersection({"Test","Id","Cases","Death","LogD","Week","Day","Date","Prov","FirstDate","DayYear","Sub","Ctry","Pop",target_col})

    

    # create predictors for each region

    

    X_r=X

    X_r =pd.get_dummies(X.copy(),columns=["Reg"])

    

    # add an indicator for countries/states with a significant number of cases

    

    for col in X_r.columns:

        if col[:4]== "Reg_":

            reg = col[4:]

            if X.loc[(X.Reg==reg) & (X.DayYear == (100)),"Cases"].mean() > 250:

                X_r[col+"1"]=X_r["Week1"]*X_r[col]

                X_r[col+"2"]=X_r["Week2"]*X_r[col]

                X_r[col+"3"]=X_r["Week3"]*X_r[col]

                          

    

    # add interactions with health spending indicators 

    # the relationship is very weak but I have kept these features in the model for now

    

    inter_features ={"DollarPPP","GDPPerc","Week1","Week2","Week3","Age65Perc","GDPperCapita","LogPop"}.difference(set(drop_cols)).intersection(set(X.columns)) 

    poly = PolynomialFeatures(degree=2,include_bias=False) 

    inter_cols = poly.fit_transform(X[inter_features])

    X1= pd.DataFrame(inter_cols,columns= poly.get_feature_names(list(inter_features)),index=X.index)                            

    X_r = pd.concat([X1,X_r.drop(columns=inter_features)],axis=1)

    

    X_train = X_r[X_r.Test==0].drop(columns=drop_cols).copy()

    

    y_train = X.loc[X_r.Test==0,target_col]

    model.fit(X_train,y_train)

    X["Pred"] = np.maximum(model.predict(X_r.drop(columns=drop_cols)),0)

    X["Res"]= X.Pred-X[target_col]

    score = (-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))**0.5

    

    return X, score





model =Ridge(alpha=0.0065,random_state=35591,max_iter=10000,fit_intercept=True,normalize=True)





graph = []



# create separate predictions for the public and private leaderboard (i.e. remove values from 1/4 from public submission)



for loop in ["Public","Private"]:

    

    X_all=X_full.copy()

    if loop == "Public":

        start=78+14 # last day of train data 

        val_end=91+14 # end date for validation data (public submission only )

        sub_start=79+14

        sub_end=92+14

        startint= 85 # fit more recent data only - this is a key parameter

    

        X_all.loc[:,"Test"] = (X_all.Date > "2020-04-01") *1

        

    else:

        start=91+15

        sub_start=93+14

        sub_end=999

        X_all.loc[:,"Test"] = (X_all.Date > "2020-04-15") *1 

        val_end=88+14

        startint= 95 # fit more recent data only - this is a key parameter

    

    last_train = "2020-04-15"    

    X_all.loc[(X_all.Date > "2020-04-01") & (X_all.Date <= last_train),"Cases"]=train.loc[

        (train.Date > "2020-04-01") & (train.Date <= last_train),"Cases"].values

    X_all.loc[(X_all.Date > "2020-04-01") & (X_all.Date <= last_train),"Death"]=train.loc[

        (train.Date > "2020-04-01") & (train.Date <= last_train),"Death"].values

    

    # select as start day of the analysis: first day with 50+ cases or a recorded death

    

    X_reg= X_all[~((X_all.Test ==0) & (X_all.Cases < 50) & (X_all.Death<=0))].copy()

    X_reg["Date"]= pd.to_datetime(X_reg.Date,infer_datetime_format=True)

    first_p_map= X_reg[["Reg","Date"]].groupby("Reg").min().to_dict()["Date"]

    X_reg["FirstDate"]=X_reg["Reg"].map(first_p_map)

    

    X_reg["Day"]=(X_reg.Date-X_reg.FirstDate).dt.days

    X_reg["Week"]=X_reg.Day/7

    X_reg["DayYear"]=X_reg.Date.dt.dayofyear

    

    

    X_reg["LogC"]= np.log(X_reg.Cases+1)

    X_reg["LogD"]= np.log(X_reg.Death+1)

    X_reg["LogPop"]=np.log(X_reg.Pop+1)



    X_reg["Week1"]=np.tanh((X_reg.Week)/10)

    X_reg["Week2"]=np.tanh((X_reg.Week)/10*3)

    X_reg["Week3"]=np.tanh((X_reg.Week)/10*5)

    X_reg["Week4"]=np.tanh((X_reg.Week-1)/10)

    X_reg["Week5"]=np.tanh((X_reg.Week-3)/10)

    X_reg["Week6"]=np.tanh((X_reg.Week-5)/10)

    

    

    X_reg["Sub"]=loop

    

    X= X_reg.copy()

    print("\nLast day of year for train",start,": ",X.loc[X.DayYear== start,"Date"].min())

    print("First day of year for test",start+1,": ",X.loc[X.Test == 1,"Date"].min())

    

    

    # fit regressmodel to log of cases to align with evaluation metric

    

    X_res, score = model_fit(model,X.loc[(X.DayYear >= startint)].copy(),"LogC")

    X.loc[(X.DayYear >= startint),"PredC"]=np.exp(X_res.Pred)-1

    

    print("\nCV score: ",score,"\nMean: {:.4f} Std: {:.4f}\n".format(score.mean(), score.std()))

    

    # Scale predicted cases to the number of cases on the final day of the train data 

    # Derive the mortality rate based on deaths and predicted cases at the final day in the train data

    # Model does not allow for expected time lag between cases and deaths, which could be significant limitation

    

    maptab = pd.DataFrame(None,index=list(set(X.Reg)))

    maptab["CaseScaler"]=0

    maptab["MortScaler"]=0

    

    for reg in set(X.Reg):        

        

        Xslice = X[(X.Reg==reg) & (X.Test==0) & (X.DayYear == start)]



        if len(Xslice) > 0:

            if Xslice["PredC"].mean() > 0:

                maptab.loc[reg,"CaseScaler"] = Xslice["Cases"].mean()/Xslice["PredC"].mean()

                # also set a maximum mortality rate to adjust outliers

                maptab.loc[reg,"MortScaler"]= np.minimum(Xslice["Death"].mean()/Xslice["Cases"].mean(),0.10)

        

        # add adjustments for missing data and set minimum mortality rate                

        

        elif len(X_all[(X_all.Reg==reg) & (X_all.Test==0) & (X_all.Cases > 0)]) > 0:

            maxcase = X_all.loc[(X_all.Reg==reg) & (X_all.Test==0) & (X_all.Cases > 0),"Cases"].max()

            maptab.loc[reg,"CaseScaler"]=maxcase/X.loc[(X.Reg==reg) &(X.Test==1),"PredC"].min()

            maptab.loc[reg,"MortScaler"]=0.01

        else:

            maptab.loc[reg,"CaseScaler"]=0

            maptab.loc[reg,"MortScaler"]=0.01

    

    

    X.loc[:,"PredC"]= X.loc[:,"Reg"].map(maptab["CaseScaler"].to_dict())*X.loc[:,"PredC"]

    X["MortS"]=X.loc[:,"Reg"].map(maptab["MortScaler"].to_dict())

    

    # apply a rough adjustment to expected minimum death rate (= rate per number of postive cases, not per population) for lower income countries

    

    X.loc[X.GDPperCapita >=5000,"PredD"]= np.maximum(X.loc[:,"Reg"].map(maptab["MortScaler"].to_dict())*X.loc[:,"PredC"],0.0008*X["PredC"]) 

    X.loc[X.GDPperCapita <5000,"PredD"]= np.maximum(X.loc[:,"Reg"].map(maptab["MortScaler"].to_dict())*X.loc[:,"PredC"],0.002*X["PredC"]) 

    

    # cap overall cases at 20% of population

    # cap mortality at 10% of maximum infected population 

    

    X.fillna(0,inplace=True)

    X.loc[:,"PredC"]=np.minimum(X.Pop * 0.2,X.PredC)

    X.loc[:,"PredD"]=np.minimum(np.minimum(X.Pop * 0.02,X.PredD),0.2*X.PredC)

    

    X["ResLC"] = np.log(X.PredC+1)-X.LogC

    X["ResLD"] = np.log(X.PredD+1)-X.LogD

    X["ResLC2"]=X.ResLC**2

    X["ResLD2"]=X.ResLD**2



    

    if len(graph) == 0: graph=X.copy()

    else: graph = pd.concat([graph,X])

    

    res_def=X[(X.DayYear >= sub_start) & (X.DayYear <= sub_end)].copy()

    mysub = mysub.combine_first(res_def.set_index('Id')[["PredC","PredD"]])

    if loop == "Public": 

        print(X.loc[(X.DayYear > start) & (X.DayYear <=val_end),["ResLC","ResLD","PredC","PredD"]].describe())

        print("\n",loop," submission - Score: ",np.sqrt(X.loc[(X.DayYear > start) & (X.DayYear <=val_end),["ResLC","ResLD"]].var().mean())) 


fig = px.scatter(graph[graph.Reg=="AustriaNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="RussiaNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="NigeriaNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="BarbadosNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="MaldivesNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="SwedenNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="United KingdomNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="USNew York"], x='Date', y='PredD', color="Sub")

fig.show()
fig = px.scatter(graph.loc[graph.Sub=="Private",["PredC","PredD","Date"]].groupby("Date").sum().reset_index(), x='Date', y='PredD')

fig.show()

fig = px.scatter(graph.loc[graph.Sub=="Public",["PredC","PredD","Date"]].groupby("Date").sum().reset_index(), x='Date', y='PredD')

fig.show()
mysub.drop(columns =["Case","Death"],inplace=True)

mysub.insert(0,"Id",mysub.index)

mysub.columns=old_cols

mysub.fillna(0,inplace=True)

mysub.to_csv('submission.csv',index=False)
train.head(200)