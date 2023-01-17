import numpy as np

import pandas as pd

import plotly.express as px

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold, cross_val_score

from datetime import datetime



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

train.columns = ["Id","Prov","Ctry","Date","Cases","Death"]

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

test.columns = ["Id","Prov","Ctry","Date"]

test["Cases"]=0

test["Death"]=0

train["Date"]= pd.to_datetime(train.Date,infer_datetime_format=True)

test["Date"]= pd.to_datetime(test.Date,infer_datetime_format=True)

world = pd.read_csv("/kaggle/input/corona-wb/WorldBankData.csv")

pop = pd.read_csv("/kaggle/input/coronawb/Population.csv")

sample_sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

old_cols = sample_sub.columns

sample_sub.columns=["Id","Case","Death"]

mysub=sample_sub.set_index('Id')



train["Test"]=0

test["Test"]=1



X_full = pd.concat((train[train.Date < "2020-03-19"], test[test.Date > "2020-03-18"]),sort=True).reset_index(drop=True)



X_full["Reg"]=X_full["Ctry"]+X_full["Prov"].fillna("None")

pop["Reg"]=pop["Ctry"]+pop["Prov"].fillna("None")



X_full= X_full.merge(world, on=["Ctry"],how="left")

X_full= X_full.merge(pop[["Pop","Reg"]], on=["Reg"],how="left")



X_full.loc[:,"GDPPerc"]= X_full.GDPPerc.astype("float")

X_full.loc[:,"GDPperCapita"]= X_full.GDPperCapita.astype("float")

    

X_full.fillna(0,inplace=True)
pd.set_option('display.max_rows', None)

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



def model_fit(model,X,target_col,folds=3):

    

    kf = KFold(folds, shuffle=True, random_state=4991)

    

    drop_cols = set(X.columns).intersection({"Test","Id","Cases","Death","LogD","Day","Week","Date","Prov","FirstDate","DayYear","Sub","Ctry","Pop",target_col})

    

    # create predictors for each region

    

    X_r=X

    X_r =pd.get_dummies(X.copy(),columns=["Reg"])

                

    for col in X_r.columns:

        if col[:4]== "Reg_":

            reg = col[4:]

            if X.loc[(X.Reg==reg) & (X.DayYear == 85),"Cases"].mean() > 500:

                X_r[col+"1"]=X_r["Week1"]*X_r[col]

                X_r[col+"2"]=X_r["Week2"]*X_r[col]

                X_r[col+"3"]=X_r["Week3"]*X_r[col]

                          

    

    # add interactions with health spending indictors

    # the relationship is very weak but I have kept the features in the model for now

    

    inter_features ={"DollarPPP","GDPPerc","Week1","Week2","Week3","Age65Perc","GDPperCapita"}.difference(set(drop_cols)).intersection(set(X.columns)) 

    poly = PolynomialFeatures(interaction_only=True,include_bias=False) 

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





model =Ridge(alpha=0.01,random_state=35591,max_iter=10000,fit_intercept=True,normalize=True)

graph = []



# separate predictions for public and private leaderboard (i.e. remove values post 3/18 from public submission)



for loop in ["Public","Private"]:

    

    X_all=X_full.copy()

    if loop == "Public":

        start=78 # last day of train data 

        val_end=91 # end date for validation data (public submission only )

        sub_start=79

        sub_end=92

        X_all.loc[:,"Test"] = (X_all.Date > "2020-03-18") *1

        

    else:

        start=91

        sub_start=93

        sub_end=999

        X_all.loc[:,"Test"] = (X_all.Date > "2020-03-31") *1 

        val_end=88

    X_all.loc[(X_all.Date > "2020-03-18") & (X_all.Date < "2020-04-01"),"Cases"]=train.loc[train.Date > "2020-03-18","Cases"].values

    X_all.loc[(X_all.Date > "2020-03-18") & (X_all.Date < "2020-04-01"),"Death"]=train.loc[train.Date > "2020-03-18","Death"].values

    

    # start day for eac region predection = first day with 5+ cases

    X_reg= X_all[~((X_all.Test ==0) & (X_all.Cases < 5))].copy()

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

    X_reg["Week2"]=np.tanh((X_reg.Week)/10*2)

    X_reg["Week3"]=np.tanh((X_reg.Week)/10*4)

    X_reg["Week4"]=np.tanh((X_reg.Week-1)/10)

    X_reg["Week5"]=np.tanh((X_reg.Week-2)/10)

    X_reg["Week6"]=np.tanh((X_reg.Week-4)/10)

   

    

    X_reg["Sub"]=loop

    

    X= X_reg.copy()

    print("Last day of year for train",start,": ",X.loc[X.DayYear== start,"Date"].min())

    print("First day of year for test",start+1,": ",X.loc[X.Test == 1,"Date"].min())

    

    startint= 60 # fit data from start of March only

    

    # fit model to log of cases to align with evaluation metric

    

    X_res, score = model_fit(model,X.loc[(X.DayYear >= startint)].copy(),"LogC")

    X.loc[(X.DayYear >= startint),"PredC"]=np.exp(X_res.Pred)-1

    

    print("\nCV score: ",score,"\nMean: {:.4f} Std: {:.4f}\n".format(score.mean(), score.std()))

    

    # Scale predicted cases to the number of cases on final day of the train data 

    # Derive the mortality rate based on deaths and predicted cases at the final day in the train data

    

    maptab = pd.DataFrame(None,index=list(set(X.Reg)))

    maptab["CaseScaler"]=0

    maptab["MortScaler"]=0

    

    for reg in set(X.Reg):        

        

        Xslice = X[(X.Reg==reg) & (X.Test==0) & (X.DayYear >= start)]



        if len(Xslice) > 0:

            if Xslice["PredC"].mean() > 0:

                maptab.loc[reg,"CaseScaler"] = Xslice["Cases"].mean()/Xslice["PredC"].mean()

                # also set a maximum mortality rate to adjust outliers

                maptab.loc[reg,"MortScaler"]= np.minimum(Xslice["Death"].mean()/Xslice["Cases"].mean(),0.1)

        

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

    X.loc[:,"PredD"]= np.maximum(X.loc[:,"Reg"].map(maptab["MortScaler"].to_dict())*X.loc[:,"PredC"],0.005*X["PredC"]) 

    

    # cap overall cases at 20% of population

    # cap mortality at 10% of maximum infected population 

    

    X.fillna(0,inplace=True)

    X.loc[:,"PredC"]=np.minimum(X.Pop * 0.02,X.PredC)

    X.loc[:,"PredD"]=np.minimum(X.Pop * 0.002,X.PredD)

    

    X["ResLC"] = np.log(X.PredC+1)-X.LogC

    X["ResLD"] = np.log(X.PredD+1)-X.LogD

    

    if len(graph) == 0: graph=X.copy()

    else: graph = pd.concat([graph,X])

    

    res_def=X[(X.DayYear >= sub_start) & (X.DayYear <= sub_end)].copy()

    mysub = mysub.combine_first(res_def.set_index('Id')[["PredC","PredD"]])

    if loop == "Public": 

        print(X.loc[(X.DayYear > start) & (X.DayYear <=val_end),["ResLC","ResLD","PredC","PredD"]].describe())

        print(loop," Score: ",np.sqrt(X.loc[(X.DayYear > start) & (X.DayYear <=val_end),["ResLC","ResLD"]].var().mean()))    

    
graph["PredL"]=np.log(X.PredD+1)

fig = px.scatter(graph[graph.Reg=="USNew York"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="ItalyNone"], x='Date', y='PredD', color="Sub")

fig.show()

fig = px.scatter(graph[graph.Reg=="United KingdomNone"], x='Date', y='PredD', color="Sub")

fig.show()
fig = px.scatter(graph.loc[graph.Sub=="Private",["PredC","PredD","Date"]].groupby("Date").sum().reset_index(), x='Date', y='PredD')

fig.show()
mysub.drop(columns =["Case","Death"],inplace=True)

mysub.insert(0,"Id",mysub.index)

mysub.columns=old_cols

mysub.fillna(0,inplace=True)

mysub.to_csv('submission.csv',index=False)