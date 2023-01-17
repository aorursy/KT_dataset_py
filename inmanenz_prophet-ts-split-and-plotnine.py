from plotnine import * #ggplot like library for python!!!!
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit #Splitting for time series CV!
from fbprophet import Prophet 
#Read in Data, Create New Frame With Relevant Columns
df=pd.read_csv("../input/avocado.csv")
#Filter To TotalUS and Conventional Avocados for simplicity
df=df.loc[np.logical_and(df.region=="TotalUS",df.type=="conventional"),:]
#Clean up data for analysis
df.Date=pd.to_datetime(df.Date)
date_price_df=df.loc[:,["Date","AveragePrice"]]
date_price_df.columns=["ds","y"]
date_price_df=date_price_df.sort_values("ds").reset_index(drop=True)

#Initialize Split Class, we'll split our data 5 times for cv
tscv = TimeSeriesSplit(n_splits=5)
def pro_ds_data_gen(df,tscv,yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False):
    out_df=pd.DataFrame()
    for i,(train_i,test_i) in enumerate(tscv.split(df)): #For Time Series Split
        #Use indexes to grab the correct data for this split
        train_df=df.copy().iloc[train_i,:]
        test_df=df.copy().iloc[test_i,:]
        #Build our model using prophet and make predictions on the test set
        model=Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        model.fit(train_df)
        predictions=model.predict(test_df)

        #Combine predictions and training into one df for plotting
        pred_df=predictions.loc[:,["ds","yhat"]]
        pred_df["y"]=test_df.y.tolist()
        train_df["train"]="Train"
        pred_df["train"]="Test"
        sub_df=train_df.append(pred_df).reset_index(drop=True)
        sub_df["split"]="Split "+str(i+1)
        sub_df["rmse"]=(np.mean((sub_df.yhat-sub_df.y)**2))**.5 #calculating rmse for the split
        out_df=out_df.append(sub_df).reset_index(drop=True)
    return out_df
year_weak_seas_df=pro_ds_data_gen(
    date_price_df,tscv,yearly_seasonality=True,weekly_seasonality=True
)

(ggplot(year_weak_seas_df,aes("ds","y",color="factor(train)"))+\
 geom_point()+facet_grid('split~.'))+\
labs(title="Train/Test Splits",x="Date",y="Price")+\
scale_x_date(date_breaks="6 months",date_labels =  "%b %Y")

no_seas_df=pro_ds_data_gen(date_price_df,tscv,yearly_seasonality=False,weekly_seasonality=False)
year_seas_df=pro_ds_data_gen(date_price_df,tscv,weekly_seasonality=False)
week_seas_df=pro_ds_data_gen(date_price_df,tscv,yearly_seasonality=False)
df_dict={"year_weak":year_weak_seas_df,"none":no_seas_df,"year":year_seas_df,"week":week_seas_df}


cv_frame=pd.DataFrame()
for name,frame in df_dict.items():
    #grab the one unique rmse for each split
    values_lol=frame.groupby("split").agg({"rmse":"mean"}).values
    values=[item for sublist in values_lol for item in sublist] #returns 2D array with sub-length 1, so we cpllapse
    sub_df=pd.DataFrame({"rmse":values})
    sub_df["model"]=name
    cv_frame=cv_frame.append(sub_df)
(ggplot(cv_frame,aes(x="model",y="rmse",fill="model"))+geom_boxplot())