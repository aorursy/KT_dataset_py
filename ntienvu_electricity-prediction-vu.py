





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt



from dateutil.relativedelta import relativedelta


## 2. Adding some datetime related features



def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Is_year_start","Is_month_start",

                "Hour", "Minute"]

    #features = ["Year", "Week", "Dayofyear","Dayofweek",

                #"Is_year_end", "Is_year_start",

                #"Hour"]

    one_hot_features = ["Month", "Dayofweek"]

    one_hot_features=[]

    

    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        else:

            df[feature] = new_column

    

  

    return df




df = pd.read_csv("../input/train_electricity.csv")



print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")

    

    
test_df = pd.read_csv("../input/test_electricity.csv")

test_df = add_datetime_features(test_df)

print(test_df.columns)





df = add_datetime_features(df)







#remove outlier

df = df[ (df['Consumption_MW']>1000)  & (df['Consumption_MW']<15000)]




## 3. Split data into train / validation (leaving the last six months for validation)





month_t1=0

month_t2=0

month_start=0

threshold_1 = df['Datetime'].max() + relativedelta(months=-month_t1)  # Here we set the 6 months threshold

threshold_2 = df['Datetime'].max() + relativedelta(months=-month_t2)  # Here we set the 6 months threshold

threshold_0 = df['Datetime'].min() + relativedelta(months=month_start)  # Here we set the 6 months threshold



#threshold_3 = df['Datetime'].min() + relativedelta(months=month_start)  # Here we set the 6 months threshold



#train_df = df[ ( df['Datetime'] < threshold_1 ) |  (( df['Datetime'] < threshold_3 ) & (df['Datetime'] > threshold_2) )]



train_df = df[ ( ( df['Datetime'] > threshold_0) & ( df['Datetime'] < threshold_1)) | (df['Datetime'] > threshold_2 ) ]



valid_df = df[(df['Datetime'] >= threshold_1) & (df['Datetime'] <= threshold_2)]





#train_df, valid_df = train_test_split(df, test_size=0.1)



#print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")

#print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")



label_col = "Consumption_MW"  # The target values are in this column

#to_drop = [label_col, "Date", "Datetime"]  # Columns we do not need for training

to_drop = ["Date", "Datetime","Dayofyear"]  # Columns we do not need for training





to_drop_train=[label_col]+to_drop







print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")

print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")



model = xgb.XGBRegressor(n_estimators=500,min_child_weight=1.77, max_depth=5,gamma=9.107,learning_rate=0.2509,

 subsample=0.6383, colsample_bylevel=0.7685, reg_lambda=10,n_jobs=-1, random_state=14)# rand 435



bst = model.fit(

    train_df.drop(to_drop_train, axis=1),train_df[label_col],

    eval_set=[(valid_df.drop(to_drop_train, axis=1), valid_df[label_col])],eval_metric='rmse'

    #,early_stopping_rounds=30

)



pred_score = bst.predict(valid_df.drop(to_drop_train, axis=1))

rms=np.sqrt(np.mean((pred_score-valid_df[label_col].values)**2))


fig,ax=plt.subplots(figsize=(5,15))

xgb.plot_importance(bst,ax=ax)


ypredicted = bst.predict(test_df.drop(to_drop, axis=1))



plt.plot(ypredicted)
ratio=1.01*1.005

yfinal=ypredicted*ratio

pred = pd.Series(yfinal, test_df.Date).rename('Consumption_MW').to_frame()

pred.to_csv('final_prediction.csv')