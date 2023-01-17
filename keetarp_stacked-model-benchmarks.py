# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def en_model (X,y):

    # get the packages 
    import numpy as np
    import pandas as pd

    
    # splitting in train and test 
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# import for linear regression

    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
        
    import scipy.stats as stats
    import sklearn
    from sklearn.metrics import r2_score
    from math import sqrt
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    rmse=sqrt(mse)
    Rsquare=r2_score(y_test, y_pred)
  



# import decision tree and predict 
    from sklearn.tree import DecisionTreeRegressor
    tree_regressor = DecisionTreeRegressor(criterion="mse",max_features="auto")

    tree_regressor.fit(x_train,y_train)
    y_tree_pred = tree_regressor.predict(x_test)
    mse_tree = sklearn.metrics.mean_squared_error(y_test, y_tree_pred)
    rmse_tree=sqrt(mse_tree)
    Rsquare_tree=r2_score(y_test, y_tree_pred)
    
    
        
    # variable importance : 
        



# Random Forest 
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(x_train, y_train)
    
    y_rf_pred = rf.predict(x_test)
    mse_rf = sklearn.metrics.mean_squared_error(y_test, y_rf_pred)
    rmse_rf=sqrt(mse_rf)
    Rsquare_rf=r2_score(y_test, y_rf_pred)





# GBM
    from sklearn.ensemble import GradientBoostingRegressor

    params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
    gbm = GradientBoostingRegressor(**params).fit(x_train, y_train)    


    y_gbm_pred = gbm.predict(x_test)
    mse_gbm = sklearn.metrics.mean_squared_error(y_test, y_gbm_pred)
    rmse_gbm=sqrt(mse_gbm)
    Rsquare_gbm=r2_score(y_test, y_gbm_pred)
    
    
    
    
# XGB
        # Let's try XGboost algorithm to see if we can get better results
    import xgboost 
    
    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
    model=xgb.fit(x_train,y_train)    
    
    y_xgb_pred = xgb.predict(x_test)
    mse_xgb = sklearn.metrics.mean_squared_error(y_test, y_xgb_pred)
    rmse_xgb=sqrt(mse_xgb)
    Rsquare_xgb=r2_score(y_test, y_xgb_pred)
        

    # getting variable importance plots 


    imp_df=pd.DataFrame(columns=['Variable','Relative_importance'])
    imp_df['Variable']=x_train.columns     
    imp_df['Relative_importance']=tree_regressor.feature_importances_
    imp_df=imp_df.sort_values(['Relative_importance'], ascending=[False])
    imp_df=imp_df.iloc[:5]
          
          
    imp_df_rf=pd.DataFrame(columns=['Variable','Relative_importance'])
    imp_df_rf['Variable']=x_train.columns     
    imp_df_rf['Relative_importance']=rf.feature_importances_
    imp_df_rf=imp_df_rf.sort_values(['Relative_importance'], ascending=[False])      
    imp_df_rf=imp_df_rf.iloc[:5]
          
    
    imp_df_gbm=pd.DataFrame(columns=['Variable','Relative_importance'])
    imp_df_gbm['Variable']=x_train.columns     
    imp_df_gbm['Relative_importance']=gbm.feature_importances_
    imp_df_gbm=imp_df_gbm.sort_values(['Relative_importance'], ascending=[False])      
    imp_df_gbm=imp_df_gbm.iloc[:5]
 
    imp_df=imp_df.sort_values(['Relative_importance'],ascending=[True])
    imp_df_rf=imp_df_rf.sort_values(['Relative_importance'],ascending=[True])
    imp_df_gbm=imp_df_gbm.sort_values(['Relative_importance'],ascending=[True])
    
    
    # plot all variable importances                    
                       
    import matplotlib.pyplot as plt
    
    %matplotlib inline
    # Set the style
    plt.style.use('fast')    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    imp_df.plot(x='Variable',y='Relative_importance',kind='barh',legend=False,title='DecisionTree',ax=axes[0,0])
    imp_df_rf.plot(x='Variable',y='Relative_importance',kind='barh',legend=False,title='RandomForest',ax=axes[0,1])
    imp_df_gbm.plot(x='Variable',y='Relative_importance',kind='barh',legend=False,title='GBM',ax=axes[1,0])
    fig.tight_layout()
 
    
    
    en_pred=(y_rf_pred+y_tree_pred+y_gbm_pred+y_xgb_pred)/4
    mse_ens = sklearn.metrics.mean_squared_error(y_test, en_pred)
    rmse_ens=sqrt(mse_ens)
    Rsquare_ens=r2_score(y_test, en_pred)
    
    import pandas as pd
    list=[["Linear Regression",round(Rsquare,2),round(rmse,2),round(mse,2)]]
    list.append(["Decision Tree",round(Rsquare_tree,2),round(rmse_tree,2),round(mse_tree,2)])
    list.append(["Random Forest",round(Rsquare_rf,2),round(rmse_rf,2),round(mse_rf,2)])
    list.append(["GBM",round(Rsquare_gbm,2),round(rmse_gbm,2),round(mse_gbm,2)])
    list.append(["XGB",round(Rsquare_xgb,2),round(rmse_xgb,2),round(mse_xgb,2)])
    list.append(["StackedEnsemble",round(Rsquare_ens,2),round(rmse_ens,2),round(mse_ens,2)])

    
    df=pd.DataFrame(list,columns=['Model','RSQUARE','RMSE','MSE'])
    print(df)
    
    
    
    
    
house = pd.read_csv("../input/kc_house_data.csv")
house.head()
del house['id']
del house['date']



X=house.iloc[:,0:19]
X.head()
y=X.pop('price')
#launch function
en_model(X,y)