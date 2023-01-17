import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import os
print(os.listdir("../input"))
# Import all of them 
sales=pd.read_csv("../input/sales_train.csv")

# settings
import warnings
warnings.filterwarnings("ignore")

item_cat=pd.read_csv("../input/item_categories.csv")
item=pd.read_csv("../input/items.csv")
sub=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")
shops.describe().T
N_SESSIONS = 34
# Now we convert the raw sales data to monthly sales, broken out by item & shop
sales_piv= sales.pivot_table(index=['item_id','shop_id'], columns='date_block_num',values='item_cnt_day',aggfunc=np.sum,fill_value=0).reset_index()
sales_piv.head()
seasonsindex=sales_piv.drop(['item_id','shop_id'],axis=1).sum()/sales_piv.drop(['item_id','shop_id'],axis=1).sum().mean()
# Merge the monthly sales data to the test data
Test=pd.merge(test, item, how='inner', on='item_id')
Test = pd.merge(Test, sales_piv, on=['item_id','shop_id'], how='left').fillna(0)
#add category
Testcat=Test['item_category_id']
Test = Test.drop(labels=['ID', 'shop_id', 'item_id','item_name','item_category_id'], axis=1).T


# times series ARIMA uitgeschakeld
#Ttrain=Test.append(Test.diff(1).dropna())
#Ttrain=Ttrain.append(pd.rolling_mean(Test,12)).dropna()
#Ttrain=Ttrain.T
#cluster in 9 groups
#from sklearn.cluster import KMeans
#clu=KMeans(n_clusters=9)
#Ttrain['clu']=clu.fit_predict(Ttrain)
#Ttrain['cat']=Testcat

Ttrain=Test.T
ALPHA = 0.06
BETA = 0.98
GAMMA= 0.48

def create_filtered_prediction(train_ts, alpha, beta):
    train_time_filtered_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    train_time_filtered_ts[0, :] = train_ts[0, :N_SESSIONS]
    train_memontum_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    prediction_ts = np.zeros((train_ts.shape[0], N_SESSIONS+1), dtype=np.float)
    for i in range(1, N_SESSIONS):
        train_time_filtered_ts[:, i] = (1-alpha) * (train_time_filtered_ts[:, i-1] + \
                                                    train_memontum_ts[:, i-1]) + alpha * train_ts[:, i]
        train_memontum_ts[:, i] = (1-beta) * train_memontum_ts[:, i-1] + \
                                  beta * (train_time_filtered_ts[:, i] - train_time_filtered_ts[:, i-1])
        prediction_ts[:, i+1] = train_time_filtered_ts[:, i] + train_memontum_ts[:, i]
    return prediction_ts


def trixps(y,a0,b0,alpha,beta,gamma,initialSeasonalIndices,period,m):
    #St = np.zeros((len(y),len(y[0])), dtype=np.float)
    St = Bt =It= Ft = np.zeros((len(y),len(y[0]) +m ), dtype=np.float)
    #print(y.shape,St.shape)
    #Initialize base values
    #St[:,1] = a0;
    #Bt[:,1] = b0;
     
    for i in range(period):
        It[:,i] = initialSeasonalIndices[i]
    Ft[:,m] = (St[:,0] + (m * Bt[:,0])) * It[:,0] #;//This is actually 0 since Bt[0] = 0
    Ft[:,m + 1] = (St[:,1] + (m * Bt[:,1])) * It[:,1] #;//Forecast starts from period + 2

    #//Start calculations
    for i in range( 2,len(y[0])):
        #//Calculate overall smoothing
        if (i - period) >= 0:
            St[:,i] = alpha * y[:,i] / It[:,i - period] + (1.0 - alpha) * (St[:,i - 1] + Bt[:,i - 1])
        else:
            St[:,i] = alpha * y[:,i] + (1.0 - alpha) * (St[:,i - 1] + Bt[:,i - 1])
        #//Calculate trend smoothing
            Bt[:,i] = gamma * (St[:,i] - St[:,i - 1]) + (1 - gamma) * Bt[:,i - 1]

        #//Calculate seasonal smoothing
        if (i - period) >= 0:
            It[:,i] = beta * y[:,i] / St[:,i] + (1.0 - beta) * It[:,i - period]
        #//Calculate forecast
        if  (i + m) >= period:
            Ft[:,i + m] = (St[:,i] + (m * Bt[:,i])) * It[:,i - period + m]
    return Ft

#predictions = create_filtered_prediction(Ttrain.values, ALPHA, BETA)
p=trixps(Ttrain.values,0.5,1,ALPHA,BETA,GAMMA,seasonsindex,12,1)
print( p.shape,Ttrain.shape )
knownmonth=32
unknownmonth=33
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print(Ttrain.shape)
yo=Ttrain.iloc[:,unknownmonth]
yo.colomns=[unknownmonth,'di','ma']
#print(yo)
Ai=0.02
Gi=0.3
Bi=0.5
p=trixps(Ttrain.values[:,:unknownmonth],0.5,1,Ai,Bi,Gi,seasonsindex,12,1)
errorm = rmse(yo[unknownmonth], p[:,unknownmonth])
errormb=errorm
errorma=errorm
directa=0.01
directg=0.09
directb=0.095
for xi in range(5):
    #ALPHA=Ai/100
    if abs(directb)>0.001:
        for yi in range(3):
            #BETA=Bi
            if abs(directg)>0.001:
                for zi in range(3):
                    #GAMMA=Gi
                    #p= create_filtered_prediction(Ttrain.values, ALPHA, BETA)
                    p=trixps(Ttrain.values[:,:unknownmonth],0.5,1,Ai,Bi,Gi,seasonsindex,12,1)
                    error = rmse(yo[unknownmonth], p[:,unknownmonth])
                    print('alpha %.3f beta %.3f gamma %.3f month %d directg %.3f directb %.3f- Error %.2f' % (Ai,Bi,Gi,knownmonth,directg,directb,error) ) 
                    errorbigger=(error>errorm)
                    if errorbigger:
                        directg=directg*-1.9
                    else:
                        directg=directg*0.5                
                    Gi+=directg
                    errorm=error
            else:
                p=trixps(Ttrain.values[:,:unknownmonth],0.5,1,Ai,Bi,Gi,seasonsindex,12,1)
                error = rmse(yo[unknownmonth], p[:,unknownmonth])
                print('alpha %.3f beta %.3f gamma %.3f month %d directg %.3f directb %.3f- Error %.2f' % (Ai,Bi,Gi,knownmonth,directg,directb,error) )                 
            errorbiggerb=(error>errormb)
            if errorbiggerb:
                directb=directb*-1.7                
            else:
                directb=directb*0.5
            errormb=error
            errorm=error
            Bi+=directb                
    else:
        p=trixps(Ttrain.values[:,:unknownmonth],0.5,1,Ai,Bi,Gi,seasonsindex,12,1)
        error = rmse(yo[unknownmonth], p[:,unknownmonth])
        print('alpha %.3f beta %.3f gamma %.3f month %d directg %.3f directb %.3f- Error %.2f' % (Ai,Bi,Gi,knownmonth,directg,directb,error) )                 

    errorbiggera=(error>errorma)
    if errorbiggera:
        directa=directa*-1.7                
    else:
        directa=directa*0.5
    errorma=error
    errormb=error
    errorm=error
    Ai+=directa               

            #predictions
    Ai=0.021
    Bi=0.599
    Gi=0.175
    p=trixps(Ttrain.values[:,:unknownmonth+1],0.5,1,Ai,Bi,Gi,seasonsindex,12,1)
        #error = rmse(yo[unknownmonth], p[:,unknownmonth])
sub["item_cnt_month"] = np.clip(p[:, unknownmonth], 0, 20)
sub.to_csv("submitalpha25bet057.csv",index=False)
                  #.format(ALPHA))
TEXPtr=Ttrain.T.append(pd.DataFrame(p).T  )
TEXPtr
def regresseer(meltx,month):
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor,GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression,RidgeCV,ARDRegression,ElasticNet,PassiveAggressiveClassifier,HuberRegressor,TheilSenRegressor,LarsCV,Lasso,RANSACRegressor,LassoLarsIC,LogisticRegression,OrthogonalMatchingPursuit,ElasticNetCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.isotonic import IsotonicRegression
    xtr = pd.merge( meltx.iloc[:,:month],meltx.iloc[:,int(month+1):],left_index=True, right_index=True )
    ytr = meltx.iloc[:,month]
    print(xtr.shape,ytr.shape)
    #train = meltx[meltx['date_block_num'] < month]
    #val = meltx[meltx['date_block_num'] == month]
    #print(train.shape,train.columns,val.shape)
    #xtr, xts = train.drop(['item_cnt_day'], axis=1), val.drop(['item_cnt_day'], axis=1)
    #ytr, yts = train['item_cnt_day'].values, val['item_cnt_day'].values
    #poly = PolynomialFeatures(1)
    #xtr=poly.fit_transform(xtr)
    #xts=poly.fit_transform(xts)
    #mdl =AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=7, random_state=1)  #0.64
    mdl = KNeighborsRegressor()  #0.64
    mdl = ExtraTreesRegressor(n_estimators=10)  #0.65
    #mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0) #0.66
    #mdl = MLPRegressor() #0.68
    #mdl =DecisionTreeRegressor(max_depth=8)  #0.71
    #mdl=TheilSenRegressor()  #0.72   
    #mdl=RANSACRegressor() #0.78
    #mdl=HuberRegressor() #0.83    
    #mdl = LinearRegression()  #1.03
    #mdl=ElasticNet(random_state=0) #1;04    
    #mdl = GradientBoostingRegressor(loss='quantile') #1.06
    #mdl = LarsCV()  #1.04
    #mdl= Lasso() #1.04
    #mdl=LassoLarsIC() #1.20    
    #mdl=PassiveAggressiveClassifier() #sleh 1.9
    #mdl=LogisticRegression() #ultratraag
    #mdl=ElasticNetCV() #1.13
    #mdl=OrthogonalMatchingPursuit()
    #mdl = ARDRegression()  # ultratraag
    #mdl = IsotonicRegression() 
    #mdl= RidgeCV() #1.04
    mdl.fit(xtr, ytr)
    #print(mdl.coef_,mdl.intercept_)
    px = mdl.predict(meltx.iloc[:,1:])

    
    return ytr,px

#p=trixps(Ttrain.values[:,:unknownmonth],0.5,1,Ai,Bi,Gi,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],12,1)
#array stacken
#TEXPtr=Ttrain.T.append(pd.DataFrame(p).T  )

yts,p=regresseer(TEXPtr.T,unknownmonth)

error = rmse(yts, p)
print('month %d - Error %.5f' % (unknownmonth, error))


subm=pd.DataFrame([])
subm['ID']=test['ID']
subm['item_cnt_month']=np.clip(p,0,20)
subm.to_csv('submission2.csv',index=False)
subm