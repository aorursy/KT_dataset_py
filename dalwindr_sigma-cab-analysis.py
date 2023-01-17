# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,roc_auc_score,roc_curve, precision_score,f1_score,auc,precision_recall_curve,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
le=LabelEncoder()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
train=pd.read_csv('../input/train-jantahack-cabmobilitycsv/train_JantaHack_Cabmobility.csv')
# train.dropna()[label_col].value_counts()/train.dropna()[label_col].shape[0]
test=pd.read_csv("../input/test-jantahack-cabmobilitycsv/test_JantaHack_Cabmobility.csv")
# append two training and test dataset
df=train.append(test)

df.shape
#Understanding Proportion of Null Values
df.isnull().mean()*100
label_col='Surge_Pricing_Type'
df.shape
def desc1(_DF,corr=0,label_col='pass_label_col',orderbyColumn='default',__order=True):
    from sklearn.preprocessing import normalize
    import math
    """
       1. Pearson linear correlation (value lies between -1 to +1.)
       Perfect value:
        ρ = +1 means perfect positive relationship (x increases, y also increases).
        ρ = -1 means perfect negative relationship (x decreases, y also decreases).
        ρ = 0 means no relation between variables.
        High degree: If ρ value lies between ± 0.50 and ± 1, then it is said to be a strong correlation.
        Moderate degree: If ρ value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation.
        Low degree: If ρ value lies below ± 0.29, then it is said to be a small correlation.
        Drawback
            Very good when we have linear relationship. Which may not be true always in real world. So,
             it does not work well when you do not have linear relationship between variables.
    """

    unique_val=pd.DataFrame(index=_DF.columns)
    def obtain_variance(_DF):
        
        if _DF.dtypes in ('float64','int64'):
             xy1= pd.DataFrame(np.array(_DF*1.0))
             # variance values less than 0.006 ( threash hold), drop the column
            # If it is categorical binary column and if values 95:5 ratio, you drop the variable.
            #If it is continuous column and variance is less than 0.0066, you drop the variable (remember 0.0066 is value obtained after normalizing the variable).

             return "%3g"%xy1.var()[0]
        elif _DF.dtypes =='object':
            xy1= pd.DataFrame(_DF)
            xy1.reset_index()        
            return 0.0
        else:
            return 0.0     
    def obtain_std(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%3g"%pd.DataFrame(_DF*1.0).std()[0]
        else:
            return 0.0
    def obtain_mean(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%3g"%pd.DataFrame(_DF*1.0).mean()[0]
        else:
            return 0.0
    def obtain_min(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%3g"%pd.DataFrame(_DF*1.0).min()[0]
        else:
            return 0.0
    def obtain_max(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%3g"%pd.DataFrame(_DF*1.0).max()[0]
        else:
            return 0.0
    def obtain_skew(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%.3f"%pd.DataFrame(_DF).skew()
        else:
            return 0.0
    def obtain_kurtosis(_DF):
        if _DF.dtypes in ('float64','int64'):
            return "%.3f"%pd.DataFrame(_DF).kurt()
        else:
            return 0.0
    def obtain_Numeric_pearSonCorr(_DF):
        if _DF.dtypes in ('float64','int64'):
            return np.abs(_DF.corr(_ytrain))
        else:
            return 0.0
        
    for i in _DF.columns:
        unique_val.loc[i,'dtypes']=_DF[i].dtypes
        unique_val.loc[i,'null_count']=_DF[i].isnull().sum() 
        unique_val.loc[i,'total count']=_DF[i].notnull().sum()
        unique_val.loc[i,'unique_count']=_DF[i].nunique()
        unique_val.loc[i,'missing value ratio']= round((_DF[i].isnull().sum()/len(_DF))*100,2)
        unique_val.loc[i,'variance of numerics']= round(float(obtain_variance(_DF[i])),2)
        unique_val.loc[i,'std']= round(float(obtain_std(_DF[i])),2)
        unique_val.loc[i,'mean']= round(float(obtain_mean(_DF[i])),2)
        unique_val.loc[i,'min']= float(obtain_min(_DF[i]))
        unique_val.loc[i,'max']= round(float(obtain_max(_DF[i])),2)
        unique_val.loc[i,'skew']= float(obtain_skew(_DF[i]))
        unique_val.loc[i,'kurt']= float(obtain_kurtosis(_DF[i]))
        if corr==2: 
            _ytrain=_DF[label_col]
            unique_val.loc[i,'Oridinals SpearmanCorr withTarget']=np.abs(_DF[i].corr(_ytrain,'spearman').round(5))
        if corr==1: 
            _ytrain=_DF[label_col]
            unique_val.loc[i,'Numeric pearSonCorr withTarget']=obtain_Numeric_pearSonCorr(_DF[i]) # _DF[i].corr(ytrain,'pearson').round(5)

    if (orderbyColumn=='default') and (corr==0):
        return unique_val.sort_values(by=['unique_count','missing value ratio'])
    
    elif (corr==0) and (orderbyColumn== 'Corr withTarget'):
        return unique_val.sort_values(by=['unique_count','missing value ratio'])
    elif corr==1:
        return unique_val.sort_values(by=['Numeric pearSonCorr withTarget','unique_count'],ascending=[False,True])
    elif corr==2:
        return unique_val.sort_values(by=['Oridinals SpearmanCorr withTarget','unique_count'],ascending=[False,True])
    
    else: 
        return unique_val.sort_values(by=[orderbyColumn],ascending=__order)
sammy=desc1(df,2,label_col)
sammy
# cat types cols
cat_colsdf=sammy[sammy.unique_count<15]
cat_colsdf
#  1. Half rounding (auto ceiling and flooring) of float
from decimal import localcontext, Decimal, ROUND_HALF_UP
df['CR']=df['Customer_Rating'].apply(lambda x: Decimal(x).to_integral_exact(rounding=ROUND_HALF_UP))
df['CR']=df['CR'].astype('int32')


# 2. Imputing 

df['Customer_Since_Months'].describe()
df['Customer_Since_Months'].fillna(6.006048,inplace=True)

df['Life_Style_Index'].describe()
df['Life_Style_Index'].fillna(2.79,inplace=True)

CustSinceMonth_2CabType_MAP=df.groupby(['Customer_Since_Months'])['Type_of_Cab'].agg(pd.Series.mode)
df.loc[df['Type_of_Cab'].isnull(),'Type_of_Cab']=df.loc[df['Type_of_Cab'].isnull(),'Customer_Since_Months'].map(CustSinceMonth_2CabType_MAP)

CustSinceMonth_2Confidence_Life_Style_Index_MAP=df.groupby(['Customer_Since_Months'])['Confidence_Life_Style_Index'].agg(pd.Series.mode)
df.loc[df['Confidence_Life_Style_Index'].isnull(),'Confidence_Life_Style_Index']=df.loc[df['Confidence_Life_Style_Index'].isnull(),'Customer_Since_Months'].map(CustSinceMonth_2Confidence_Life_Style_Index_MAP)

df['Var1'].describe()
df['Var1'].fillna(61.0,inplace=True)
# 3. Billing of Continous Variable
def BiningContiCols_over_yBinary(contiColList,dataframe,bins=4):
    """
        Pass a list of continues columns , It will convert it into four bins and 

    """

    group=[i for i in range(1,bins+1)]
    dict1={'Low':1,'Average':2,'High':3, 'Very high':4}
    binsdf= pd.DataFrame()
    for conti_colname in contiColList:
        _colData=pd.DataFrame({conti_colname: dataframe[conti_colname]
                              },index=dataframe.index)
        _n_cols=pd.qcut(dataframe[conti_colname],q=bins,duplicates='drop').nunique()
        _colData['bins']=pd.qcut(dataframe[conti_colname], q=bins,duplicates='drop',labels=group[:_n_cols])
        
        binsdf[conti_colname+'_bins'] = _colData['bins']#.map(dict1)
        binsdf=binsdf.astype('int')
    return binsdf

ContiBins=BiningContiCols_over_yBinary(['Var1','Var2','Var3','Customer_Since_Months','Customer_Rating','Trip_Distance','Life_Style_Index'],df,5)
ContiBins=ContiBins.astype('object')
df[ContiBins.columns]=ContiBins
ntrain=train.shape[0]
desc1(df,2,label_col)
# 4. Describe method
# focus on columns
# [dtypes ,null_count,total count,unique_count,missing value ratio,variance of numerics,std,mean,min,max,skew,kurt,Oridinals SpearmanCorr withTarget]
# If you have continous valibale Use 1 in the 2nd parameter of the desc1 method , it will return pearson correlatinn with target label column
#['Var1','Var2','Var3','Customer_Rating','Trip_Distance','Life_Style_Index']
#def FE_2Add_CatFrequencyCount3(_ntrain,_df,apply=1):
#    """ Its a percentage encoding of each category"""
    #import category_encoders as ce
    #from imblearn import categorical_encoders as ce
#     if apply==1:
#         cat_temp=['Type_of_Cab','Cancellation_Last_1Month','CR','Life_Style_Index_bins','Destination_Type','Customer_Since_Months','Gender']
#         _cols=[i+"_freq_encode" for i in cat_temp]
#         freq_encoder=ce.CountFrequencyCategoricalEncoder(encoding_method="frequency")
#         _fe_train=imp.fit_transform(_df[:_ntrain][["Type_of_Cab","Confidence_Life_Style_Index","Destination_Type","Gender"]])
#         _fe_train.columns=_cols
#         _fe_test=imp.fit_transform(_df[_ntrain:][["Type_of_Cab","Confidence_Life_Style_Index","Destination_Type","Gender"]])
#         _fe_test.columns=_cols
#         _fe_all=_fe_train.append(_fe_test)
#         _df=pd.concat([_df,_fe_all],1)
#         return _df
# 5. Freqeuency Encoding of Cat columns
def FE_2Add_CatFrequencyCount3(_ntrain,_df,apply_or_test=0):
    print(""" Note: Its a count frequence encoding of each category
                # if categories have same frequency it can be an issue
                # will need to change it to ranked frequency encoding
                from scipy.stats import rankdata
          """)
    
    cat_temp=['Type_of_Cab','Cancellation_Last_1Month','CR','Life_Style_Index','Destination_Type','Customer_Since_Months','Gender']   
    
    _fe_train=pd.DataFrame()
    _fe_test=pd.DataFrame()
    for c in cat_temp:
        dic_train=dict(_df[:ntrain][c].value_counts()/_ntrain)
        _fe_train['FreqEncode_'+c]=_df[:ntrain][c].map(dic_train).astype('float32')
        dic_test=dict(_df[ntrain:][c].value_counts()/_ntrain)
        _fe_test['FreqEncode_'+c]=_df[ntrain:][c].map(dic_test).astype('float32')

    if apply_or_test==1:   
        _fe=_fe_train.append(_fe_test)
        _df=pd.concat([_df,_fe],1)
        return _df
    else:
        print(_fe_train.var().sort_values(ascending=False))
FE_2Add_CatFrequencyCount3(ntrain,df,apply_or_test=0)
# 6. Mean Encoding of Conti columns
def FE_2Add_CatMeanEncoding2(_label_Col,_ntrain,_df,apply_or_test=0):
    print(""" Note: Its a mean/target encoding of each category
               
          """)
    
    cat_temp=['Type_of_Cab','Cancellation_Last_1Month','CR','Life_Style_Index','Destination_Type','Customer_Since_Months','Gender']   
    
    _fe=pd.DataFrame()
    for c in cat_temp:
        dic_train1=_df[:ntrain].groupby([c]).agg({label_col:['mean']})
        dic_train1.columns=['target']
        dic_train=dict(dic_train1.target.round(3))
        #data_df = data_df.merge(mean_encoding,on=c,how='left')
        _fe['MeanEncode_'+c]=_df[c].map(dic_train)#.astype('float32')

    if apply_or_test==1:   
        _df=pd.concat([_df,_fe],1)
        return _df
    else:
        print(_fe.var().sort_values(ascending=False))

# For now below method is use less .. need to investiage on this concept
def FE_2Add_CatMeanOverContiAggregation2(_label_col,_df,apply=1):   
    if apply==1:
        cat_temp=['Type_of_Cab','CR','Destination_Type','Customer_Since_Months','Gender']
        conti_temp=['Var1','Var2','Var3','Customer_Rating','Trip_Distance','Life_Style_Index']
        
        tempdata=_df[[_label_col]+cat_temp+conti_temp].copy()
        _returnDF= pd.DataFrame()
        for conti_colname in conti_temp:
            print(conti_colname)
            for cat_colname in cat_temp :
                for aggr in ['mean']:
                    #print(conti_colname,cat_colname,aggr)
                    tempAgg1=tempdata[tempdata[_label_col]==1].groupby(cat_colname)[conti_colname].agg(aggr)
                    tempAgg2=tempdata[tempdata[_label_col]==2].groupby(cat_colname)[conti_colname].agg(aggr)
                    tempAgg3=tempdata[tempdata[_label_col]==3].groupby(cat_colname)[conti_colname].agg(aggr)
                    tempAgg12=tempAgg1/tempAgg2
                    tempAgg23=tempAgg2/tempAgg3
                    tempAgg123=1/tempAgg12* (1/tempAgg23)
                    #tempdata[cat_colname+'0-'+aggr+'-over-'+conti_colname]=df_all[cat_colname].map(dict(tempAgg1))
                    #tempdata[cat_colname+'1-'+aggr+'-over-'+conti_colname]=df_all[cat_colname].map(dict(tempAgg0))
                    _returnDF[cat_colname+'1/0-'+aggr+'-over-'+conti_colname]=tempdata[cat_colname].map(dict(tempAgg123))
    return _returnDF
    #df_all=tempdata.copy()
    #del tempdata
#resultDF=FE_2Add_CatMeanOverContiAggregation2(label_col,df)
#df[resultDF.columns]=resultDF
#resultDF.isna().sum()
#'Customer_Since_Months'

# 7.  dummyfying 
def dummify(_df):
    cat_cols=list(_df.columns[_df.dtypes=='object'])
    print(cat_cols)
    if label_col in cat_cols:
        cat_cols.pop(label_col)
    
    print(_df.shape)
    _df[cat_cols]
    dummiedDF=pd.get_dummies(_df[cat_cols],cat_cols,drop_first=True)
    _df[dummiedDF.columns]=dummiedDF
    _df.drop(cat_cols,1,inplace=True)


    # Lets drop all the columns in entire data where testing data columns holding single value
    #### 4.1 Zero test for train and test data drop lowest variablity data
    zerotest_testing_col=[]
    for i in dummiedDF.columns:
        if i in _df.columns:
            zerotest_cond=_df[i][ntrain:].sum()==0
            if zerotest_cond:
                zerotest_testing_col.append(i)
    print(len(zerotest_testing_col),"-",zerotest_testing_col)
    if len(zerotest_testing_col) > 0:
       _df.drop(zerotest_testing_col,1,inplace=True)
    _df.shape

    # # Lets drop all the columns in entire data where training data columns holding single value
    zerotest_training_col=[]
    for i in dummiedDF.columns:
        if i in _df.columns:
            zerotest_cond=_df[i][:ntrain].sum()==0
            if zerotest_cond:
                zerotest_training_col.append(i)
    print(len(zerotest_training_col),"-",zerotest_training_col)
    if len(zerotest_training_col) > 0:
       _df.drop(zerotest_training_col,1,inplace=True)
    return _df

def statandardize(_label_col,_ntrain, _full_data,applyScale=''):
    if _label_col in _full_data.columns:
        _ytrain=_full_data[_label_col][:_ntrain]
        _full_data=_full_data.drop([_label_col],1)
    # Feature Scaling
    _ytrain = pd.Series([int(i) for i in _ytrain])
    _xtrain=_full_data[:ntrain]
    _xtest=_full_data[ntrain:]
    
    if applyScale =='std':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        _xtrain = pd.DataFrame(sc.fit_transform(_xtrain),columns=_xtrain.columns)
        _xtest =  pd.DataFrame(sc.fit_transform(_xtest),columns=_xtest.columns)
    if applyScale =='minmax':
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        _xtrain = pd.DataFrame(sc.fit_transform(_xtrain),columns=_xtrain.columns)
        _xtest =  pd.DataFrame(sc.fit_transform(_xtest),columns=_xtest.columns)

    return _xtrain,_ytrain,_xtest
#xtrain,ytrain,xtest=statandardize(label_col,ntrain,final_data,'') 

def bestRFE(ybinary,_xdata,k_value):
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    #estimator = SVR(kernel="linear")
    model_RFE = LogisticRegression()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model_lda1 = LinearDiscriminantAnalysis()
    rfe = RFE(estimator=model_lda1, n_features_to_select=k_value, step=2)
    fit= rfe.fit(_xdata, ybinary)
    ranking_df = pd.DataFrame()
    ranking_df['Feature_name'] = _xdata.columns[fit.n_features_]
    return _xdata.columns[fit.support_]
# 8. Cross Validation method
def classificationModelfit_CV2(_algo,_xdata,_ydata,_cv):
    from sklearn.model_selection import cross_validate
    _algo.fit(_xdata, _ydata)
    #Perform cross-validation:
    from sklearn.metrics import confusion_matrix
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
#     scoring1 = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
#                 'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    cv_matrix_score= cross_validate(_algo, _xdata, _ydata, cv=_cv,
                                    #scoring=('accuracy', 'precision','f1', 'recall','roc_auc','neg_log_loss',),
                         return_train_score=True)
    avg_model_Scores=pd.DataFrame(cv_matrix_score).mean()
    for i in avg_model_Scores.index:
        if 'neg' in i :
            score="%s =%0.3f" %(i, avg_model_Scores[i]*-1) 
            #_df.loc[i]=np.sqrt(avg_model_Scores[i]*-1)
            print(score)
        else:
            score="%s =%0.3f" %(i, avg_model_Scores[i] )
            #_df.loc[i]=
            #avg_model_Scores[i]
            print(score)
    return avg_model_Scores

def generate_submission_file(submission_csv_name,model,org_test,cleaned_test):
    y_pred = model.predict(cleaned_test)

    #Export submission file:
    org_test[label_col]=[int(i) for i in y_pred]
    submission=org_test[['Trip_ID',label_col]]
    submission[label_col]= submission[label_col]#.map({1:"Y",0:"N"})
#     negRec=len(submission[submission.Item_Outlet_Sales<0])
#     if negRec >0:
#         print("Output contain Negative records:", negRec)
#         submission.to_csv(submission_csv_name, index=False)
#         return submission[submission.Item_Outlet_Sales<0]
#     else:
    print("saved File :",submission_csv_name)
    submission.to_csv(submission_csv_name, index=False)
        
    return submission[label_col]


# 9. Normal Quitile Tranformation methid
def get_conti_cols_transformed(_conti_cols,_df,e=0,type='NQ'):
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import QuantileTransformer
    boxCox = PowerTransformer(method='box-cox')
    YeoJohnson = PowerTransformer(method='yeo-johnson')
    # n_quantiles is set to the training set size rather than the default value
    # to avoid a warning being raised by this example
    QuantileTransF = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                             random_state=1123)

    def getTF(columns,_df,e):
        _df11=pd.DataFrame()
        _df11[[columns]]=_df[[columns]]
        org_skew=round(_df[[columns]].skew()[0],3)
        _df11['box-cox']=boxCox.fit_transform(_df[[columns]]+e)
        _df11['Yeo-Johnson']=YeoJohnson.fit_transform(_df[[columns]]+e)
        _df11['QuantileTrans']=QuantileTransF.fit_transform(_df[[columns]]+e)
        #_df11['exponential']=np.power(_df[[columns]],e)
        #_df11['Nroot']=np.power(_df[[columns]],1/e)
        #_df11['nplog']=np.log(_df[[columns]]+e )
        
        # Create dataFrame for all the skew values of all the transformation
        _df12=pd.DataFrame(_df11.skew(),columns=[columns])
        _df12_min=_df12[_df12[columns]>0 ].min()[0]
        _df12_max=_df12[_df12[columns]<0 ].max()[0]
        import math
      
        if math.isnan(_df12_max) == False:
             best_transformation=list(_df12[_df12[columns]==_df12_max ].index)[0]
             ret1= (columns,best_transformation,round(_df12_max,3),org_skew)
        elif math.isnan(_df12_min) == False:
            best_transformation=list(_df12[_df12[columns]==_df12_min ].index)[0]
            ret1=  (columns,best_transformation,round(_df12_min,3),org_skew)
        return _df11[best_transformation]
    def getBC(columns,_df,e):
        return boxCox.fit_transform(_df[[columns]]+e)
    def getYJ(columns,_df,e):
        return YeoJohnson.fit_transform(_df[[columns]]+e)
    def getNorm_Quantile(columns,_df,e):
        return QuantileTransformer(output_distribution='normal').fit_transform(_df[[columns]]+e)
                                           

    _df13=pd.DataFrame()
    for i  in _conti_cols[:20]:
        print(i)
        if type=='NQ':
            best_transformation=getNorm_Quantile(i,_df,e)
        elif type=='BC':
            best_transformation=getBC(i,_df,e)
        elif type=='YJ':
            best_transformation=getYJ(i,_df,e)
        elif type=='CUST':
            best_transformation=getTF(i,_df,e)   
        else:
            best_transformation=getNorm_Quantile(i,_df,e)
        best_transformation=pd.DataFrame(best_transformation,columns=[i])
        #print(best_transformation)
        _df13[i]=best_transformation[i]
    return _df13
# 9. Normal Quitile Tranformation
def ContiTransformation(_df):
    conti_cols_FE=['Var2','Var1', 'Var3','Life_Style_Index','Trip_Distance','Customer_Rating','Customer_Since_Months_X_Customer_Rating']
    NORMQUINT_TRANS=get_conti_cols_transformed(conti_cols_FE[1:],_df,e=1,type='NQ')
    _df[NORMQUINT_TRANS.columns]=np.array(NORMQUINT_TRANS) 
    return _df

# 10. Feature Engineering
def customFE(_df):
    cond=[(_df['Gender']=='Male')& (_df['Cancellation_Last_1Month']==0), # 4
          (_df['Gender']=='Female')& (_df['Cancellation_Last_1Month']==0), # 3
          (_df['Gender']=='Male')& (_df['Cancellation_Last_1Month'].isin([1])), # 3
          (_df['Gender']=='Female')& (_df['Cancellation_Last_1Month'].isin([1,2])), # 2
          (_df['Gender']=='Male')& (_df['Cancellation_Last_1Month'].isin([2,3])), # 2
         ]
    choices=[4,3,3,2,2]
    np.select(cond,choices,default=1)
    _df['GenderWise_Cancellation_Last_1Month']=np.select(cond,choices,default=1)
    _df['GenderWise_Cancellation_Last_1Month_OT']=_df['GenderWise_Cancellation_Last_1Month'].astype('object')
    print(_df['GenderWise_Cancellation_Last_1Month'].isna().sum())
    
    
    cond=[ (_df['Confidence_Life_Style_Index']=='A')&(_df['Gender'] =='Male'),
       (_df['Confidence_Life_Style_Index']=='B')&(_df['Gender'] =='Male'),
      (_df['Confidence_Life_Style_Index']=='C')&(_df['Gender'] =='Male'),
      (_df['Confidence_Life_Style_Index']=='C')&(_df['Gender'] =='Female'),
     ]

    _df['GenderWise_Confidence_Life_Style_Index']=np.select(cond,[1,2,1,4],default=3)
    _df['GenderWise_Confidence_Life_Style_Index_OT']=_df['GenderWise_Confidence_Life_Style_Index'].astype('object')
    _df['GenderWise_Confidence_Life_Style_Index_OT']

    return _df
    
################### Main Execution #########################

final_data=df.copy()
# Feature Engeering 1
final_data=customFE(final_data)
# Feature Engeering 2
final_data['Customer_Since_Months_X_Customer_Rating']=final_data['Customer_Since_Months']*final_data['Customer_Rating']
final_data['CRWiseMeanTripDistance']=final_data['CR'].map(df.groupby('CR')['Trip_Distance'].mean().astype('int32'))

# Continous variable Transformation 2
final_data=ContiTransformation(final_data)
if 'Trip_ID' in final_data.columns:
    final_data.drop('Trip_ID',1,inplace=True)

# apply frequency Endcoding    
final_data=FE_2Add_CatFrequencyCount3(ntrain,final_data,apply_or_test=1)
print(final_data.isna().sum())
# apply mean Endcoding    

final_data=FE_2Add_CatMeanEncoding2(label_col,ntrain,final_data,apply_or_test=1)
fdata=dummify(final_data)

final_data.head()
summy=desc1(final_data,2,label_col)
c=summy[summy['dtypes']=='object'].index
# Feature Scaling
ytrain = fdata[:ntrain][label_col]
_xtrain=fdata[:ntrain].drop(label_col,1)
_xtest=fdata[ntrain:].drop(label_col,1)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
xtrain = pd.DataFrame(sc.fit_transform(_xtrain),columns=_xtrain.columns)
xtest =  pd.DataFrame(sc.fit_transform(_xtest),columns=_xtest.columns)

# 11. Lets me reduce the dataset
#(pd.Series(ytrain).value_counts()* .35)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.datasets import make_imbalance
X, y = make_imbalance(xtrain, ytrain,
                      sampling_strategy={1: 9524, 2: 19854, 3: 16702},
                      random_state=45)
xtrain.shape,xtest.shape,X.shape,y.shape
# Xgboost estimator
# [.1,.08,0.05,0.03,.01]
from xgboost import XGBClassifier
from xgboost import XGBClassifier
xgb1 = XGBClassifier(learning_rate =0.05, n_estimators=10, max_depth=8, min_child_weight=4, subsample=0.8,
                     colsample_bytree=0.8, objective= 'multi:softmax', 
                                 num_class = 3, nthread=4, seed=27)#, gamma=2,scale_pos_weight=.90,reg_alpha=2,reg_lambda=2)
#modelfit(xgb1, train, predictors)
d=classificationModelfit_CV2(xgb1,X,y,5)
#xGBC_pred=generate_submission_file("n1.csv",xgb1,test,xtest)
from catboost import CatBoostClassifier
cb=CatBoostClassifier()
d=classificationModelfit_CV2(cb,X,y,5)
from catboost import Pool, cv
# from catboost import pool
cv_dataset = Pool(data=X,
                  label=y)

params = {"iterations": 100,
          "depth": 6,
          "loss_function": "MultiClass",
          "verbose": False}

scores = cv(cv_dataset,
            params,
            fold_count=2)

eval_dataset = Pool(data=X)
model = CatBoostClassifier(**params)
model.fit(cv_dataset)
pred=model.predict(eval_dataset)
accuracy_score(pred,y)



