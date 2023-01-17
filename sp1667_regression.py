# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.concat([

        pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv"),

        pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

]).drop(columns = ['SalePrice']).reset_index(drop = True)      
#Using for refrerences

learning = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

q1,q2 = len(pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")),len(pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv"))
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import patsy



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from tensorflow import keras



import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer



import statsmodels.formula.api as smf

import scipy.stats as stats

import patsy
#funtions go here

def scale_df(df,df2,drop,scaler_,order):

    #for train set

    ref = df.select_dtypes(exclude = ['object'])

    ycol = ref[drop]

    ref = ref.drop(columns = drop)

    ref2 = df.select_dtypes(include = ['object'])

    scaler = scaler_.fit(ref)

    r_scaled = pd.DataFrame(scaler.transform(ref),columns = ref.columns)

    out = pd.concat([r_scaled,ref2,ycol], axis = 1)[order]

    #for test set

    ref = df2.select_dtypes(exclude = ['object'])

    ycol = ref[drop]

    ref = ref.drop(columns = drop)

    ref2 = df2.select_dtypes(include = ['object'])

    r_scaled = pd.DataFrame(scaler.transform(ref),columns = ref.columns)

    out2 = pd.concat([r_scaled,ref2,ycol], axis = 1)[order]

    return out,out2



def wacko_impute(df_,K):

    out = pd.DataFrame()

    for _ in tqdm(range(K)):

        df = df_.copy()

        for cols in df.columns:

            if df[cols].isnull().sum() > 0:           

                X,y = df.drop(columns = [cols]).reset_index(drop = True),df[[cols]].reset_index(drop = True)

                X,X2 = pd.get_dummies(X).fillna(-6),pd.get_dummies(X).fillna(-6)

                Q = y[y[cols].notnull()].index.to_numpy()

                Q2 = y[y[cols].isnull()].index.to_numpy()

                X = X.loc[Q]

                y = y.loc[Q]

                SK = MinMaxScaler().fit(X) 

                X = SK.transform(X)

                if df[cols].dtypes == 'object':

                    model = KNeighborsClassifier(np.random.choice(range(1,8))).fit(X,np.ravel(y))

                else: 

                    model = KNeighborsRegressor(np.random.choice(range(1,8))).fit(X,np.ravel(y))

                df.loc[df[df[cols].isnull()].index.to_numpy(),cols] = model.predict(SK.transform(X2.loc[Q2]))

        out = pd.concat([out,df])

    return out



def diagnostics(model, qq = "norm", resid = False, res_ = None, pred_ = None):

    f,ax = plt.subplots(1,2,figsize = (20,6))

    if resid == False:

        fit,res = model.fittedvalues, model.resid

    else:

        fit = pred_

        res = res_

    ax[0].scatter(fit , res)

    ax[1] = stats.probplot(res, dist=qq, plot=plt)

    plt.show()

    

def createDsets(Train_,Test_,formula):

    v1,v2 = len(Train_),len(Test_)

    M = pd.concat([Train_, Test_])

    y_,X_ = patsy.dmatrices(formula,M)

    X_train,Y_train = X_[0:v1],y_[0:v1]

    X_test,Y_test = X_[v1:],y_[v1:]

    return X_train,Y_train,X_test,Y_test



def evalTest(model,Test,X_test,log = True):

    Eval = Test[['C_0','Y']].reset_index(drop = True)

    if log == True:

        Eval['Yhat'] = np.exp(model.predict(X_test))

    else:

        Eval['Yhat'] = model.predict(X_test)

    Eval = Eval.groupby('C_0').median().reset_index()

    f,ax = plt.subplots(1,2,figsize = (20,6))

    r2Sc = r2_score(Eval['Y'],Eval['Yhat'])

    mSc = mean_absolute_error(Eval['Y'],Eval['Yhat'])

    plt.title("Plot Predicted vs Fitted correlation(left)  Residual Histogram (Right)   Rsquared => " + str(round(r2Sc,2)) + "     MASE => "+ str(round(mSc,2)))

    ax[0].scatter(Eval['Yhat'], Eval['Y'])

    ax[1].hist(Eval['Yhat'] - Eval['Y'], bins = 50)

    plt.show()
summary_data = []

for i in df.columns:

    summary_data.append ([i, len(df[i].unique()),df[i].isnull().sum(),int(df[i].isnull().sum()/len(df)*100) ,len(df) - df[i].isnull().sum(),df[i].dtypes])

summary = pd.DataFrame(summary_data,columns = ['colname','unique','missing','misspct','nfq','dt']).sort_values(by = ['misspct'], ascending = False)

df = df.drop(columns = ['PoolQC','MiscFeature','Alley','Fence'])

obclass = df.select_dtypes(include=['object'])

numeric = df.select_dtypes(exclude=['object'])

xdf = wacko_impute(df,20)
xdf2 = xdf.copy()

trainpoints = df['Id'].to_numpy()[0:q1]

testpoints = df['Id'].to_numpy()[q1:]

train = xdf2[xdf2['Id'].isin(trainpoints)]

test = xdf2[~xdf2['Id'].isin(trainpoints)]

M = train.merge(learning[['Id','SalePrice']], on = 'Id', how = 'left')

M.columns = ["C_" + str(i) for i in range(len(M.columns))]

mlpoints = trainpoints.copy()

np.random.shuffle(mlpoints)

tpoints = mlpoints[0:1200]

vpoints = mlpoints[1200:]

train_ = M[M['C_0'].isin(tpoints)]

test_ = M[M['C_0'].isin(vpoints)]
Train_0 = train_.copy().reset_index(drop =True)

Test_0 = test_.copy().reset_index(drop =True)

Train_0 = Train_0.rename(columns = {'C_76':'Y'})

Test_0 = Test_0.rename(columns = {'C_76':'Y'})

Train_1,Test_1 =scale_df(Train_0,Test_0,['C_0','Y'],MinMaxScaler(),Train_0.columns)



formula = """

Y ~ C_1 + C_2 + C_3 + C_4 + C_5 + C_6 + C_7 + C_8 + C_9 + C_10 + C_11 + C_12 + C_13 + 

C_14 + C_15 + C_16 + C_17 + C_18 + C_19 + C_20 + C_21 + C_22 + C_23 + C_24 + C_25 + C_26 + 

C_27 + C_28 + C_29 + C_30 + C_31 + C_32 + C_33 + C_34 + C_35 + C_36 + C_37 + C_38 + C_39 + 

C_40 + C_41 + C_42 + C_43 + C_44 + C_45 + C_46 + C_47 + C_48 + C_49 + C_50 + C_51 + C_52 + 

C_53 + C_54 + C_55 + C_56 + C_57 + C_58 + C_59 + C_60 + C_61 + C_62 + C_63 + C_64 + C_65 + C_66 + 

C_67 + C_68 + C_69 + C_70 + C_71 + C_72 + C_73 + C_74 + C_75

"""
Figure = plt.figure(figsize = (8,8))

plt.imshow(train_.corr(), cmap ='seismic')

plt.colorbar()

plt.show()
X_train,Y_train,X_test,Y_test = createDsets(Train_1,Test_1,formula)

model = sm.OLS(Y_train,X_train).fit()

diagnostics(model)

evalTest(model,Test_1,X_test, log = False)
X_train,Y_train,X_test,Y_test = createDsets(Train_1,Test_1,formula)

model1 = sm.OLS(np.log(Y_train),X_train).fit()

diagnostics(model1)

evalTest(model1,Test_1,X_test, log = False)
Train_2,Test_2 =scale_df(Train_0,Test_0,['C_0','Y'],QuantileTransformer(

    n_quantiles = 100,

    output_distribution = 'uniform'

    ),Train_0.columns)



X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)



model2 = sm.OLS(np.log(Y_train),X_train).fit()

diagnostics(model2)

evalTest(model2,Test_1,X_test)
Train_3,Test_3 =scale_df(Train_0,Test_0,['C_0','Y'],Normalizer(

    norm = 'l2'

    ),Train_0.columns)



X_train,Y_train,X_test,Y_test = createDsets(Train_3,Test_3,formula)



model3 = sm.OLS(np.log(Y_train),X_train).fit()

diagnostics(model3)

evalTest(model3,Test_1,X_test)
Train_2,Test_2 =scale_df(Train_0,Test_0,['C_0','Y'],QuantileTransformer(

    n_quantiles = 100,

    output_distribution = 'uniform'

    ),Train_0.columns)



X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)



model4 = sm.OLS(np.log(Y_train),X_train).fit_regularized(alpha = 0.0001,L1_wt = 0.003)

fit = np.exp(model4.predict(X_train))

res = Y_train.reshape(-1) - fit

diagnostics(model4, resid = True, res_ = res, pred_ = fit)

evalTest(model4,Test_1,X_test)
X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)



model5 = RandomForestRegressor(n_estimators = 10).fit(X_train,np.ravel(np.log(Y_train)))

fit = np.exp(model5.predict(X_train))

res = Y_train.reshape(-1) - fit

evalTest(model5,Test_1,X_test)
f = plt.figure(figsize = (10,5))

imp  = model5.feature_importances_

feat_imp = pd.DataFrame({'Imp' : imp, 'lab' : [str(i) for i in range(len(imp))]}).sort_values(by = 'Imp',ascending = False)

feat_imp_10 = feat_imp.copy().head(10)

plt.barh(feat_imp_10['lab'],feat_imp_10['Imp'])

plt.title("RF analysis top 10 most important features #Num = Column in X-Train Matrix")

plt.show()
t10f = feat_imp_10['lab'].astype('int').to_numpy()

X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)

X_train = X_train[:,t10f]

X_test = X_test[:,t10f]



model5 = RandomForestRegressor(n_estimators = 10).fit(X_train,np.ravel(np.log(Y_train)))

fit = np.exp(model5.predict(X_train))

res = Y_train.reshape(-1) - fit

evalTest(model5,Test_1,X_test)
# The same method as before fit using a GBM

t10f = feat_imp_10['lab'].astype('int').to_numpy()

X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)

X_train = X_train[:,t10f]

X_test = X_test[:,t10f]



model6 = GradientBoostingRegressor(n_estimators = 100).fit(X_train,np.ravel(np.log(Y_train)))

fit = np.exp(model6.predict(X_train))

res = Y_train.reshape(-1) - fit

evalTest(model6,Test_1,X_test)
t10f = feat_imp['lab'].astype('int').to_numpy()[0:10]

X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)

X_train = X_train[:,t10f]

X_test = X_test[:,t10f]



model7 = SVR().fit(X_train,np.ravel(np.log(Y_train)))

fit = np.exp(model7.predict(X_train))

res = Y_train.reshape(-1) - fit

evalTest(model7,Test_1,X_test)
def mlp(dim):

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(100, input_dim = dim, activation = 'sigmoid'))

    model.add(keras.layers.Dense(100, activation = 'sigmoid',kernel_regularizer=keras.regularizers.l1(0.0001)))

    model.add(keras.layers.Dense(100, activation = 'sigmoid',kernel_regularizer=keras.regularizers.l1(0.0001)))

    model.add(keras.layers.Dense(100, activation = 'sigmoid',kernel_regularizer=keras.regularizers.l1(0.0001)))

    model.add(keras.layers.Dense(1, input_dim = dim, activation = 'linear',kernel_regularizer=keras.regularizers.l1(0.0001)))

    model.compile(optimizer = keras.optimizers.Adam(0.00005), loss = 'MAPE')

    return model
X_train,Y_train,X_test,Y_test = createDsets(Train_2,Test_2,formula)



model8 = mlp(X_train.shape[1])

model8.fit(X_train,np.ravel(np.log(Y_train)),epochs = 80, validation_split = 0.15, shuffle = True)

fit = np.exp(model8.predict(X_train))

res = Y_train.reshape(-1) - fit

evalTest(model8,Test_1,X_test)
C = test .copy().reset_index(drop = True)

C.columns = ["C_" + str(i) for i in range(len(C.columns))]

C['Y'] = 0



_t_ = pd.concat([Train_0,Test_0]).reset_index(drop = True)

_i_,Test_n =scale_df(Train_0,C,['C_0','Y'],QuantileTransformer(

    n_quantiles = 100,

    output_distribution = 'uniform'

    ),Train_0.columns)



X_train,Y_train,X_test,Y_test = createDsets(_i_,Test_n,formula)

X_train = X_train[:,t10f]

X_test = X_test[:,t10f]



model8 = GradientBoostingRegressor(n_estimators = 100).fit(X_train,np.ravel(np.log(Y_train)))



sub1 = test[['Id']].reset_index(drop = True)

sub1['y'] = np.exp(model8.predict(X_test))

sub1 = sub1.groupby('Id').mean().reset_index(drop = True)



sub2 = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

sub2['SalePrice'] = sub1['y']



sub2.to_csv("submission.csv",index = False)