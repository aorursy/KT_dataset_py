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
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style = "white",color_codes=True)
sns.set(font_scale=1.5)
from IPython.display import display
pd.options.display.max_columns = None

from sklearn.model_selection import GridSearchCV #to fine tune Hyperparamters using Grid search
from sklearn.model_selection import RandomizedSearchCV# to seelect the best combination(advance ver of Grid Search)

# importing some ML Algorithms 
from sklearn.linear_model import LinearRegression # y=mx+c
from sklearn.tree import DecisionTreeRegressor # Entropy(impurities),Gain. 
from sklearn.ensemble import RandomForestRegressor # Average of Many DT's

# Testing Libraries - Scipy Stats Models
from scipy.stats import shapiro # Normality Test 1
from scipy.stats import normaltest # Normality Test 2
from scipy.stats import anderson # Normality Test 3
from statsmodels.graphics.gofplots import qqplot # plotting the Distribution of Y with a Line of dot on a 45 degree Line.

# Model Varification/Validation Libraries
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


# Matrices and Reporting Libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import make_scorer
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import learning_curve

ds= pd.read_csv("../input/mercedesbenz-greener-manufacturing/train.csv")
dt = pd.read_csv("../input/mercedesbenz-greener-manufacturing/test.csv")
ds.head()
ds.columns , dt.columns
ds.shape , dt.shape
ds.dtypes
ds.select_dtypes(include='float').columns
ds.select_dtypes(include='int64').columns
ds.select_dtypes(include='object').columns
# training Data
np.unique(ds[ds.columns[10:]])
# testomg Data
np.unique(dt[dt.columns[10:]])
#training 
ds.isnull().sum().unique()
# testing
dt.isnull().sum().unique()
ds["y"].describe()
ds.shape
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,6))
fig.suptitle("Distribution of Vaehicle Testing Times",fontsize=12)

ax1.scatter(range(ds.shape[0]),np.sort(ds["y"].values)) 
ax2.plot(ds["y"])
sns.distplot(ds["y"],kde=True,ax=ax3,bins=70)
qqplot(ds["y"],line='s',ax=ax4)
# test 1 - Shapiro Test 
stat , p = shapiro(ds["y"])
print('Stat = %.3f, p = %.3f'%(stat,p))

if p>0.05:
    print('Sampe looks like Gausian or Sample is drawn from it.(Failed TO reject the H0)')
else:
    print('Reject the H0. THis is not Noemaly Distributed')
# test 1 - K2 test 
stat , p = normaltest(ds["y"])
print('Stat = %.3f, p = %.3f'%(stat,p))

if p>0.05:
    print('Sampe looks like Gausian or Sample is drawn from it.(Failed TO reject the H0)')
else:
    print('Reject the H0. THis is not Noemaly Distributed')
# Anderson Test
st,cv,sl= anderson(ds["y"])
for a,b in zip(cv,sl):
    if st < a:
        print("{}   {}      {}".format())
        print("{} {} {}(accept H0)".format(a,b,st))
    else:
        print('{} {}% {:.2f}(reject H0)'.format(a,int(b),st))

# Training
print(ds.select_dtypes(include='object').columns)
print(ds.select_dtypes(include='float').columns)
print(ds.select_dtypes(include='int64').columns)
# Testing
print(dt.select_dtypes(include='object').columns)
print(dt.select_dtypes(include='float').columns)
print(dt.select_dtypes(include='int64').columns)
# Lets Tackkle the Object columns to see if the same values are there in Testing data also, or anything new 
t = ds.select_dtypes(include='object').columns
tt = dt.select_dtypes(include='object').columns
for a,b in zip(t,tt):
    print('\nunique values in '+a+' for Training sample are : {}  '.format(ds[a].nunique()))
    print('unique values in '+b+' for Testing samples are : {}  '.format(dt[b].nunique()))
    print('values that are available in '+a+' Training but not in '+b+' Testing  : {}'.format(list(set(ds[a]).difference(dt[b]))))
    print('values that are available in '+b+' Testing but not in '+a+' Training : {}'.format( list(set(dt[b]).difference(ds[a]))))


for col in tt:
    plt.figure(figsize=(16,6))
    sns.boxplot(x=col,y='y',data=ds)
    plt.xlabel(col,fontsize=10)
    plt.ylabel('y',fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=10)
#### ANOVA test - Since input is Categorical and Output is Numerical.
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols ('y ~ C(X4)',data=ds).fit()    
print('F-Statics: ',model.fvalue)
print('p-value: ',model.f_pvalue)
anova_table = sm.stats.anova_lm(model,type=2)
anova_table
from statsmodels.stats.multicomp import pairwise_tukeyhsd
m_comp = pairwise_tukeyhsd(endog=ds['y'],groups=ds['X4'],alpha=0.05)
print(m_comp)
columns = t.values
columns

for col in columns: 
    model = ols('y ~ '+col, data=ds).fit()
    print('Column {}, F-stat: {:.2f} , F-Pvalues {:.2f}'.format(col,model.fvalue,model.f_pvalue))
# calculating the 25% percentile
Q1 = np.percentile(ds.loc[:,'y'],25) # fancy way of slicing only the Y column. You can also use ds["y"]

# calculating the 50% percentile
Q2 = np.percentile(ds.loc[:,'y'],50)

# calculating the 75% percentile
Q3 = np.percentile(ds.loc[:,'y'],75)

# Calculating the outlier using the IQR (Q3-Q1) and 1.5 (left over 1% in normal Distribution curve)
step = (Q3-Q1) *3 

print('Q1 = {}, Q2= {}, Q3={} , Outlier Step = {}'.format(Q1,Q2,Q3,step))
Left_side_outlier = ds[ds['y'] <= (Q1-step)].index  # Int64Index([], dtype='int64') / no outliers on Left side
Right_side_outlier = ds[ds['y'] > (Q3+step)].index # Int64Index([342, 883, 1459, 3133], dtype='int64') 4 indexs of Outliers on Right side
num_left_outlier = len(Left_side_outlier) # 0
num_right_outlier = len(Right_side_outlier) # 4 

print('Number of the Outliers on the lower Side = {}'.format(num_left_outlier))
print('Number of the Outliers on the upper Side = {}'.format(num_right_outlier))

print('lower outliers = {}%'.format((num_left_outlier/ds.shape[0])*100)) #(0/4209) * 100 - this is like accuracy, Number of errors(outliers) / total number of elements(len of Y column). 
print('lower outliers = {}%'.format((num_right_outlier/ds.shape[0])*100)) #( 4/4209) * 100
ds.iloc[Right_side_outlier]['y'].sort_values(ascending= False) # Slicing those outliers Index from the DS to see their Y values.
# Only one Outlier 
ds[ds['y'] >= 170]
combined_data = ds.append(dt,ignore_index=True)
combined_data = pd.get_dummies(combined_data)
ds.shape , dt.shape , combined_data.shape
train,test = combined_data[0:len(ds)],combined_data[len(dt):] # train 0-4209 ,  Test 4209-8418
train.shape,test.shape
x_feature = train.drop(['y','ID'],axis=1)
y_target = train['y']
X_train_features,X_test_features,Y_train_target,Y_test_target = train_test_split(x_feature,y_target,test_size= 0.25 , random_state=4)
X_train_features.shape, Y_train_target.shape , X_test_features.shape,Y_test_target.shape
# Instantiate
lin = LinearRegression()

%%time
lin.fit(X_train_features,Y_train_target) # Fit the Data
# Prediction - Training Samples
y_pred = lin.predict(X_train_features)

print('\nTraining Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_train_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_train_target,y_pred))

# Prediction - Testing Samples
y_pred = lin.predict(X_test_features)

print('\nTesting Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_test_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_test_target,y_pred))
# instantiate
rf = RandomForestRegressor(n_estimators = 20, #Number of decision trees used by the RF
                           criterion='mse',# method to calculate the quality of split
                          max_depth= 4 , # Max depth of Each Decision Tree
                          max_features='auto', 
#say for example We have couple of DT's with 40 columns each, lets take Max feature =8 . 
#Then we calculate the Entropy to all the 40 columns and take the Mean and Max information Gain. This process is common in DT. 
#But here, in Max feature, we will take random heighest features of 8 to decide the Max info Gain. That’s all the difference. Why we are doing this is to choose the averaging. Net effect of this will give you Unbaisad output.

                           min_samples_split = 0.05
                           )
%%time
rf.fit(X_train_features,Y_train_target) # Fit the Data
# Prediction - Training Samples
y_pred = rf.predict(X_train_features)

print('\nTraining Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_train_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_train_target,y_pred))

# Prediction - Testing Samples
y_pred = rf.predict(X_test_features)

print('\nTesting Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_test_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_test_target,y_pred))
from sklearn.neighbors import KNeighborsRegressor
# instantiate
knn = KNeighborsRegressor()
#fit
knn.fit(X_train_features,Y_train_target) # Fit the Data
# Prediction - Training Samples
y_pred = knn.predict(X_train_features)

print('\nTraining Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_train_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_train_target,y_pred))

# Prediction - Testing Samples
y_pred = rf.predict(X_test_features)

print('\nTesting Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_test_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_test_target,y_pred))
from sklearn.ensemble import GradientBoostingRegressor
# instantiate
gbm = GradientBoostingRegressor(n_estimators=600,
                               max_depth=8,
                               min_samples_split=.1,
                               max_features='sqrt',
                               learning_rate=0.01,
                               loss='ls')
#fit
gbm.fit(X_train_features,Y_train_target) # Fit the Data
# Prediction - Training Samples
y_pred = gbm.predict(X_train_features)

print('\nTraining Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_train_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_train_target,y_pred))

# Prediction - Testing Samples
y_pred = rf.predict(X_test_features)

print('\nTesting Score : ')
print('Mean Squared Error = %.2f'% mean_squared_error(Y_test_target,y_pred))
print('R2 score : %.2f'% r2_score(Y_test_target,y_pred))
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
X_train_features,X_test_features,Y_train_target,Y_test_target = train_test_split(x_feature,y_target,test_size= 0.2 , random_state=45)
# creating DMatrices for the XGB training. This is a Binary Representation
dtrain = xgb.DMatrix(X_train_features,label = Y_train_target)
dtest = xgb.DMatrix(X_test_features,label = Y_test_target)
mean_train = Y_train_target.mean()
# Get prediction on the test set
baseline_predictions  = np.ones(Y_test_target.shape)*mean_train
baseline_predictions
mae_baseline = mean_absolute_error(Y_test_target,baseline_predictions)
print("BaseLine Mae is : %.2f"%mae_baseline)
xgb_param = { "max_depth" : 8,
             "min_child_weight":1,
             'eta':.35,
             'subsample':1,
             'colsample_bytree':.9,
             'objective':'reg:squarederror',
             'eval_metric':'mae',
             'validate_parameters':1,
             'verbose_eval':False
            }
%%time
model = xgb.train(xgb_param,
                 dtrain,
                 num_boost_round=999,
                 evals=[(dtest,'Test')],
                 early_stopping_rounds=10
                 )
# predictions
y_pred=model.predict(dtrain) # adding the DMetrics Binary Values of Training Data.

print("Training : Metrics...")
print("Mean ABS Error MAE :     ",metrics.mean_absolute_error(Y_train_target,y_pred))
print("Mean sq Error MSE :      ",metrics.mean_squared_error(Y_train_target,y_pred))

print('Root Mean Sq Error RMSE: ',np.sqrt(metrics.mean_squared_error(Y_train_target,y_pred)))
print('r2value      :           ',metrics.r2_score(Y_train_target,y_pred))

y_pred = model.predict(dtest) # adding the DMetrics Binary Values of Testing Data.

print('\nTesting : Metrics ... ')
print('Mean Abs Error MAE :     ',metrics.mean_absolute_error(Y_test_target,y_pred))
print("Mean sq Error MSE :      ",metrics.mean_squared_error(Y_test_target,y_pred))

print('Root Mean Sq Error RMSE: ',np.sqrt(metrics.mean_squared_error(Y_test_target,y_pred)))
print('r2value      :           ',metrics.r2_score(Y_test_target,y_pred))
