import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
%pylab inline
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()
from scipy import stats
from scipy.stats import norm, skew #for some statistics
Admission_test =  pd.read_csv('../input/Admission_Predict.csv')
Admission_test.head(5)
Admission_test.describe().T
print ("Size of train data : {}" .format(Admission_test.shape)) 
Admission_test.columns= [phrase.strip().replace(' ', '_') for phrase in Admission_test.columns]
Admission_test.head(5)
Admission_test.info()
Admission_test.plot(kind='scatter', x='GRE_Score', y='Chance_of_Admit', alpha=0.2)
pandas_profiling.ProfileReport(Admission_test)
#Handling Outliers - Method2
def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

Admission_test=Admission_test.apply(lambda x: outlier_capping(x))
Admission_test.describe().T
### most correlated features
corrmat = Admission_test.corr()
top_corr_features = corrmat.index[abs(corrmat["Chance_of_Admit"])>0.5]
plt.figure(figsize=(8,8))
g = sns.heatmap(Admission_test[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#correaltion charts
sns.set()
cols = ['Chance_of_Admit', 'Research', 'CGPA', 'LOR', 'SOP', 'University_Rating', 'TOEFL_Score', 'GRE_Score']
sns.pairplot(Admission_test[cols], size = 2.5)
plt.show();
def check_skewness(col):
    sns.distplot(Admission_test[col] , fit=norm);
    fig = plt.figure()
    res = stats.probplot(Admission_test[col], plot=plt)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(Admission_test[col])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    
check_skewness("Chance_of_Admit")
#Splitting the data
feature_columns = Admission_test.columns.difference( ['Chance_of_Admit','SOP'] )
feature_columns
from sklearn.model_selection import train_test_split 

train_X, test_X, train_y, test_y = train_test_split( Admission_test[feature_columns],
                                                  Admission_test['Chance_of_Admit'],
                                                  test_size = 0.3,
                                                  random_state = 123 )
print(len( train_X ))
print(len( test_X))
import statsmodels.api as sm
train_X = sm.add_constant(train_X)
lm=sm.OLS(train_y,train_X).fit()
print(lm.summary())
print('Parameters: ', lm.params)
print('R2: ', lm.rsquared)
from sklearn.model_selection import train_test_split
train ,test = train_test_split(Admission_test,test_size=0.3,random_state = 123 ) 
#=Dropping Serial No it is not providing any insight
#Save the 'Id' column
train_Serial = train['Serial_No.']
test_Serial = test['Serial_No.']


#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Serial_No.", axis = 1, inplace = True)
test.drop("Serial_No.", axis = 1, inplace = True)

train.head(5)
train.columns
train.columns= [phrase.strip().replace('-', '') for phrase in train.columns]
train.columns
all_columns = "+".join(train.columns.difference( ['Chance_of_Admit'] ))

print(all_columns)

my_formula = "Chance_of_Admit~" + all_columns

print(my_formula)
import statsmodels.formula.api as smf
lm=smf.ols(formula=my_formula, data=train).fit()
lm.summary()
import statsmodels as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
%%capture
#gather features
#features = "+".join(car_sales.columns - ["Sales_in_thousands"])

# get y and X dataframes based on this regression
y, X = dmatrices('Chance_of_Admit~CGPA+GRE_Score+LOR+Research+SOP+TOEFL_Score+University_Rating', Admission_test, return_type='dataframe')
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
X.head(5)
lm1=smf.ols('Chance_of_Admit~CGPA+GRE_Score+LOR+Research+TOEFL_Score+University_Rating', Admission_test).fit()
lm1.summary()
y_pred = lm1.predict(test_X)
y_pred
Final = pd.DataFrame()
Final['Serial_No.'] = test_Serial
Final['Chance_of_admit']  = test_y
Final['Pred_Chance_of_admit'] = y_pred
Final.to_csv('submission.csv',index=False)
# calculate these metrics by hand!
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(test_y, y_pred))
print('MSE:', metrics.mean_squared_error(test_y, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))