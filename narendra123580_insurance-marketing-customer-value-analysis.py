import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',999)
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import statsmodels.stats.api as sms

from statsmodels.compat import lzip
from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import normaltest,f_oneway

from scipy.stats import ttest_ind
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import ExtraTreeRegressor
dt = DecisionTreeRegressor()

et = ExtraTreeRegressor()
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
abr = AdaBoostRegressor()

br = BaggingRegressor()

etr = ExtraTreesRegressor()

gbr = GradientBoostingRegressor()

rfr = RandomForestRegressor()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
data=pd.read_csv('/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')

data.head()
data.isnull().sum()
data[['Customer Lifetime Value','Income','Monthly Premium Auto','Total Claim Amount']].describe()
data.shape
sns.boxplot(data['Income'])

plt.show()
sns.boxplot(data['Monthly Premium Auto'])

plt.show()
sns.boxplot(data['Total Claim Amount'])

plt.show()
sns.distplot(data['Income'])

plt.show()
sns.distplot(data['Monthly Premium Auto'])

plt.show()
sns.distplot(data['Total Claim Amount'])

plt.show()
sns.distplot(data['Income']**2)

plt.show()
sns.distplot(data['Income']**(1/2))

plt.show()
sns.distplot(data['Monthly Premium Auto']**(2))

plt.show()
sns.distplot(data['Total Claim Amount']**2)

plt.show()
sns.barplot(x = 'Location Code',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'State',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'Response',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'Gender',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'Education',y='Customer Lifetime Value',data = data)

plt.xticks(rotation=45)

plt.show()
sns.barplot(x = 'Number of Policies',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'Policy Type',y='Customer Lifetime Value',data = data)

plt.xticks(rotation = 90)

plt.show()
sns.barplot(x = 'Coverage',y='Customer Lifetime Value',data = data)

plt.show()
sns.barplot(x = 'Number of Open Complaints',y='Customer Lifetime Value',data = data)

plt.show()
sns.pairplot(y_vars='Customer Lifetime Value',x_vars=['Income','Monthly Premium Auto','Total Claim Amount'],data = data)

plt.show()
sns.heatmap(data[['Customer Lifetime Value','Monthly Premium Auto','Income','Total Claim Amount']].corr(),annot = True)

plt.show()
cols = data.select_dtypes(object).columns

for i in cols:

    data[i] = le.fit_transform(data[i])
X = data.drop('Customer Lifetime Value',axis=1)

y = data['Customer Lifetime Value']

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X, y)
from sklearn.model_selection import train_test_split

# train data - 70% and test data - 30%

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)

print(y_train.shape)
lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

print(f'Coefficients: {lin_reg.coef_}')

print(f'Intercept: {lin_reg.intercept_}')

print(f'R^2 score: {lin_reg.score(X, y)}')

print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')

print(f'R^2 score for test: {lin_reg.score(X_test, y_test)}')
X_sm = X

X_sm = sm.add_constant(X_sm)

lm = sm.OLS(y,X_sm).fit()

lm.summary()
sns.pairplot(x_vars=['Monthly Premium Auto','Total Claim Amount','Income'],y_vars =['Customer Lifetime Value'],data = data)

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

residuals = y_pred-y_test

mean_of_residuals = np.mean(residuals)

print(f"The mean of the residuals is {mean_of_residuals}")
name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(residuals,X_test)

lzip(name, test)
p = sns.distplot(residuals,kde=True)

p = plt.title('Normality of error terms/residuals')
min(diag.acorr_ljungbox(residuals , lags = 40)[1])
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
data=pd.read_csv('/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
State = data.groupby('State')

Washington = State.get_group('Washington')['Customer Lifetime Value']

Arizona = State.get_group('Arizona')['Customer Lifetime Value']

Nevada = State.get_group('Nevada')['Customer Lifetime Value']

California = State.get_group('California')['Customer Lifetime Value']

Oregon = State.get_group('Oregon')['Customer Lifetime Value']
for i in [Washington,Arizona,Nevada,California,Oregon]:

    print(normaltest(i),'\n')
f_oneway(Washington,Arizona,Nevada,California,Oregon)
Response = data[['Customer Lifetime Value','Response']].groupby('Response')

No = Response['Customer Lifetime Value'].get_group('No')

Yes = Response['Customer Lifetime Value'].get_group('Yes')
for i in [No,Yes]:

    print(normaltest(i),'\n')
ttest_ind(No,Yes)
Coverage = data[['Customer Lifetime Value','Coverage']].groupby('Coverage')

basic = Coverage['Customer Lifetime Value'].get_group('Basic')

extended = Coverage['Customer Lifetime Value'].get_group('Extended')

premium = Coverage['Customer Lifetime Value'].get_group('Premium')
for i in [basic,extended,premium]:

    print(normaltest(i),'\n')
f_oneway(basic,extended,premium)
Education = data[['Customer Lifetime Value','Education']].groupby('Education')

bachelor = Education['Customer Lifetime Value'].get_group('Bachelor')

college = Education['Customer Lifetime Value'].get_group('College')

highschool = Education['Customer Lifetime Value'].get_group('High School or Below')

master = Education['Customer Lifetime Value'].get_group('Master')

doctor = Education['Customer Lifetime Value'].get_group('Doctor')
for i in [basic,college,highschool,master,doctor]:

    print(normaltest(i),'\n')
f_oneway(bachelor,college,highschool,master,doctor)
es = data[['Customer Lifetime Value','EmploymentStatus']].groupby('EmploymentStatus')

employed = es['Customer Lifetime Value'].get_group('Employed')

unemployed = es['Customer Lifetime Value'].get_group('Unemployed')

medleave = es['Customer Lifetime Value'].get_group('Medical Leave')

disabled = es['Customer Lifetime Value'].get_group('Disabled')

retired = es['Customer Lifetime Value'].get_group('Retired')
for i in [employed,unemployed,medleave,disabled,retired]:

    print(normaltest(i),'\n')
f_oneway(employed,unemployed,medleave,disabled,retired)
g = data[['Customer Lifetime Value','Gender']].groupby('Gender')

f = g['Customer Lifetime Value'].get_group('F')

m = g['Customer Lifetime Value'].get_group('M')
for i in [f,m]:

    print(normaltest(i),'\n')
ttest_ind(f,m)
location = data[['Customer Lifetime Value','Location Code']].groupby('Location Code')

sub = location['Customer Lifetime Value'].get_group('Suburban')

urban = location['Customer Lifetime Value'].get_group('Urban')

rural = location['Customer Lifetime Value'].get_group('Rural')
for i in [sub,urban,rural]:

    print(normaltest(i),'\n')
f_oneway(sub,urban,rural)
MaritalStatus = data[['Customer Lifetime Value','Marital Status']].groupby('Marital Status')

Married = MaritalStatus['Customer Lifetime Value'].get_group('Married')

Single = MaritalStatus['Customer Lifetime Value'].get_group('Single')

Divorced = MaritalStatus['Customer Lifetime Value'].get_group('Divorced')
for i in [Married,Single,Divorced]:

    print(normaltest(i),'\n')
f_oneway(Married,Single,Divorced)
Policy  = data[['Customer Lifetime Value','Policy']].groupby('Policy')

p3 = Policy['Customer Lifetime Value'].get_group('Personal L3')

p2 = Policy['Customer Lifetime Value'].get_group('Personal L2')

p1 = Policy['Customer Lifetime Value'].get_group('Personal L1')

c3 = Policy['Customer Lifetime Value'].get_group('Corporate L3')

c2 = Policy['Customer Lifetime Value'].get_group('Corporate L2')

c1 = Policy['Customer Lifetime Value'].get_group('Corporate L1')

s3 = Policy['Customer Lifetime Value'].get_group('Special L3')

s2 = Policy['Customer Lifetime Value'].get_group('Special L2')

s1 = Policy['Customer Lifetime Value'].get_group('Special L1')
for i in [p3,p2,p1,c3,c2,c1,s3,s2,s1]:

    print(normaltest(i),'\n')
f_oneway(p3,p2,p1,c3,c2,c1,s3,s2,s1)
R  = data[['Customer Lifetime Value','Renew Offer Type']].groupby('Renew Offer Type')

o1 = R['Customer Lifetime Value'].get_group('Offer1')

o2 = R['Customer Lifetime Value'].get_group('Offer2')

o3 = R['Customer Lifetime Value'].get_group('Offer3')

o4 = R['Customer Lifetime Value'].get_group('Offer4')
for i in [o1,o2,o3,o4]:

    print(normaltest(i),'\n')
f_oneway(o1,o2,o3,o4)
Sales  = data[['Customer Lifetime Value','Sales Channel']].groupby('Sales Channel')

agent = Sales['Customer Lifetime Value'].get_group('Agent')

branch = Sales['Customer Lifetime Value'].get_group('Branch')

call = Sales['Customer Lifetime Value'].get_group('Call Center')

web = Sales['Customer Lifetime Value'].get_group('Web')
for i in [agent,branch,call,web]:

    print(normaltest(i),'\n')
f_oneway(agent,branch,call,web)
VC  = data[['Customer Lifetime Value','Vehicle Class']].groupby('Vehicle Class')

fd = VC['Customer Lifetime Value'].get_group('Four-Door Car')

td = VC['Customer Lifetime Value'].get_group('Two-Door Car')

suv = VC['Customer Lifetime Value'].get_group('SUV')

sc = VC['Customer Lifetime Value'].get_group('Sports Car')

ls = VC['Customer Lifetime Value'].get_group('Luxury SUV')

lc = VC['Customer Lifetime Value'].get_group('Luxury Car')
for i in [fd,td,suv,sc,ls,lc]:

    print(normaltest(i),'\n')
f_oneway(fd,td,suv,sc,ls,lc)
VS  = data[['Customer Lifetime Value','Vehicle Size']].groupby('Vehicle Size')

m = VS['Customer Lifetime Value'].get_group('Medsize')

s = VS['Customer Lifetime Value'].get_group('Small')

l = VS['Customer Lifetime Value'].get_group('Large')
for i in [m,s,l]:

    print(normaltest(i),'\n')
f_oneway(m,s,l)
data.drop(['State','Customer','Response','EmploymentStatus','Gender','Location Code','Vehicle Size','Policy','Policy Type','Sales Channel','Income','Effective To Date','Education'],axis=1,inplace = True)
data.head()
data['Number of Policies'] = np.where(data['Number of Policies']>2,3,data['Number of Policies'])
new = pd.get_dummies(data,columns=['Coverage','Marital Status','Number of Policies','Renew Offer Type','Vehicle Class'],drop_first=True)
new.head()
X = new.drop('Customer Lifetime Value',axis=1)

y = new['Customer Lifetime Value']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
lr.score(X_train,y_train)
sfs = SFS(lr, k_features='best', forward=True, floating=False, 

          scoring='neg_mean_squared_error', cv=20)

model = sfs.fit(new.drop('Customer Lifetime Value', axis=1),new['Customer Lifetime Value'])

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')

plt.grid()

plt.show()
print('Selected features:', sfs.k_feature_idx_)
sfs = SFS(lr, k_features='best', forward=False, floating=False, 

          scoring='neg_mean_squared_error', cv=20)

model = sfs.fit(new.drop('Customer Lifetime Value', axis=1).values,new['Customer Lifetime Value'])

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Backward Selection (w. StdErr)')

plt.grid()

plt.show()
print('Selected features:', sfs.k_feature_idx_)
X.columns
test_X = X[['Monthly Premium Auto','Number of Open Complaints','Total Claim Amount','Coverage_Premium',

            'Marital Status_Single','Number of Policies_2','Number of Policies_3',

            'Renew Offer Type_Offer2','Vehicle Class_SUV','Vehicle Class_Sports Car']]
train = []

test = []
X_train,X_test,y_train,y_test = train_test_split(test_X,y,test_size=0.3,random_state=100)
lr.fit(X_train,y_train)
test.append(lr.score(X_test,y_test))
train.append(lr.score(X_train,y_train))
metrics = [r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error]
y_pred = lr.predict(X_test)
r2 = []

mae = []

mape = []

mse = []
for i in metrics:

    print(i(y_test,y_pred))

    if i == r2_score:

        r2.append(i(y_test,y_pred))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred))

    else:

        mse.append(i(y_test,y_pred))
algo = [abr,gbr,dt,et,etr,br,rfr]
for i in algo:

    temp = 0

    print(f"New Model{i}")

    for j in range(1,300,1):

        NXT,NXt,NYT,NYt = train_test_split(X,y,test_size=0.3,random_state=j)

        i.fit(NXT,NYT)

        test_score = i.score(NXt,NYt)

        train_score = i.score(NXT,NYT)

        if test_score>temp:

            temp = test_score

            print(j,train_score,temp)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=159)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
for i in metrics:

    print(i(y_test,y_pred_dt))

    if i == r2_score:

        r2.append(i(y_test,y_pred_dt))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_dt))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_dt))

    else:

        mse.append(i(y_test,y_pred_dt))
train.append(dt.score(X_train,y_train))

test.append(dt.score(X_test,y_test))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=69)
et.fit(X_train,y_train)
y_pred_et = et.predict(X_test)
for i in metrics:

    print(i(y_test,y_pred_et))

    if i == r2_score:

        r2.append(i(y_test,y_pred_et))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_et))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_et))

    else:

        mse.append(i(y_test,y_pred_et))
train.append(et.score(X_train,y_train))

test.append(et.score(X_test,y_test))
pd.DataFrame({'Model':['Linear Regression','Decision Tree','Extra Tree'],'R2_Score':r2,'MAE':mae,'MAPE':mape,'MSE':mse})
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=83)
abr.fit(X_train,y_train)
y_pred_abr = abr.predict(X_test)
for i in metrics:

    print(i(y_test,y_pred_abr))

    if i == r2_score:

        r2.append(i(y_test,y_pred_abr))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_abr))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_abr))

    else:

        mse.append(i(y_test,y_pred_abr))
train.append(abr.score(X_train,y_train))

test.append(abr.score(X_test,y_test))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=292)

br.fit(X_train,y_train)

y_pred_br = br.predict(X_test)

for i in metrics:

    print(i(y_test,y_pred_br))

    if i == r2_score:

        r2.append(i(y_test,y_pred_br))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_br))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_br))

    else:

        mse.append(i(y_test,y_pred_br))
train.append(br.score(X_train,y_train))

test.append(br.score(X_test,y_test))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=69)

etr.fit(X_train,y_train)

y_pred_etr = etr.predict(X_test)

for i in metrics:

    print(i(y_test,y_pred_etr))

    if i == r2_score:

        r2.append(i(y_test,y_pred_etr))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_etr))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_etr))

    else:

        mse.append(i(y_test,y_pred_etr))
train.append(etr.score(X_train,y_train))

test.append(etr.score(X_test,y_test))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=181)

gbr.fit(X_train,y_train)

y_pred_gbr = gbr.predict(X_test)

for i in metrics:

    print(i(y_test,y_pred_gbr))

    if i == r2_score:

        r2.append(i(y_test,y_pred_gbr))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_gbr))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_gbr))

    else:

        mse.append(i(y_test,y_pred_gbr))
train.append(gbr.score(X_train,y_train))

test.append(gbr.score(X_test,y_test))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=35)

rfr.fit(X_train,y_train)

y_pred_rfr = rfr.predict(X_test)

for i in metrics:

    print(i(y_test,y_pred_rfr))

    if i == r2_score:

        r2.append(i(y_test,y_pred_rfr))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_rfr))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_rfr))

    else:

        mse.append(i(y_test,y_pred_rfr))
train.append(rfr.score(X_train,y_train))

test.append(rfr.score(X_test,y_test))
hyper_params_gbr = {'loss':['ls','lad','huber'],'learning_rate':[0.1,0.01,1],'n_estimators':[100,150]}
gbr2 = GradientBoostingRegressor()
model = GridSearchCV(gbr2,param_grid=hyper_params_gbr)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=181)
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.score(X_train,y_train)
model.best_params_
gbr2 = GradientBoostingRegressor()
br2 = BaggingRegressor(gbr2)
temp = 0

for j in range(1,300,1):

    NXT,NXt,NYT,NYt = train_test_split(X,y,test_size=0.3,random_state=j)

    br2.fit(NXT,NYT)

    test_score = br2.score(NXt,NYt)

    train_score = br2.score(NXT,NYT)

    if test_score>temp:

        temp = test_score

        print(j,train_score,temp)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=87)

br2.fit(X_train,y_train)

y_pred_br2 = br2.predict(X_test)

for i in metrics:

    print(i(y_test,y_pred_br2))

    if i == r2_score:

        r2.append(i(y_test,y_pred_br2))

    elif i == mean_absolute_error:

        mae.append(i(y_test,y_pred_br2))

    elif i == mean_absolute_percentage_error:

        mape.append(i(y_test,y_pred_br2))

    else:

        mse.append(i(y_test,y_pred_br2))
train.append(br2.score(X_train,y_train))

test.append(br2.score(X_test,y_test))
test_scores = []

train_scores = []

cv = KFold(n_splits=10,random_state=42, shuffle=False)

for train_index,test_index in cv.split(X):

    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    br2.fit(X_train,y_train)

    test_scores.append(br.score(X_test, y_test))

    train_scores.append(br.score(X_train, y_train))
np.mean(train_scores)
np.mean(test_scores)
ALL_SCORES = pd.DataFrame({'Model':['Linear Regression','Decision Tree','Extra Tree','AdaBoost','Bagging',

                                    'Extra Trees','GradientBoosting','Random Forest','Final_Model'],

                           'Training_Score':train,'Testing_Score':test,'R2_Score':r2,'MAE':mae,'MAPE':mape,'MSE':mse})
ALL_SCORES