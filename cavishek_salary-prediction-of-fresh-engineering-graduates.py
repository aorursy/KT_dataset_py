import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)   

pd.set_option('display.width', 1000)
train = pd.read_excel('train.xlsx') 

test = pd.read_excel('test.xlsx')

results = pd.read_excel('results.xlsx')
train.head()
test.head()
data = pd.concat([train,test])
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data.head()
data.tail()
data.shape
data.columns
data.drop(['ID','DOJ','DOL','Designation','JobCity','CollegeCityID'],axis=1,inplace=True)
data.shape
data.columns
#treating for CBSE

data['10board'] = data['10board'].replace({'cbse':'CBSE',

                                             'central board of secondary education':'CBSE',

                                             'cbse board':'CBSE',

                                             'central board of secondary education, new delhi':'CBSE',

                                             'delhi public school':'CBSE',

                                             'c b s e':'CBSE',

                                             'central board of secondary education(cbse)':'CBSE',

                                             'dav public school,hehal':'CBSE',

                                             'cbse ':'CBSE',

                                             'cbse[gulf zone]':'CBSE',

                                             'cbsc':'CBSE'})
#treating for ICSE

data['10board'].replace({'icse':'ICSE',

                          'icse board':'ICSE',

                          'icse board , new delhi':'ICSE'},inplace=True)
#treating for State Board

stateboards = []



for i in data['10board']:

    if i!='CBSE' and i!='ICSE':

        stateboards.append(i)



#print(stateboards)
#replacing all values with 'State Board'

data['10board'].replace(stateboards,'State_Board',inplace=True)  
data['10board'].value_counts()
b = []

for i in data['10board']:

    b.append(i)

y = np.array(b)

print(np.unique(y))
#treating for CBSE

data['12board'] = data['12board'].replace({'cbse':'CBSE',

                                             'central board of secondary education':'CBSE',

                                             'cbse board':'CBSE',

                                             'central board of secondary education, new delhi':'CBSE',

                                             'delhi public school':'CBSE',

                                             'c b s e':'CBSE',

                                             'central board of school education':'CBSE',

                                             'central board of secondary education(cbse)':'CBSE',

                                             'dav public school,hehal':'CBSE',

                                             'cbse ':'CBSE',

                                             'cbse[gulf zone]':'CBSE',

                                             'cbsc':'CBSE'})
#treating for ICSE

data['12board'].replace({'icse':'ICSE',

                          'icse board':'ICSE',

                          'icse board , new delhi':'ICSE'},inplace=True)
#treating for State Board

stateboardss = []



for i in data['12board']:

    if i!='CBSE' and i!='ICSE':

        stateboardss.append(i)



#print(stateboardss)
#replacing all values with 'State Board'

data['12board'].replace(stateboardss,'State_Board',inplace=True)  
data['12board'].value_counts()
data.head()
data.shape
c = []

for i in data['12board']:

    c.append(i)

z = np.array(c)

print(np.unique(z))
len(data.loc[data['collegeGPA']>10])
len(data['collegeGPA'])
for i in range(5498):

    if data['collegeGPA'].values[i]>10:

        data.collegeGPA.values[i] = data.collegeGPA.values[i]/10
len(data.loc[data['collegeGPA']>10])
len(data.loc[data['ElectronicsAndSemicon']==-1])/len(data['ElectronicsAndSemicon'])*100
len(data.loc[data['ComputerScience']==-1])/len(data['ComputerScience'])*100
len(data.loc[data['MechanicalEngg']==-1])/len(data['MechanicalEngg'])*100
len(data.loc[data['ElectricalEngg']==-1])/len(data['ElectricalEngg'])*100
len(data.loc[data['TelecomEngg']==-1])/len(data['TelecomEngg'])*100
len(data.loc[data['CivilEngg']==-1])/len(data['CivilEngg'])*100
data.drop(['ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg'],axis=1,inplace=True)
data.shape
len(data.loc[data['ComputerProgramming']==-1])/len(data['ComputerProgramming'])*100
data['ComputerProgramming'].corr(data['Salary'])
len(data.loc[data['Domain']==-1])/len(data['Domain'])*100
data['Domain'].corr(data['Salary'])
data['ComputerProgramming'] = data['ComputerProgramming'].replace(-1,np.nan)
data['ComputerProgramming'].isnull().sum()
data['ComputerProgramming'] = data['ComputerProgramming'].replace(np.nan,data['ComputerProgramming'].mean())
data['ComputerProgramming'].isnull().sum()
data['ComputerProgramming'].corr(data['Salary'])
data['Domain'] = data['Domain'].replace(-1,np.nan)
data['Domain'].isnull().sum()
data['Domain'] = data['Domain'].replace(np.nan,data['Domain'].mean())
data['Domain'].isnull().sum()
data['Domain'].corr(data['Salary'])
data['CollegeTier'] = data['CollegeTier'].replace(2,0)
data['CollegeTier'].value_counts()
data['DOB'] = pd.to_datetime(data['DOB'])
data['DOB'] = pd.DatetimeIndex(data['DOB']).year
data['DOB'] = data['DOB'].astype(int)
data.Specialization = data.Specialization.replace({'construction technology and management':'civil engineering'})
data.Specialization = data.Specialization.replace({'computer application':'computer science and engineering',

                                                     'electronics and computer engineering':'computer science and engineering',

                                                  'computer science and technology':'computer science and engineering',

                                                     'software engineering':'computer science and engineering',

                                                    'computer science':'computer science and engineering',

                                                   'computer engineering':'computer science and engineering',

                                                     'electronics and computer engineering':'computer science and engineering'})
data.Specialization = data.Specialization.replace({'computer science & engineering':'computer science and engineering'})
data.Specialization = data.Specialization.replace({'electronics engineering':'electronics and communication engineering',

                                                       'communication engineering':'electronics and communication engineering'})
data.Specialization = data.Specialization.replace({'electronics':'electronics and communication engineering'})
data.Specialization = data.Specialization.replace({'electronics & instrumentation eng':'electronics and instrumentation engineering',

                                                    'applied electronics and instrumentation':'electronics and instrumentation engineering',

                                                    'instrumentation engineering':'electronics and instrumentation engineering'})
data.Specialization = data.Specialization.replace({'telecommunication engineering':'electronics and telecommunication engineering',

                                                   'electronics and telecommunication':'electronics and telecommunication engineering',

                                                   'electronics & telecommunications':'electronics and telecommunication engineering'})
data.Specialization = data.Specialization.replace({'information science engineering':'information technology'})
data.Specialization = data.Specialization.replace({'information science':'information technology'})
data.Specialization = data.Specialization.replace({'information & communication technology':'information technology'})
data.Specialization = data.Specialization.replace({'vlsi design and cad':'computer aided design',

                                                     'cad / cam':'computer aided design'})
data.Specialization = data.Specialization.replace({'industrial & production engineering':'industrial and production engineering',

                                                    'industrial engineering and management':'industrial and production engineering'})
data.Specialization = data.Specialization.replace({'biotechnology':'biotechnology engineering'})
data.Specialization = data.Specialization.replace({'electronics and telecommunication engineering':'electronics and communication engineering'})
data.Specialization = data.Specialization.replace({'electronics and instrumentation engineering':'electronics and communication engineering'})
data.Specialization = data.Specialization.replace({'electrical engineering':'electronics and electrical engineering'})
data['Specialization'].value_counts()
data.drop(data[data.Specialization == 'polymer technology'].index, inplace=True)
data.drop(data[data.Specialization == 'environment science'].index, inplace=True)
data.drop(data[data.Specialization == 'textile engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'power systems and automation'].index, inplace=True)
data.drop(data[data.Specialization == 'operational research'].index, inplace=True)
data.drop(data[data.Specialization == 'aerospace engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'embedded systems technology'].index, inplace=True)
data.drop(data[data.Specialization == 'ceramic engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'mechatronics'].index, inplace=True)

data.drop(data[data.Specialization == 'computer aided design'].index, inplace=True)

data.drop(data[data.Specialization == 'electrical and power engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'automobile/automotive engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'metallurgical engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'chemical engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'biomedical engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'aeronautical engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'automobile engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'mechatronics engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'biotechnology engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'industrial and production engineering'].index, inplace=True)

data.drop(data[data.Specialization == 'other'].index, inplace=True)
data['Specialization'].value_counts()
data.head()
data.shape
data['CollegeCityTier'].value_counts()
data['CollegeCityTier'] = data['CollegeCityTier'].replace({0:'A',1:'B'})
data['CollegeCityTier'].value_counts()
data['Degree'].value_counts()
data['Degree'] = data['Degree'].replace({'B.Tech/B.E.':0,'MCA':1,'M.Sc. (Tech.)':3,'M.Tech./M.E.':4})
data['Degree'].value_counts()
data.info()
dummy_data = pd.get_dummies(data,drop_first=True)
dummy_data.shape
dummy_data.head()
new_train = dummy_data.loc[:3998]
new_train.shape
new_train.tail()
new_test = dummy_data.loc[3999:]
new_test.shape
new_test.tail()
x = new_train.drop(['Salary'], axis=1)

y = new_train['Salary']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 0.3, random_state = 3)
#xtrain = dummy_data.iloc[0:3898,1:]
#xtest = dummy_data.iloc[3999:,1:]
#ytrain = dummy_data.iloc[0:3898,0:1]
#ytest = dummy_data.iloc[3999:,0:1]
xtrain.shape
xtest.shape
ytrain.shape
ytest.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import r2_score
r_sqrd = r2_score(ytest,ypred)

r_sqrd
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(ytest,ypred))
adj_r_sqrd = 1-(1-r_sqrd)*((1179-1)/(1179-57-1))

adj_r_sqrd
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(ytest,ypred)

mape
Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
data.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]).T
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,3))

sns.boxplot(x=data['10percentage'])

plt.show()
lower_limit = 72 - 1.5 * 13.6

lower_limit
data['10percentage']=data["10percentage"].map(lambda x:51 if x <51 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['12percentage'])
lower_limit = 66.4 - 1.5 * 16.4065

lower_limit
np.where(data['12percentage']<41.79)
data['12percentage'].min()
data['12percentage']=data["12percentage"].map(lambda x:41.79025 if x <41.79025 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['collegeGPA'])
lower_limit = 6.69 - 1.5 * 0.95

lower_limit
np.where(data['collegeGPA']<5.265)
np.where(data['collegeGPA']<5)
upper_limit = 7.64 + 1.5 * 0.95

upper_limit
data['collegeGPA'].max()
np.where(data['collegeGPA']>9.065)
data['collegeGPA'].min()
data['collegeGPA']=data["collegeGPA"].map(lambda x:5 if x <5 else x)
#data['collegeGPA']=data["collegeGPA"].map(lambda x:9.99 if x >9.99 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['English'])
lower_limit = 425 - 1.5 * 145

lower_limit
np.where(data['English']<207.5)
upper_limit = 570 + 1.5 * 145

upper_limit
np.where(data['English']>787.5)
#data['English']=data["English"].map(lambda x:207.5 if x <207.5 else x)
#data['English']=data["English"].map(lambda x:787.5 if x >787.5 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['Quant'])
lower_limit = 430 - 1.5 * 165

lower_limit
upper_limit = 595 + 1.5 * 165

upper_limit
#data['Quant']=data["Quant"].map(lambda x:182.5 if x <182.5 else x)
#data['English']=data["English"].map(lambda x:842.5 if x >842.5 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['Logical'])
lower_limit = 445 - 1.5 * 120

lower_limit
upper_limit = 565 + 1.5 * 120

upper_limit
#data['Logical']=data["Logical"].map(lambda x:265 if x <265 else x)
#data['Logical']=data["Logical"].map(lambda x:745 if x >745 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['Domain'])
plt.figure(figsize=(15,3))

sns.boxplot(x=data['conscientiousness'])
lower_limit = -0.726400 - 1.5 * 1.4291

lower_limit
upper_limit = 0.702700 + 1.5 * 1.4291

upper_limit
#data['conscientiousness']=data["conscientiousness"].map(lambda x:-2.87005 if x <-2.87005 else x)
#data['conscientiousness']=data["conscientiousness"].map(lambda x:2.84635 if x >2.84635 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['agreeableness'])
lower_limit = -0.287100 - 1.5 * 1.0999

lower_limit
upper_limit = 0.812800 + 1.5 * 1.0999

upper_limit
#data['agreeableness']=data["agreeableness"].map(lambda x:-1.93695 if x <-1.93695 else x)
#data['agreeableness']=data["agreeableness"].map(lambda x:2.46265 if x <2.46265 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['extraversion'])
lower_limit = -0.7264 - 1.5 * 1.2768

lower_limit
upper_limit = 0.702700 + 1.5 * 1.2768

upper_limit
#data['extraversion']=data["extraversion"].map(lambda x:-2.6416 if x <-2.6416 else x)
#data['extraversion']=data["extraversion"].map(lambda x:2.6179 if x >2.6179 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['nueroticism'])
upper_limit = 0.5262 + 1.5 * 1.3944

upper_limit
#data['nueroticism']=data["nueroticism"].map(lambda x:2.6178 if x > 2.6178 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['openess_to_experience'])
lower_limit = -0.669200 - 1.5 * 1.1716

lower_limit
#data['openess_to_experience']=data["openess_to_experience"].map(lambda x:-2.4266 if x < -2.4266 else x)
plt.figure(figsize=(15,3))

sns.boxplot(x=data['ComputerProgramming'])
lower_limit = 405 - 1.5 * 90

lower_limit
upper_limit = 495 + 1.5 * 90

upper_limit
#data['ComputerProgramming']=data["ComputerProgramming"].map(lambda x:270 if x < 270 else x)
#data['ComputerProgramming']=data["ComputerProgramming"].map(lambda x:630 if x > 630 else x)
data.columns
data['GraduationYear'] = data['GraduationYear'].astype(int)

data['DOB'] = data['DOB'].astype(int)

data['GraduationAge'] = data['GraduationYear'] - data['DOB']
data['GraduationAge'].value_counts()
data.info()
data['12_age'] = data['12graduation'] - data['DOB']
data['12graduation'].value_counts()
data.drop(['DOB'],axis=1,inplace = True)
new_data = pd.get_dummies(data,drop_first=True) 

new_data.shape
n_train = new_data.loc[:3998]
n_train.tail()
n_test = new_data.loc[3999:]
n_test.head()
x = n_train.drop(['Salary'],axis=1)

y = n_train['Salary']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 3,test_size=0.3)
sns.distplot(n_train['Salary'])
n_train['Salary_log'] = np.log(n_train['Salary'])
sns.distplot(n_train['Salary_log'])
plt.figure(figsize=(15,3))

sns.boxplot(x=n_train['Salary'])
n_train.drop(n_train[n_train['Salary'] > 2000000].index, inplace = True)
sns.distplot(n_train['Salary'])
n_train['Salary_log'] = np.log(n_train['Salary'])
sns.distplot(n_train['Salary_log'])
n_train.info()
x = n_train.drop(['Salary_log','Salary'],axis=1)

y = n_train['Salary_log']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 3,test_size=0.3)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

model1.fit(xtrain,ytrain)

model1_pred = model1.predict(xtest)
r_sqrd = r2_score(ytest,model1_pred)

r_sqrd
np.sqrt(mean_squared_error(ytest,model1_pred))
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
mape = mean_absolute_percentage_error(ytest,model1_pred)

mape
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()

model2.fit(xtrain,ytrain)

model2_pred = model2.predict(xtest)
r_sqrd = r2_score(ytest,model2_pred)

r_sqrd
np.sqrt(mean_squared_error(ytest,model2_pred))
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
mean_absolute_percentage_error(ytest,model2_pred)
from sklearn.ensemble import GradientBoostingRegressor
model3 = GradientBoostingRegressor()

model3.fit(xtrain,ytrain)

model3_pred = model3.predict(xtest)
r_sqrd =  r2_score(ytest,model3_pred)

r_sqrd
np.sqrt(mean_squared_error(ytest,model3_pred))
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
mean_absolute_percentage_error(ytest,model3_pred)
model3_pred
np.exp(model3_pred)
from sklearn.neighbors import KNeighborsRegressor
model4 = KNeighborsRegressor()

model4.fit(xtrain,ytrain)

model4_pred = model4.predict(xtest)
r_sqrd =  r2_score(ytest,model4_pred)

r_sqrd
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
np.sqrt(mean_squared_error(ytest,model4_pred))
mean_absolute_percentage_error(ytest,model4_pred)
from sklearn.tree import DecisionTreeRegressor
model5 = DecisionTreeRegressor()

model5.fit(xtrain,ytrain)

model5_pred = model5.predict(xtest)
r_sqrd =  r2_score(ytest,model5_pred)

r_sqrd
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
np.sqrt(mean_squared_error(ytest,model5_pred))
mean_absolute_percentage_error(ytest,model5_pred)
from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=0.3,random_state=3)
model6 = LinearRegression()

model6.fit(x_train, y_train)

model6_pred = model6.predict(x_test)
r_sqrd =  r2_score(y_test,model6_pred)

r_sqrd
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
np.sqrt(mean_squared_error(y_test,model6_pred))
mean_absolute_percentage_error(y_test,model6_pred)
from scipy.stats import zscore
z_data = n_train.copy()
z_data.head()
X = z_data.drop(['Salary','Salary_log'],axis=1)
Y = z_data['Salary_log']
X[['10percentage', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience']] = X[['10percentage', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience']].apply(zscore)
X.head()
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,random_state = 3, test_size = 0.3)
Xtrain.head()
model7 = LinearRegression()

model7.fit(Xtrain,Ytrain)

model7_pred = model7.predict(Xtest)
r_sqrd =  r2_score(Ytest,model7_pred)

r_sqrd
adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))

adj_r_sqrd
np.sqrt(mean_squared_error(Ytest,model7_pred))
mean_absolute_percentage_error(Ytest,model7_pred)
def multi_collinearity(X):

    

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    VIF=pd.DataFrame()

    VIF['columns']=X.columns

    VIF['vif']=vif

    return(VIF)
v = multi_collinearity(n_train)
v[v['vif']>5]
GradientBoostingRegressor

parameters = {

    'n_estimators': [50,100,150],

    'max_depth': [5,8,12,16],

    'min_samples_split' : [50,100,120],

    'min_samples_leaf': [20,30,50],

    'max_features' : [6,40,20,60,80,100,104]}

model9=GradientBoostingRegressor()

clf = GridSearchCV(model9,parameters,cv=10)

clf.fit(xtrain,ytrain)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant
num_data = data._get_numeric_data()
num_data.head()
X = add_constant(num_data)
vif = pd.DataFrame([variance_inflation_factor(X.values, i) 

               for i in range(X.shape[1])], 

              index=X.columns)
vif