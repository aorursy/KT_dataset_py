import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from statsmodels.formula.api import ols
#Importing neccessary plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

df.head(10)
type(df)
print('Descriptive Statastics of our Data:')

df.describe().T
print('Showing Meta Data :')

df.info()
df.columns
#Renaming columns 

df.columns = ['sno','GRE','TOEFL','university_rating','SOP','LOR','CGPA','research','admit_chance']

df.head()
#Checking for missing values

pd.isnull(df).sum()
sns.pairplot(data=df,diag_kind='kde')
df[['GRE','TOEFL','university_rating','CGPA','SOP','LOR','research']].hist(figsize=(10,8),bins=15,linewidth='1',edgecolor='black')

plt.tight_layout()

plt.show()
df.research.value_counts()
#Chances of admission wrt research

chances=df.groupby('research')['admit_chance'].median()

print(chances)

sns.factorplot('research','admit_chance',data=df)

plt.show()
sns.regplot(x="GRE",y="CGPA", data=df,line_kws={'color':'red'})

plt.title("GRE Score vs CGPA")

plt.show()
sns.regplot(x="GRE",y="TOEFL", data=df,line_kws={'color':'red'})

plt.title("GRE Score vs TOEFL Score")

plt.show()
df.university_rating.value_counts()
sns.countplot(df.university_rating)
sns.scatterplot(x="CGPA", y="university_rating", hue="research", data=df)

plt.show()
print('Avg. GRE scores based on University Ratings')



pd.DataFrame(df.groupby('university_rating')['GRE'].mean())
print('Avg. TOEFL scores based on University Ratings')



pd.DataFrame(df.groupby('university_rating')['TOEFL'].mean())
df.groupby('university_rating')[['SOP','LOR','CGPA']].mean()
sns.regplot(x="CGPA",y="admit_chance", data=df,line_kws={'color':'red'})

plt.title("CGPA vs Chance of Admit")

plt.show()
#The mean values for getting admisssion chances greater than 80%



admt_sort = df.sort_values(by=df.columns[-1],ascending=False)



admt_sort[(admt_sort['admit_chance']>0.80)].mean().reset_index().T
cat = ['university_rating','research']

cont = ['GRE','TOEFL','SOP','LOR','CGPA','admit_chance']
print('Correlation Heat map of the data')

plt.figure(figsize=(12,8))

sns.heatmap(df[cont].corr(),annot=True,fmt='.2f',vmin=-1,vmax=1)

plt.show()
sns.boxplot(x='university_rating',y='admit_chance',data=df)
sns.boxplot(x='research',y='admit_chance',data=df)
x = df.drop(['sno','admit_chance'],axis=1)

x.head()
y=df['admit_chance']
from sklearn.preprocessing import StandardScaler,MinMaxScaler

x_std = StandardScaler().fit_transform(x)

x_std = pd.DataFrame(x_std,columns=x.columns)

x_std.head()
x_std1 = MinMaxScaler().fit_transform(x)

x_std1 = pd.DataFrame(x_std1,columns=x.columns)

x_std1.head()
plt.figure(figsize=(14,8))

sns.boxplot(data=x_std,orient='h')

plt.show()
std_df=pd.concat([x_std,y],axis=1)

print(std_df.shape)

std_df.head()
std_df1 = pd.concat([x_std1,y],axis=1)

print(std_df1.shape)

std_df1.head()
from statsmodels.formula.api import ols

M2 = ols('admit_chance~GRE+TOEFL+SOP+LOR+CGPA+university_rating+research',std_df).fit()

M2.summary()
#best estimators:

M = ols('admit_chance ~ CGPA+GRE+TOEFL+LOR+research',std_df1).fit()

M.summary()
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.025)

ls.fit(x_std,y)
ls.coef_
pd.DataFrame([x_std.columns,ls.coef_]).T
from sklearn.decomposition import PCA

pca = PCA(n_components=7)

pc = pca.fit_transform(x_std)

pc_df = pd.DataFrame(pc)

pc_df.head()
pc_df.shape
#Explained Variance Ratio

evr = pca.explained_variance_ratio_

print(evr)
#cumulative Variance Ratio

cvr=np.cumsum(evr)

print(cvr)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold
X = x_std1.drop(['university_rating','SOP'],axis=1)

y = df['admit_chance']

X.head()
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.30, shuffle=True,random_state = 25)

Xtrain.shape,Xtest.shape
model = LinearRegression()

model.fit(Xtrain,ytrain)
prediction = model.predict(Xtest)

mse = mean_squared_error(ytest,prediction)

error = np.sqrt(mse)

print('The RMSE value is :',error)
Xp = pc_df.iloc[:,:4]

Y = y

Xp.head(2)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xp,Y,test_size = 0.30, shuffle=True,random_state = 25)

Xtrain.shape,Xtest.shape
model = LinearRegression()

model.fit(Xtrain,ytrain)



prediction = model.predict(Xtest)

mse = mean_squared_error(ytest,prediction)

error = np.sqrt(mse)

print('The RMSE value is :',error)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5,shuffle=True,random_state=0)

training = []

testing = []

rmse = []



for train,test in kf.split(X,y):

    M = LinearRegression()

    Xtrain,Xtest = X.iloc[train,:],X.iloc[test,:]

    Ytrain,Ytest = y.iloc[train],y.iloc[test]

    M.fit(Xtrain,Ytrain)

    Y_PRED = M.predict(Xtest)

    

    train_score = M.score(Xtrain,Ytrain)

    test_score = M.score(Xtest,Ytest)

    training.append(np.round(train_score,3))

    testing.append(np.round(test_score,3))

    mse = mean_squared_error(Ytest,Y_PRED)

    error = np.sqrt(mse)

    rmse.append(error)

    

print('Training scores: ',training)

print('\nTesting scores: ', testing)

print('\nroot mean squared errors are : ',np.round(rmse,4))

print("\nthe Average RMSE is : " ,np.mean(rmse)) 

print("the Model variance is : " ,np.var(rmse)) 
lr = LinearRegression()

dtr = DecisionTreeRegressor(max_depth=3,random_state=25)

rfr = RandomForestRegressor(n_estimators=28,random_state=25)

abr = AdaBoostRegressor(lr,n_estimators=5,random_state=25)

gbr = GradientBoostingRegressor(n_estimators=29,random_state=25)

br = BaggingRegressor(lr,n_estimators=55,random_state=25)
#parameter={'n_estimators':np.arange(1,81)}

#gs= GridSearchCV(abr,parameter,cv=4,scoring='neg_mean_squared_error')

#gs.fit(Xtrain,ytrain)

#gs.best_params_
models = []



models.append(('linear regression',lr))

models.append(('DT regressor',dtr))

models.append(('RF regressor',rfr))

models.append(('ADA boost regressor',abr))

models.append(('Gradient boost',gbr))

models.append(('Bagging Regressor',br))
results = []

names =  []



for name,mod in models:

    kf=KFold(n_splits=5)

    cv_results = cross_val_score(mod,X,y,cv = kf,scoring='neg_mean_squared_error')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, np.mean(cv_results), cv_results.var())

    print(msg)  
# boxplot algorithm comparison

fig = plt.figure(figsize=[12,6])

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(1,1,1)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()