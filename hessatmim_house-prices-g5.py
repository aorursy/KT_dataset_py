import numpy as np

import pandas as pd

import patsy



from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import cross_val_score



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)

plt.style.use('fivethirtyeight')



%config InlineBackend.figure_format = 'retina'

%matplotlib inline
import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)
df.head()
#df.set_index('Id')
test.rename(str.lower, axis='columns',inplace=True)
df.rename(str.lower, axis='columns',inplace=True)
null_value = df.isnull().sum()[df.isnull().sum() !=0] * 100/ df.shape[0]

null_value.sort_values(ascending = True).plot(kind='barh' ,title=" % of Null values " ,color = '#004d61' ,figsize=(15,5))
df['alley'].fillna('NO',inplace=True)
df.loc[df.index == 948 , 'bsmtexposure']  = 'No'
df.loc[df.index == 332 , 'bsmtfintype2'] = 'Unf'
df.masvnrtype.fillna( value='None' , inplace=True)

df.masvnrarea.fillna( value= 0 , inplace= True)
df['bsmtqual'].fillna('No Basement' , inplace=True)

df['bsmtfintype2'].fillna('No Basement' , inplace=True)

df['bsmtfintype1'].fillna('No Basement' , inplace=True)

df['bsmtexposure'].fillna('No Basement' , inplace=True)

df['bsmtcond'].fillna('No Basement' , inplace=True)
df['garagefinish'].fillna("Not finish", inplace = True)
df['garagequal'].fillna("No", inplace = True)
df['garagecond'].fillna("NO", inplace = True)
df.drop(['poolqc','miscfeature','fence'], inplace=True, axis=1)
df.drop(columns=['bsmtfinsf1' , 'bsmtfinsf2' ,'bsmtunfsf'] , axis=1  , inplace=True)
df.dropna(subset=['electrical'],inplace=True) # i dropped this to make train and test equal in rows
df['lotfrontage'].fillna(test.lotfrontage.mean(),inplace=True)

df['masvnrarea'].fillna(test.masvnrarea.mean(),inplace=True)

df['bsmtfullbath'].fillna(test.bsmtfullbath.mean(),inplace=True)

df['bsmthalfbath'].fillna(test.bsmthalfbath.mean(),inplace=True)

df['garageyrblt'].fillna(test.garageyrblt.mean(),inplace=True)
df['fireplacequ'].fillna('NO',inplace=True)

df['garagetype'].fillna('NO',inplace=True)
df.isnull().sum()
test['alley'].fillna('No',inplace=True)

test['bsmtqual'].fillna('NO',inplace=True)

test['bsmtcond'].fillna('NO',inplace=True)

test['bsmtexposure'].fillna('NO',inplace=True)

test['bsmtfintype1'].fillna('NO',inplace=True)

test['bsmtfintype2'].fillna('NO',inplace=True)

test['fireplacequ'].fillna('NO',inplace=True)

test['garagetype'].fillna('NO',inplace=True)

test['garagefinish'].fillna('NO',inplace=True)

test['garagequal'].fillna('NO',inplace=True)

test['garagecond'].fillna('NO',inplace=True)

test['garagecars'].fillna(test.garagecars.mean(), inplace=True)

test['garagearea'].fillna(test.garagearea.mean(), inplace=True)

test['totalbsmtsf'].fillna(test.totalbsmtsf.mean(),inplace=True)
test.drop(['poolqc','miscfeature','fence'], inplace=True, axis=1)
test.drop(columns=['bsmtfinsf1' , 'bsmtfinsf2' ,'bsmtunfsf'] , axis=1  , inplace=True)
test['lotfrontage'].fillna(test.lotfrontage.mean(),inplace=True)

test['masvnrarea'].fillna(test.masvnrarea.mean(),inplace=True)

test['bsmtfullbath'].fillna(test.bsmtfullbath.mean(),inplace=True)

test['bsmthalfbath'].fillna(test.bsmthalfbath.mean(),inplace=True)

test['garageyrblt'].fillna(test.garageyrblt.mean(),inplace=True)
test['mszoning'].fillna('NO',inplace=True)

test['utilities'].fillna('NO',inplace=True)

test['utilities'].fillna('NO',inplace=True)

test['exterior1st'].fillna('NO',inplace=True)

test['exterior2nd'].fillna('NO',inplace=True)

test['masvnrtype'].fillna('NO',inplace=True)
test.saletype.dtypes
test['kitchenqual'].fillna('NO',inplace=True)

test['totrmsabvgrd'].fillna(0,inplace=True)

test['functional'].fillna('No',inplace=True)
test['saletype'].fillna('No',inplace=True)
test.isnull().sum()
#test.isnull().sum()
df.shape , test.shape
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x="mssubclass", y="saleprice",

                     hue="mssubclass", size="mssubclass",

                     data=df,ax=ax[0][0])

sns.scatterplot(x="lotfrontage", y="saleprice",

                     hue="mssubclass", 

                     sizes=(20, 200), palette='BuPu_r',

                     legend="full", data=df,ax=ax[0][1])

sns.scatterplot(x="lotarea", y="saleprice",

                      hue="mssubclass", 

                     palette="Set2",

                     data=df,ax=ax[1][0])

sns.scatterplot(x="overallqual", y="saleprice",

                     hue='overallqual',palette='YlOrRd' ,data=df,ax=ax[1][1])

fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x="overallcond", y="saleprice",

                     hue='overallqual',palette='PuOr_r' ,data=df,ax=ax[0][0])

sns.scatterplot(x="yearbuilt", y="saleprice",

                      hue="overallcond",

                     palette="Greens",

                     data=df,ax=ax[0][1])

sns.scatterplot(x="garagearea", y="saleprice",

                      hue="garagequal", 

                     palette="Set2",

                     data=df,ax=ax[1][0])

sns.scatterplot(x="garagecars", y="saleprice",

                     hue='garagequal',palette='YlOrRd' ,data=df,ax=ax[1][1])
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x="grlivarea", y="saleprice",

                     hue='1stflrsf',palette='PuOr_r' ,data=df,ax=ax[0][0])

sns.scatterplot(x="1stflrsf", y="saleprice",

                     palette="Greens",hue='grlivarea',

                     data=df,ax=ax[0][1])

sns.scatterplot(x="fullbath", y="saleprice",hue='grlivarea', palette="Greens",

                     data=df,ax=ax[1][0])

sns.scatterplot(x="totrmsabvgrd", y="saleprice",palette='YlOrRd',hue='grlivarea' ,data=df,ax=ax[1][1])
df.groupby('bsmtcond')[['saleprice']].sum()
pd.crosstab(df.bsmtexposure, 'sum' , values=df.saleprice , aggfunc=sum ,normalize=True ).plot(kind = 'barh' ,color = '#004d61', figsize=(15,5));
pd.crosstab(df.bsmtexposure, 'sum' , values=df.saleprice , aggfunc=sum ,normalize=True ).plot(kind = 'barh' ,color = '#004d61', figsize=(15,5));
pd.crosstab(df.bsmtfintype1, 'sum' , values=df.saleprice , aggfunc=sum ,normalize=True )
pd.crosstab(df.yearremodadd, 'sum' , values=df.saleprice , aggfunc=sum ).plot(color = '#004d61' , figsize=(15,5));
sns.scatterplot(x = df.yearremodadd , y = df.saleprice , hue=df.yearremodadd);
df._get_numeric_data()
df.corrwith(df.totalbsmtsf)
df.shape
sns.scatterplot(x = df.totalbsmtsf , y = df.saleprice);
sns.scatterplot(x = df.totalbsmtsf , y = df.masvnrarea);
df.totalbsmtsf.hist()
df.saleprice.hist();
df.yearremodadd.hist();
fig, ax1 = plt.subplots(figsize=(10,8))

sns.boxplot(x=df['saleprice']);
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="salecondition", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="saletype", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(10,10))

sns.barplot(x="paveddrive", y="saleprice", data=df, palette="colorblind");
plt.scatter(x = df['paveddrive'] , y= df['saleprice'],s=100, c='g')
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="garagecond", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="garagefinish", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.boxplot(x="salecondition", y="saleprice", data=df, palette="colorblind");
df.select_dtypes('object')
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.barplot(x="mszoning", y="saleprice",palette='PuOr_r',data=df,ax=ax[0][0])

sns.barplot(x="street", y="saleprice", data=df,palette='plasma',ax=ax[0][1])

sns.barplot(x="alley", y="saleprice", data=df,palette='BuPu_r',ax=ax[1][0])

sns.barplot(x="lotshape", y="saleprice", data=df,palette='Oranges',ax=ax[1][1])
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.barplot(x="landcontour", y="saleprice", data=df,palette='Greens',ax=ax[0][0])

sns.barplot(x="utilities", y="saleprice", data=df,palette='BuPu',ax=ax[0][1])

sns.barplot(x="neighborhood", y="saleprice", data=df,palette='YlOrRd',ax=ax[1][0])

sns.barplot(x="bldgtype", y="saleprice", data=df,palette='PuRd',ax=ax[1][1])
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.barplot(x="lotconfig", y="saleprice", data=df,palette='PuBuGn',ax=ax[0][0])

sns.barplot(x="landslope", y="saleprice", data=df,palette='OrRd',ax=ax[0][1])

sns.barplot(x="condition1", y="saleprice", data=df,palette='OrRd',ax=ax[1][0])

sns.barplot(x="condition2", y="saleprice", data=df,palette='Blues',ax=ax[1][1])
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.barplot(x="housestyle", y="saleprice", data=df,palette='BuGn',ax=ax[0][0])

sns.barplot(x="electrical", y="saleprice", data=df,palette='OrRd',ax=ax[0][1])

sns.barplot(x="kitchenqual", y="saleprice", data=df,palette='OrRd',ax=ax[1][0])

sns.barplot(x="functional", y="saleprice", data=df,palette='Blues',ax=ax[1][1])
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(16, 10))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.barplot(x="fireplacequ", y="saleprice", data=df,palette='BuGn',ax=ax[0][0])

sns.barplot(x="garagetype", y="saleprice", data=df,palette='OrRd',ax=ax[0][1])

sns.barplot(x="garagefinish", y="saleprice", data=df,palette='OrRd',ax=ax[1][0])

pd.crosstab(df.roofstyle , columns='sum' , values=df.saleprice , aggfunc=sum).sort_values(by= 'sum' , ascending=True).plot(kind = 'bar' ,color = '#004d61' ,figsize=(15,5));
pd.crosstab([df.roofmatl  ,df.roofstyle ], columns='sum' , values=df.saleprice , aggfunc=sum).sort_values(by = 'sum' ,ascending =True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,5));
pd.crosstab(df.roofmatl   , columns='sum' , values=df.saleprice , aggfunc=sum).sort_values(by = 'sum' ,ascending =True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,5));
pd.crosstab(df.foundation   , columns='sum' , values=df.saleprice , aggfunc=sum).sort_values(by = 'sum' ,ascending =True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,5));
pd.crosstab([df.foundation ,df.roofstyle , df.roofmatl ]     , columns='sum' , values=df.saleprice , aggfunc=sum).sort_values(by = 'sum' ,ascending =True).plot(kind = 'barh' ,color = '#004d61' , figsize=(20,25));
pd.crosstab([df.exterior1st ] ,columns='sum' , values=df.saleprice , aggfunc=sum).plot(kind = 'barh' , color = '#004d61');
ex_f_ex_s  = pd.crosstab([df.exterior1st , df.exterior2nd ] ,columns='sum' , values=df.saleprice , aggfunc=sum)

ex_f_ex_s.sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' , figsize=(20,25));
sns.barplot( x= 'masvnrtype'  , y = 'saleprice' ,hue = 'masvnrtype' , data = df);
sns.violinplot(x="masvnrtype" ,y="masvnrarea", hue='masvnrtype',data=df, dodge=True)
sns.catplot(x="masvnrtype" ,y="masvnrarea", hue='masvnrtype' , data=df, kind='swarm', aspect=2);
sns.swarmplot(x ='masvnrtype', y =df.saleprice, data = df) ;
sns.boxplot(x ='masvnrtype', y =df.saleprice, data = df);
ex_quality = pd.crosstab([df.exterior1st , df.exterqual ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_quality.sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,15));
ex_quality = pd.crosstab([df.exterior1st , df.exterqual ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_quality.sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,15));
ex_quality_sec = pd.crosstab([df.exterior2nd , df.exterqual]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_quality_sec.sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' , figsize=(15,15));
ex_quality_cond = pd.crosstab([df.exterior1st,df.exterior2nd , df.exterqual , df.extercond ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_quality_cond.nlargest(3 , columns = 'sum').sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' );
ex_cond = pd.crosstab([df.exterior1st, df.extercond ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_cond.nlargest(3 , columns = 'sum').sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' );
ex_cond2 = pd.crosstab([df.exterior2nd, df.extercond ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

ex_cond2.nlargest(3 , columns = 'sum').sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' );
sns.barplot( x= df.heating  , y = 'saleprice' ,hue = df.heatingqc , data = df);
heat = pd.crosstab([df.heating, df.heatingqc ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

heat.nlargest(3 , columns = 'sum').sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' );
heat = pd.crosstab([df.heating, df.heatingqc ]   , columns='sum' , values=df.saleprice , aggfunc=sum)

heat.nlargest(10 , columns = 'sum').sort_values( by = 'sum' , ascending = True).plot(kind = 'barh' ,color = '#004d61' );
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="garagecond", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="salecondition", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="saletype", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="paveddrive", y="saleprice", data=df, palette="colorblind");
plt.scatter(x = df['paveddrive'] , y= df['saleprice'],s=100, c='g')
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="garagecond", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.barplot(x="garagefinish", y="saleprice", data=df, palette="colorblind");
fig, ax1 = plt.subplots(figsize=(15,10))

sns.boxplot(x="salecondition", y="saleprice", data=df, palette="colorblind");
train_dum = pd.get_dummies(df,drop_first=True)
test_dum = pd.get_dummies(test ,drop_first=True)
test_dum.shape , train_dum.shape
# this for loop to only get off of the columns that two df are not equal at, cuz after we convert them to a dummy variables

# the coulmns were have diffrent unique values to we get ride of the extra columns in the train (in our case df)

df_new = []

for col in train_dum.columns:

    for col2 in test_dum.columns:

        if col == col2 :

            df_new.append(col)

            

df_new
df.shape , test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
feature_col = ['overallqual','totalbsmtsf','1stflrsf','grlivarea',

               'garagecars' ,'garagearea']
XX = df[feature_col]

y= df.saleprice
ss = StandardScaler()

X_std = ss.fit_transform(XX)
my_model = RandomForestRegressor()

my_model.fit(XX,y)

my_model.score(XX,y)
X_train , x_test , y_train , y_test =train_test_split(X_std,y,test_size =.3 , shuffle=True) # by StandardScaler
my_model.score(X_train,y_train)
my_model.score(x_test,y_test)
test_X = test[feature_col]
#test_X.isnull().sum()


# Use the model to make predictions

predicted_prices = my_model.predict(test_X)
predicted_prices
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
X_lasso = train_dum[df_new]

y_lasso = train_dum.saleprice
#X_lasso.isnull().sum()
sss = StandardScaler()

X_train , X_test , y_train , y_test = train_test_split(X_lasso , y_lasso  , test_size = .3 , shuffle = True)

sss.fit(X_train)

X_train_std = sss.transform(X_train)

X_test_std = sss.transform(X_test)
ls = Lasso()

ls.fit(X_train_std, y_train)

print(ls.score(X_train_std, y_train))

ls.score(X_test_std, y_test)
test_X_lasso = test_dum[df_new]
predicted_prices_lasso = ls.predict(test_X_lasso)
predicted_prices_lasso
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices_lasso})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_lasso_2.csv', index=False)
X_Forest = train_dum[df_new]

y_Forest = train_dum.saleprice
sss_4 = StandardScaler()

XX_train , XX_test , yy_train , yy_test = train_test_split(X_Forest , y_Forest , test_size = .3 , shuffle = True)

sss_4.fit(XX_train)

XX_train_std = sss_4.transform(XX_train)

XX_test_std = sss_4.transform(XX_test)
rfr = RandomForestRegressor(n_estimators=100, max_depth=10)

rfr.fit(XX_train_std , yy_train)
rfr.score(XX_train_std, yy_train)
rfr.score(XX_test_std, yy_test)
test_X_Forest = test_dum[df_new]
predicted_prices_Forest = rfr.predict(test_X_Forest)
predicted_prices_Forest 
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices_Forest})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_forest.csv', index=False)
#X_Grid = train_dum[df_new]

y_Grid = train_dum.saleprice
X_Grid = df[feature_col]
y_Grid.shape , X_Grid.shape
sss_5 = StandardScaler()

XXX_train , XXX_test , yyy_train , yyy_test = train_test_split(X_Grid  , y_Grid  , test_size = .3 , shuffle = True)

sss_5.fit(XXX_train)

XXX_train_std = sss_5.transform(XXX_train)

XXX_test_std = sss_5.transform(XXX_test)
from sklearn.model_selection import GridSearchCV         

grd = GridSearchCV(ls, {'alpha':[0.001,0.002,0.02,0.05,0.2,0.7]}, n_jobs=-1, verbose=1)

grd.fit(XXX_train_std, yyy_train)
grd.score(XXX_train_std, yyy_train)
grd.score(XXX_test_std, yyy_test)
test_X_Grid = df[feature_col]
#test_X_Grid = test_dum[df_new]
predicted_prices_Grid = grd.predict(test_X_Grid)
predicted_prices_Grid
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices_Grid})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_grid.csv', index=False)
y = train_dum.saleprice

X = df[feature_col]
s_lasso_cv = StandardScaler()

X_train , X_test , y_train , y_test = train_test_split(X  , y , test_size = .3 , shuffle = True)

s_lasso_cv.fit(X_train)

s_lasso_cv_std = s_lasso_cv.transform(X_train)

s_lasso_cv_std = s_lasso_cv.transform(X_test)
from sklearn.linear_model import LassoCV

lscv = LassoCV(n_alphas=100)

print(lscv.fit(X_train, y_train))

lscv.score(X_train, y_train)
lscv.score(X_test, y_test)
test_X_lassocv = test_dum[feature_col]
predicted_prices_lassocv = lscv.predict(test_X_lassocv)
predicted_prices_lassocv
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices_lassocv})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_lasso_cv.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
ss_6 = StandardScaler()

Xs_train = ss_6.fit_transform(X_train)

xs_test = ss_6.fit_transform(X_test)
knn = KNeighborsClassifier()

knn.fit(Xs_train,y_train)

knn.score(Xs_train,y_train)
knn.score(xs_test,y_test)
test_X_knn = test_dum[feature_col]
predicted_prices_knn = lscv.predict(test_X_knn)
predicted_prices_knn
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices_knn})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_knn.csv', index=False)
def correlation_heat_map(df):

    corrs = df.corr()



    # Set the default matplotlib figure size:

    fig, ax = plt.subplots(figsize=(40,40))



    # Generate a mask for the upper triangle (taken from the Seaborn example gallery):

    mask = np.zeros_like(corrs, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Plot the heatmap with Seaborn.

    # Assign the matplotlib axis the function returns. This allow us to resize the labels.

    ax = sns.heatmap(corrs, mask=mask, annot=True, vmin=-1, vmax=1,linewidths=.5,cmap="YlGnBu")



    # Resize the labels.

    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=30)

    

    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=15, rotation=0)



    # If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.

    plt.show()
#correlation_heat_map(df.corr())