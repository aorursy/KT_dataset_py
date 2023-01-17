import os

import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels.api as sm

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gplay_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
gplay_df.info()
gplay_df.head()
gplay_df.shape
gplay_df.isna().sum()
duplicate_ser = gplay_df[gplay_df.duplicated()]

len(duplicate_ser)
gplay_df.drop_duplicates(inplace=True)
gplay_df.rename(columns={'Reviews':'ReviewCount','Size':'AppSize'},inplace=True)
def strip_cols(col_name):

    col_name=col_name.str.replace('$','')

    col_name=col_name.str.replace('+','')

    col_name=col_name.str.replace(',','')

    col_name=col_name.str.replace('M','e+6')

    col_name=col_name.str.replace('k','e+3')

    col_name=col_name.str.replace(' and up','')

    #col_name=col_name.str.strip('.GP','')

    #col_name=col_name.str.strip('W','')

    #col_name=col_name.str.strip('-prod','')

    

    return col_name

    

    

def change_dtype(col_name):

    col_name=col_name.astype('float')

    return col_name



def change_intdtype(col_name):

    col_name=col_name.astype('int64')

    return col_name



def replace_nan(col):

    col = col.replace('Varies with device',np.nan)

    return col

    

    
gplay_df.App.value_counts()
gplay_df.App.nunique()
gplay_df['Rating'].value_counts()
# taking the Rating as 1.9 instead of 19

gplay_df['Rating'].replace('19.0','1.9',inplace=True)
gplay_df.Price.value_counts().sort_index()
gplay_df.drop(gplay_df[gplay_df['Price']=='Everyone'].index,inplace=True)


gplay_df['Price'] = strip_cols(gplay_df['Price'])

gplay_df['Price'] = change_dtype(gplay_df['Price'])
gplay_df.Price.value_counts().sort_index()
gplay_df.AppSize.sample(20)
gplay_df.AppSize.value_counts()
gplay_df['AppSize'] = replace_nan(gplay_df['AppSize'])
gplay_df['AppSize'] = strip_cols(gplay_df['AppSize'])

gplay_df['AppSize'] = change_dtype(gplay_df['AppSize'])

gplay_df['AppSize'] = gplay_df['AppSize'] /1000000 # Appsize in MB
gplay_df['AppSize'].value_counts()
gplay_df['Installs'].value_counts()
gplay_df['Installs'] = strip_cols(gplay_df['Installs'])

gplay_df['Installs'] = change_intdtype(gplay_df['Installs'])
gplay_df['Installs'].value_counts().sort_index()
gplay_df.ReviewCount.value_counts()
gplay_df['ReviewCount'] = strip_cols(gplay_df['ReviewCount'])

gplay_df['ReviewCount'] = change_intdtype(gplay_df['ReviewCount'])
gplay_df['ReviewCount']=gplay_df['ReviewCount']/1000000 #Count in  Million
gplay_df.ReviewCount.value_counts().sort_index()
gplay_df['Genres'].value_counts().sort_values()
prim = gplay_df.Genres.apply(lambda x:x.split(';')[0])

gplay_df['Prim_Genre']=prim

gplay_df['Prim_Genre'].tail()
sec = gplay_df.Genres.apply(lambda x:x.split(';')[-1])

gplay_df['Sec_Genre']=sec

gplay_df['Sec_Genre'].tail()
group_gen=gplay_df.groupby(['Category','Prim_Genre','Sec_Genre'])

group_gen.size().head(20)
gplay_df.drop(['Genres','Prim_Genre'],axis=1,inplace=True)
gplay_df['Last Updated'].value_counts().sort_values()
gplay_df['Last Updated'] = pd.to_datetime(gplay_df['Last Updated'])
gplay_df['Last Updated'].value_counts().sort_index()
#### data is from year 2010,May to 2018,Aug

from datetime import datetime,date

gplay_df['Last_Updated_Days']=gplay_df['Last Updated'].apply(lambda x: date.today()-datetime.date(x))

gplay_df['Last_Updated_Days'].head()
gplay_df['Last_Updated_Days'] = gplay_df['Last_Updated_Days'].dt.days
gplay_df.drop(['Current Ver'],axis=1,inplace=True)
gplay_df['Android Ver'].value_counts().sort_values()
gplay_df['Android Ver'] = strip_cols(gplay_df['Android Ver'])

gplay_df['Android Ver'] = replace_nan(gplay_df['Android Ver'])

gplay_df['Android Ver'].replace('4.4W','4.4',inplace=True)
gplay_df['Android Ver'].value_counts().sort_values()
gplay_df['Category'].value_counts().sort_values()
gplay_df['Type'].value_counts() 
gplay_df['Content Rating'].value_counts() 
gplay_df.info()
# categorical and Numerical Values:

num_var = gplay_df.select_dtypes(include=['int64','float64']).columns

cat_var = gplay_df.select_dtypes(include=['object','datetime64','timedelta64']).columns

num_var,cat_var
gplay_df.isna().sum()
missing_perc = (gplay_df.isna().sum()*100)/len(gplay_df)

missing_df = pd.DataFrame({'columns':gplay_df.columns,'missing_percent':missing_perc})

missing_df
col_cat = ['Type','Android Ver'] #Categorical Var.

for col in col_cat:

    gplay_df[col].fillna(gplay_df[col].mode()[0],inplace=True)

    

col_num=['Rating','AppSize'] #Numerical Var.

for col in col_num:

    gplay_df[col].fillna(gplay_df[col].median(),inplace=True)
gplay_df.isna().sum()
gplay_df.info()
gplay_df.head()


gplay_df.to_csv('Clean_GplayApps.csv',index=False)
# After Cleaning

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

gplay_df = pd.read_csv('Clean_GplayApps.csv')
gplay_df.head()
cat_var = gplay_df.select_dtypes(include=['object'])

col_cat = cat_var.columns

cat_var
sns.boxplot(gplay_df['Rating']);


sns.boxplot(gplay_df['ReviewCount']);
sns.boxplot(gplay_df['Installs']);
sns.boxplot(gplay_df.Price);
sns.boxplot(gplay_df['AppSize']);
num_var = gplay_df.select_dtypes(include=['int64','float64'])

col_num = num_var.columns

num_var
num_var.hist(figsize=(9,9),bins=50);
sns.countplot(data=gplay_df,x='Type');
gplay_df.Category.value_counts().plot(kind='bar');
gplay_df['Content Rating'].value_counts().plot(kind='bar');
gplay_df['Sec_Genre'].value_counts().plot(kind='bar');
sns.lineplot(x='AppSize',y='Installs',data=gplay_df);
sns.lineplot(x='Price',y='Installs',data=gplay_df);

plt.xlabel('Price (Dollars)');
sns.lineplot(y='ReviewCount',x='Installs',data=gplay_df);
sns.lineplot(x='Rating',y='Price',data=gplay_df);
sns.lineplot(x='Rating',y='AppSize',data=gplay_df);
sns.lineplot(x='Rating',y='ReviewCount',data=gplay_df);
sns.lineplot(x='AppSize',y='ReviewCount',data=gplay_df);
sns.barplot(x='Type',y='Installs',data=gplay_df);
sns.barplot(x='Type',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(8,5));

sns.barplot(x='Content Rating',y='Price',data=gplay_df);
plt.figure(figsize=(8,5));

sns.barplot(x='Content Rating',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(8,5));

sns.barplot(x='Content Rating',y='Installs',data=gplay_df);
plt.figure(figsize=(30,5));

sns.barplot(x='Android Ver',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(30,5));

sns.barplot(x='Android Ver',y='Price',data=gplay_df);
plt.figure(figsize=(30,5));

sns.barplot(x='Android Ver',y='Installs',data=gplay_df);
gplay_df.columns
col=['App','Android Ver', 'Category','Sec_Genre','Content Rating', 'Type','ReviewCount', 'AppSize', 'Installs',

       'Price', 'Last_Updated_Days','Rating']

gplay_df =gplay_df [col]
gplay_df.head()
gplay_df.shape
enc_var = gplay_df.select_dtypes(include=['object']).columns

enc_var
enc_var = ['Category', 'Sec_Genre', 'Type', 'Content Rating'] 


lbl_enc = LabelEncoder()

for feat in enc_var:

    gplay_df[feat] = lbl_enc.fit_transform(gplay_df[feat].astype(str))
gplay_df.sample(10)
#df_copy = gplay_df.copy()

#df_copy = pd.get_dummies(df_copy,columns=enc_var,drop_first = True)

X=gplay_df.iloc[:,2:10].values

y=gplay_df.iloc[:,-1].values

X


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)





reg = LinearRegression()

reg.fit(X_train,y_train)

reg.score(X_test,y_test)
y_pred=reg.predict(X_test)

y_pred


print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


X_train.shape
X =np.append(np.ones([X.shape[0],1]).astype(int),values=X,axis=1)



X_opt = X[:, [0,1,2,3,4,5,6,7,8]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0,2,3,4,5,6,7,8]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:, [0,2,4,5,6,7,8]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[0,2,4,5,6,8]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_train,X_test,y_train,y_test = train_test_split(X[:, [0,2,4,5,6,8]],y,test_size=0.2,random_state=0)

model=LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)
y_pred = model.predict(X_test)

y_pred
y_test
model.intercept_
model.coef_


print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred))) 

#It is recommended that RMSE be used as the primary metric to interpret your model.