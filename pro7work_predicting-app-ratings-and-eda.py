# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

gps_df=pd.read_csv('../input/googleplaystore.csv')
gps_df.shape
gps_df.isnull().any()
#Deleting Null rows from Rating, Type, Content Rating (Leaving Current Ver and Android ver as right now 

#we are not using these columns for any prediction)



gps_df.dropna(subset=['Rating','Type','Content Rating'], inplace=True)

gps_df.shape
gps_df.describe()

gps_df.sample(7)
gps_df['Price'].unique()
gps_df['Size'].unique()
#REplace $ by a '' and 'M' also by a ''

gps_df['Price'].replace(to_replace=r'[\$]',value='',regex=True, inplace=True)





#gps_df.sort_values('Price', ascending=False)
gps_df['Size'].replace(to_replace='Varies with device',value=np.nan,inplace=True)
#gps_df['Size'].replace(to_replace=r'[\\Mk]',value='',regex=True).apply(lambda x: float(x)*1000000),regex=True)

gps_df_extracted=gps_df['Size'].str.extract(r'(\d?.\d?)([\\Mk])')
gps_df_extracted=gps_df_extracted.replace(to_replace=['k','M'],value=[1000,1000000]).astype(float)
gps_df_extracted
gps_df['Size']=gps_df_extracted[0]*gps_df_extracted[1]
gps_df['Size'].unique
gps_df[gps_df['Size'].isnull()].shape
gps_df.groupby('Category').Size.mean()
gps_df['Size'].fillna(gps_df.groupby('Category').Size.transform('mean'), inplace=True)
#Converting Price to numeric

gps_df['Price']=pd.to_numeric(gps_df.Price)
# Lets find most reviewed apps

#Converting the Reviews column to numeric

gps_df['Reviews']=pd.to_numeric(gps_df.Reviews)
#REplace , and + by a ''

gps_df['Installs'].replace(to_replace=r'[\,]|[\+]',value='',regex=True,inplace=True)
#Converting Installs to numeric

gps_df['Installs']=pd.to_numeric(gps_df.Installs)
gps_df.dtypes
gps_df.sort_values('Reviews', ascending=False).head(8)
gps_df.sort_values('Reviews', ascending=False, inplace=True)
gps_df[gps_df.duplicated(['App','Category'])].head(10)
gps_df.drop_duplicates(['App','Category'], inplace=True)
plt.figure(figsize=(12,8))

ax=sns.barplot(y='App',x='Reviews',data=gps_df.head(10))

plt.show()
gps_df_categ=gps_df.groupby('Category')
gps_df_categ.count().shape
gps_df_categcount=gps_df_categ.count()
gps_df_categcount.sort_values('App', ascending=False, inplace=True)


plt.figure(figsize=(15,10))

ax=sns.barplot(y=gps_df_categcount.index, x='App',data=gps_df_categcount, palette='Blues_d')

ax.set(xlabel='Number of Apps')

plt.show()
gps_df_type=gps_df.groupby(['Type','Category']).count()
gps_df_type.index.levels

gps_df_type=gps_df_type.unstack(level=0)
#Some of the categories have no paid apps so filling NaN values with 0

gps_df_type=gps_df_type['App'].fillna(value=0,axis=1)
#Plotting the Type of Paid and free Apps

gps_df_type.plot(kind='bar',figsize=(15,10))

sns.scatterplot(x='Rating',y='Content Rating',data=gps_df)
#removing the unrated data

gps_df=gps_df[gps_df['Content Rating']!='Unrated']


from pandas.plotting import scatter_matrix



sns.distplot(gps_df['Rating'], bins=50)

#gps_df.hist()

sns.pairplot(gps_df)

plt.show()
gps_df.corr()
new_gps_df= gps_df[['Category','Reviews','Size','Installs','Price','Content Rating','Rating']]
#applying log to data



log_gps_df=np.log1p(new_gps_df[['Reviews','Size','Installs','Price','Rating']])
log_gps_df['Category']=new_gps_df[['Category']]
log_gps_df['Content Rating']=new_gps_df[['Content Rating']]
#plotting the tranformed data

sns.pairplot(log_gps_df)
log_gps_df.corr()
from tempfile import mkdtemp

from shutil import rmtree

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.linear_model import LinearRegression,Ridge

from sklearn.svm import SVR

from sklearn.linear_model import LassoCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.metrics import r2_score

import statistics





    

numeric_features = ['Reviews','Installs','Size','Price']

numeric_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())])



categorical_features = ['Content Rating','Category']

categorical_transformer = Pipeline(steps=[

        ('onehot', OneHotEncoder(handle_unknown='ignore'))])



column_trans = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



column_trans_cat = ColumnTransformer(

    transformers=[

        

        ('cat', categorical_transformer, categorical_features)],

        remainder='passthrough')





    

clf = Pipeline(steps=[('preprocessor', column_trans),

                          ('classifier',Ridge(alpha=0.1, random_state=2)

                            )])





X = log_gps_df.drop(['Rating'], axis=1)

y = log_gps_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf.fit(X_train, y_train)

y_test_pred=clf.predict(X_test)

Test_mse= mean_squared_error(y_test,y_test_pred)



cat_features=list(clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names())

coefficients=list(clf.named_steps['classifier'].coef_)

   

num_features=[]

num_features=['Reviews','Installs','Size','Price']

coefficients=list(clf.named_steps['classifier'].coef_)

cat_features.extend(num_features)



features_importance_1=[(cat_features, np.round(coefficients, 3)) for cat_features, coefficients in zip(cat_features, coefficients)]

coef_sum=0

    

print("Intercepts::",clf.named_steps['classifier'].intercept_)



print("Test MAE::", mean_absolute_error(y_test,y_test_pred))

print("Test MSE::",(Test_mse))



print("Test R^2 score::",(r2_score(y_test,y_test_pred)))

   

for pair in sorted(features_importance_1, key=lambda x: x[1]):

    coef_sum=coef_sum+pair[1]

    print(pair)

  

print("coef_sum::",coef_sum)
clf_new=Pipeline(steps=[('scaler', StandardScaler(with_mean=False)),

                           ('classifier',Ridge(alpha=0.1, random_state=2))

                            ])



X = log_gps_df.drop(['Rating'], axis=1)

y = log_gps_df['Rating']

                            

X_transformed=column_trans_cat.fit_transform(X)

print("X_transformed::",X_transformed.shape)



X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)



   

clf_new.fit(X_train, y_train)

y_test_pred=clf_new.predict(X_test)

Test_mse= mean_squared_error(y_test,y_test_pred)

    

    

cat_features=list(column_trans_cat.named_transformers_['cat'].named_steps['onehot'].get_feature_names())

    #Category_features=list(clf.named_steps['preprocessor'].named_transformers_['cat'])

num_features=[]

num_features=['Reviews','Installs','Size','Price']



coefficients=list(clf_new.named_steps['classifier'].coef_)

cat_features.extend(num_features)



features_importance_2=[(cat_features, np.round(coefficients, 3)) for cat_features, coefficients in zip(cat_features, coefficients)]

coef_sum=0

    



    

print("Intercepts::",clf_new.named_steps['classifier'].intercept_)





print("Test MAE::", mean_absolute_error(y_test,y_test_pred))

print("Test MSE::",(Test_mse))



print("Test R^2 score::",r2_score(y_test,y_test_pred))

   

for pair in sorted(features_importance_2, key=lambda x: x[1]):

    coef_sum=coef_sum+pair[1]

    print(pair)

 

print("coef_sum::",coef_sum)
def plotting_feature_importances(features_importance,title):

    plt.figure(figsize=(30,15))



    for pair in sorted(features_importance, key=lambda x: x[1]):

            plt.barh(pair[0],pair[1])



    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.xlabel("Features", fontsize=18)

    plt.ylabel("Importance",fontsize=18)

    plt.title(title,  fontsize=20)

    plt.show()


plotting_feature_importances(features_importance_1, "Features Importances/weights when OnehotEncoding+Scaling+Learning in Pipeline")



plotting_feature_importances(features_importance_2, "Features Importances/weights when first OnehotEncoding and then Scaling+Learning in Pipeline")