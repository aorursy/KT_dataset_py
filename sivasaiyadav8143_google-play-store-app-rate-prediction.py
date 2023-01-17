

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



#supress warnings

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_full = pd.read_csv('/kaggle/input/google-playstore-apps/Google-Playstore-Full.csv')

df_full.head()
df_full.columns
print(df_full.shape)

df_full.isna().sum()
# column Unnamed: 11 



un11 = df_full['Unnamed: 11'][df_full['Unnamed: 11'].notna()]

print(f'No.of records in Unnamed: 11 ->  {len(un11)} \n')

print(f'% of data present -> {(len(un11) / len(df_full) * 100)}\n')

print(un11.head(3))

print('\n')

df_full[df_full['Unnamed: 11'].notna()].head(3)
# column Unnamed: 12



un12 = df_full['Unnamed: 12'][df_full['Unnamed: 12'].notna()]

print(f'No.of records in Unnamed: 12 ->  {len(un12)} \n')

print(f'% of data present -> {(len(un12) / len(df_full) * 100)}\n')

print(un12)
# column Unnamed: 13



un13 = df_full['Unnamed: 13'][df_full['Unnamed: 13'].notna()]

print(f'No.of records in Unnamed: 13 ->  {len(un13)} \n')

print(f'% of data present -> {(len(un13) / len(df_full) * 100)}\n')

print(un13)
# column Unnamed: 14



un14 = df_full['Unnamed: 14'][df_full['Unnamed: 14'].notna()]

print(f'No.of records in Unnamed: 14 ->  {len(un14)} \n')

print(f'% of data present -> {(len(un14) / len(df_full) * 100)}\n')

print(un14)
un_11 = df_full[df_full['Unnamed: 11'].notna()].index

un_12 = df_full[df_full['Unnamed: 12'].notna()].index

un_13 = df_full[df_full['Unnamed: 13'].notna()].index

un_14 = df_full[df_full['Unnamed: 14'].notna()].index



#pass them into set to remove duplicates

un_index = set(list(un_11) + list(un_12) + list(un_13) + list(un_14))
# delete undefined columns and NA's

print(f'Before Delete : {df_full.shape}')

df_full.drop(un_index,inplace=True)

df_full.drop(columns=['Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'],axis=1,inplace=True)

print(f'After Delete : {df_full.shape}')
df_full.head()
df_full.info()
df_full.describe()
df_full.isna().sum()
# df_full.columns = df_full.columns.str.replace(' ', '_')
df_full[df_full['App Name'].isna()]
df_full.dropna(axis=0,subset=['App Name'],inplace=True)
df_full['App Name'].value_counts().head(25)
df_full[df_full['App Name'].isin(['????'])].head()
wrong_df = df_full[df_full['App Name'].str.contains('?',regex = False)]

wrong_df
print(f'Total {len(wrong_df)} records i.e. ({round((len(wrong_df)/len(df_full))*100,2)}%) contains "?" in them')

# wrong_df[wrong_df.iloc[:,1:].duplicated(keep=False )]
# dropping App Name columns

# df_full.drop(columns=['App Name'],inplace=True)
# ques_index = df_full[df_full['App Name'].str.contains('?',regex = False)].index

# df_full.drop(index=ques_index,inplace=True)
# df_full.head(15)
# checking for duplicates

df_full[df_full.duplicated(subset='App Name')].sort_values(by=['App Name'])
# Lets check with some apps instead of all at once

df_full[df_full['App Name'].isin(['#NAME?'])].sort_values(by='App Name')
df_full[df_full['App Name'].isin(['??'])].sort_values(by='App Name').head()
# df_full[df_full.duplicated()]

# checking for duplicates

dupli = df_full[df_full.duplicated()].sort_values(by=['Category','Rating','Last Updated'])

print(len(dupli))

dupli
#drop duplicates

df_full.drop_duplicates(inplace=True)
# checking for duplicates

df_full[df_full.duplicated()].sort_values(by=['Category','Rating','Last Updated'])
df_full.shape
df_full.Category.unique()
df_full.Category.value_counts()
df_full.Category.value_counts(normalize=True)[:10]
plt.figure(figsize=(12,5))

p = sns.set(style="darkgrid")

p = sns.countplot(x='Category',data=df_full)

_ = plt.setp(p.get_xticklabels(), rotation=90)  # Rotate labels

plt.title('App Category',size = 20);
df_full.Rating.unique()
df_full.isna().sum()
df_full.Rating.value_counts()
df_full.info()
# Since its a numerical columns, lets convert it to float

df_full.Rating = pd.to_numeric(df_full.Rating,errors='coerce')
df_full.Rating.isna().sum()
df_full.Rating.describe()
df_full.sort_values(by=['Rating'],ascending=False).head()
# Review distibution 

g = sns.kdeplot(df_full.Rating,shade=True,color='blue')

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of Rating',size = 20);
df_full.info()
# df_full[(df_full.Reviews.astype('str').str.isnumeric())]



# Since its a numerical columns, lets convert it to float

df_full.Reviews = pd.to_numeric(df_full.Reviews,errors='coerce',downcast='integer')
df_full.Reviews.head()
df_full.Reviews.isna().sum()
df_full.Reviews.describe()
df_full.sort_values(by='Reviews',ascending=False).head()
# Review distibution 

g = sns.kdeplot(df_full.Reviews,shade=True,color='blue')

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Distribution of Review',size = 20);
# df_full.info()
df_full.Installs.value_counts(dropna=False)
# remove "," and "+"

df_full.Installs = df_full.Installs.str.replace(',','').str.replace('+','')
df_full.Installs.value_counts(dropna=False)
# convert to numerical 

df_full.Installs = pd.to_numeric(df_full.Installs,errors='coerce')
# TODO



# pandas replace can replace list - list

# df_full['Install_cat'] = df_full.Installs.replace(sorted(df_full.Installs.unique()),range(0,len(sorted(df_full.Installs.unique())),1))
df_full.info()
df_full.sort_values(by='Installs',ascending=False).head()
# Review distibution 

g = sns.kdeplot(df_full.Installs,shade=True,color='blue')

g.set_xlabel("Installs")

g.set_ylabel("Frequency")

plt.title('Distribution of Installs',size = 20);
pd.set_option('display.max_rows', 2000)
df_full.Size.value_counts()[:15]
df_full.Size[~(df_full.Size.str.contains('M') | df_full.Size.str.contains('k'))]
# (11726 / len(df_full))*100
df_full.Size[(df_full.Size.str.contains(','))].head()
# df_full.Size.unique()
kb_index = df_full.Size[df_full.Size.str.contains('k')].index

mb_index = df_full.Size[(df_full.Size.str.contains('M'))].index

print(f"No.of App's in KB's : {len(kb_index)} ({round((len(kb_index)/len(df_full))*100,2)}%)")

print(f"No.of App's in MB's : {len(mb_index)} ({round((len(mb_index)/len(df_full))*100,2)}%)")
df_full.Size = df_full.Size.str.replace('M','')

df_full.Size = df_full.Size.str.replace('k','')

df_full.Size = df_full.Size.str.replace(',','')
df_full.Size.value_counts()[:5]

# tail charachter has removed
df_full.Size[(df_full.Size.str.contains(','))].head()

# comma has removed
# to_numeric() converts to float(default) and non-numeric values will be replaced with NAN 

df_full.Size = pd.to_numeric(df_full.Size,errors='coerce')
df_full.Size.head()
df_full.Size.value_counts(dropna=False).head()

# df_full.Size.isna().sum()
#holding NA's in a separate DataFrame

df_full_NA = df_full.copy()
# converting the size MB to KB (1 MB = 1000 KB)

df_full.Size.loc[mb_index] = df_full.Size.loc[mb_index] * 1000

df_full.Size.head()
# Drop NA's w.r.t Size column

df_full.dropna(subset=['Size'],inplace=True)
df_full.Size.describe()
df_full.Size.head()
df_full.Size.isna().sum()
# review the shape after removing of NA's

df_full.shape
df_full.sort_values(by='Size',ascending=False).head(2)
# Size distibution 

g = sns.kdeplot(df_full.Size,shade=True,color='blue')

g.set_xlabel("Size")

g.set_ylabel("Frequency")

plt.title('Distribution of Size',size = 20);
df_full.Price.value_counts().head()
df_full.Price = df_full.Price.str.replace('$','')
df_full.Price = pd.to_numeric(df_full.Price,errors='coerce')
df_full.Price.value_counts().head()
df_full.info()
df_full.Price.describe()
# df_full.sort_values(by='Price',ascending=False).head(5)

df_full[df_full.Price == 399.99]
# df_full[df_full.Price.astype('str').str.contains('scott')]
# Price distibution 

g = sns.kdeplot(df_full.Price,shade=True,color='blue', bw=1.5)

g.set_xlabel("Price")

g.set_ylabel("Frequency")

plt.title('Distribution of Price',size = 20);
df_full['Content Rating'].value_counts()
# 231211/len(df_full)
# removing numericals and + operator

df_full['Content Rating'] = df_full['Content Rating'].str.split(n=1,expand=True)[0]
df_full['Content Rating'].unique()
df_full[df_full['Content Rating'] == 'Unrated'].head()
df_full[df_full['Content Rating'] == 'Adults'].head()
df_full.info()
df_full.drop(columns=['App Name','Last Updated','Minimum Version','Latest Version'],inplace=True)
df_full.head()
df_full[df_full.duplicated()]
df_full.drop_duplicates(inplace=True)
# df_full.Category.value_counts()
# method : 1 (One-Hot)

df_full_dummy = pd.get_dummies(df_full,columns=['Category','Content Rating'],drop_first=True)

# df_full_dummy = pd.get_dummies(df_full,columns=['Category'],drop_first=True)
df_full_dummy.head()
X_dummy = df_full_dummy.drop(columns=['Rating'],axis=1)

y_dummy = df_full_dummy.loc[:,'Rating']

X_dummy.shape, y_dummy.shape
X_dummy.head(2)
y_dummy.head(2)
x_train,x_test,y_train,y_test = train_test_split(X_dummy,y_dummy,test_size=0.3,random_state = 14)
x_test.head()
# df_full.head()
# method : 2 (CatBoostEncoder)

import category_encoders as ce



df_full_cat = df_full.copy()



cbe = ce.CatBoostEncoder(cols=['Category','Content Rating'])

df_full_cat.loc[:,['Category','Content Rating']] = cbe.fit_transform(df_full_cat.loc[:,['Category','Content Rating']],df_full_cat['Rating'])



# cbe = ce.CatBoostEncoder(cols=['Category'])

# df_full_cat.loc[:,['Category']] = cbe.fit_transform(df_full_cat.loc[:,['Category']],df_full_cat['Rating'])
df_full_cat.head()
X_cat = df_full_cat.drop(columns = ['Rating'],axis=1)

y_cat = df_full_cat.loc[:,'Rating']

X_cat.shape , y_cat.shape
X_train,X_test,Y_train,Y_test = train_test_split(X_cat,y_cat,test_size=0.3,random_state = 14)
# Model traing and predicting

def dummy_model_building(model):

    from sklearn.metrics import mean_squared_error

    model.fit(x_train,y_train)

    print('trained')

    train_score = model.score(x_train , y_train)

    test_score = model.score(x_test , y_test)

    predict = model.predict(x_test)



    print('Train Score on Dummy : {}'.format(train_score))

    print('Test Score on Dummy : {}'.format(test_score))

    print(f'MSE : {mean_squared_error(y_test, predict)}')

#     print(classification_report(y_test, predict))



    print('\n \n')



    try:

        features = X_dummy.columns[:10]

        importances = model.feature_importances_[:10]

        indices = np.argsort(importances)



        plt.title('Feature Importances')

        plt.barh(range(len(indices)), importances[indices], color='b', align='center')

        plt.yticks(range(len(indices)), [features[i] for i in indices])

        plt.xlabel('Relative Importance')

        plt.show()

    except :

        print('This model does not support Feature Selection')



# cat score function

def cat_model_building(model):

    from sklearn.metrics import mean_squared_error

    model.fit(X_train,Y_train)

    train_score_ = model.score(X_train , Y_train)

    test_score_ = model.score(X_test , Y_test)

    predict_ = model.predict(X_test)



    print('Train Score on Cat_encode : {}'.format(train_score_))

    print('Test Score on Cat_encode : {}'.format(test_score_))

#     print(confusion_matrix(Y_test, predict_))

    print(f'MSE : {mean_squared_error(Y_test, predict_)}')

#     print(classification_report(Y_test, predict_))



    print('\n \n')



    try:

        features = X_cat.columns

        importances = model.feature_importances_

        indices = np.argsort(importances)



        plt.title('Feature Importances')

        plt.barh(range(len(indices)), importances[indices], color='b', align='center')

        plt.yticks(range(len(indices)), [features[i] for i in indices])

        plt.xlabel('Relative Importance')

        plt.show()

    except :

        print('This model does not support Feature Selection')
# DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dummy_model_building(dt)

print('\n')

cat_model_building(dt)
# LinearRegression

from sklearn.linear_model import LinearRegression

le = LinearRegression()

dummy_model_building(le)

print('\n')

cat_model_building(le)
# RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

dummy_model_building(rf)

print('\n')

cat_model_building(rf)
# Bagging Classifier

from sklearn.ensemble import BaggingRegressor



bc = BaggingRegressor()

dummy_model_building(bc)

print('\n')

cat_model_building(bc)
# Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor



gc = GradientBoostingRegressor()

dummy_model_building(gc)

print('\n')

cat_model_building(gc)
# AdaBoosting

from sklearn.ensemble import AdaBoostRegressor



ac = AdaBoostRegressor()

dummy_model_building(ac)

print('\n')

cat_model_building(ac)
# Stacking

from sklearn.ensemble import StackingRegressor

estimators = [('decisiontree', dt), ('randomforest', rf), ('bagging', bc), ('gradientboost', gc), ('Ada Boost', ac)]

sc = StackingRegressor(estimators)

dummy_model_building(sc)

print('\n')

cat_model_building(sc)