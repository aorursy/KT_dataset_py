#import plotly 

#import plotly.plotly as py

#import plotly.graph_objs as go



import matplotlib.pyplot as plt

import seaborn as sns







#Scikit learn librairies

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import scale



import pandas as pd

import numpy as np



%matplotlib inline
df=pd.read_csv('../input/home-credit-default-risk/application_train.csv')

df_bureau=pd.read_csv('../input/home-credit-default-risk/bureau.csv')

dftmp= pd.read_csv('../input/home-credit-default-risk/application_train.csv')
df.shape
df.head(10)
df.dtypes.value_counts()
fig11=plt.figure()

ax11=plt.axes()

the_target = dftmp['TARGET']

the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)

plt.title('Target repartition')

ax11 = ax11.set(xlabel='Default proportion', ylabel='Number of people')

the_target.value_counts().plot.bar()

plt.show()
print(df.shape)

df=df.merge(right=df_bureau,how='inner', on='SK_ID_CURR')

print(df.shape)
df.TARGET.isnull().sum()
df.dtypes.value_counts()
fig11=plt.figure()

ax11=plt.axes()

the_target = dftmp['TARGET']

the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)

plt.title('Target repartition')

ax11 = ax11.set(xlabel='Default proportion', ylabel='Number of people')

the_target.value_counts().plot.pie(startangle=90, autopct='%1.1f%%')

plt.show()
#A function to print every graph with the ID as 

def print_all_values():

    df1=df.drop('SK_ID_CURR',axis=1)

    cols=df1.columns

    for col in cols:

        if (df[col].dtypes !='object'):



            fig1=plt.figure()

            ax1=plt.axes()

            plt.scatter(df[[col]],df.SK_ID_CURR,alpha=1,s=0.5)

            plt.title(col)

            ax1 = ax1.set(xlabel=col, ylabel='ID')

            plt.show()

            

            

print_all_values()
#Plotting the income of the people making default

df1=df[df.AMT_INCOME_TOTAL <1600000.0]

df1=df1[df.TARGET ==1 ]

df2=df[df.TARGET ==1 ]





fig2=plt.figure()

ax2=plt.axes()

plt.scatter(df2.SK_ID_CURR ,df2.AMT_INCOME_TOTAL ,alpha=1)

plt.title('Repartition des salaires sans limite de maximum')

ax2 = ax2.set(xlabel='ID', ylabel='Salaire')



fig1=plt.figure()

ax1=plt.axes()

plt.scatter(df1.SK_ID_CURR ,df1.AMT_INCOME_TOTAL ,alpha=1)

plt.title('Repartition des salaires avec une limite de maximum de 1600000$')

ax1 = ax1.set(xlabel='ID', ylabel='Salaire')

plt.show()
print(df.shape)

df=df[df.AMT_INCOME_TOTAL <1750000.0]

df=df[df.CNT_FAM_MEMBERS <12]

df=df[df.OBS_30_CNT_SOCIAL_CIRCLE <50]

df=df[df.DEF_30_CNT_SOCIAL_CIRCLE <20]

df=df[df.OBS_60_CNT_SOCIAL_CIRCLE <55]

df=df[df.DEF_60_CNT_SOCIAL_CIRCLE <15]

df=df[df.AMT_REQ_CREDIT_BUREAU_HOUR <4]

df=df[df.AMT_REQ_CREDIT_BUREAU_QRT <55]

df=df[df.CNT_CREDIT_PROLONG <6.5]

print(df.shape)
df['OWN_CAR_AGE']=df['OWN_CAR_AGE'].fillna(0)





cols=['APARTMENTS_AVG','BASEMENTAREA_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI']        

for i in df.index:

    if (df.loc[i,'FLAG_OWN_REALTY'] =='N'):

        for col in cols:

            df.set_value(i,col,0)

            



df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna('Unknown')

df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].fillna('Unknown')
fig1=plt.figure()

ax1=plt.axes()

plt.scatter(df.OWN_CAR_AGE,df.FLAG_OWN_CAR,color='cyan')

plt.title('Graphique de "Own_Car" et "Own_Car_Age"')

ax1 = ax1.set(xlabel='Own_Car_Age', ylabel='Own_Car')

plt.show()
df['DAYS_BIRTH'] = df['DAYS_BIRTH']/(-365)

df=df.rename(columns={'DAYS_BIRTH':'AGE'})

df.AGE.describe()
#Dataset of missing values order by percentage

def nan_count_df(df_to_print):

    

    nan_count = df_to_print.isnull().sum()



    nan_percentage = (nan_count / len(df))*100



    nan_df=pd.concat([nan_percentage], axis=1)

    nan_df=nan_df.rename(columns={0:'Percentage'})

    nan_df=nan_df[nan_df.Percentage != 0]

    nan_df = nan_df.sort_values(by='Percentage',ascending=False)

    return nan_df



nan_df=nan_count_df(df)

nan_df
print(df.shape)

def delete_columns(df_transformed,df_missing_values,max_value):

        cols=df_missing_values[df_missing_values['Percentage']>=max_value].T.columns

        for col in cols:

            df_transformed=df_transformed.drop(col, axis=1)

        

        return df_transformed

df=delete_columns(df,nan_df,62)

print(df.shape)
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#useless columns

columns_to_drop = ['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','NAME_TYPE_SUITE','FLAG_MOBIL','FLAG_CONT_MOBILE']

df=df.drop(columns=columns_to_drop)
#Encodage pour 2 catégories

def two_cat_encoding(df_to_transf):

    le = LabelEncoder()



    for cols in df_to_transf:

        if df_to_transf[cols].dtype == 'object':

            if len(list(df_to_transf[cols].unique())) == 2:

                le.fit(df_to_transf[cols])

                df_to_transf[cols] = le.transform(df_to_transf[cols])

    return df_to_transf

df=two_cat_encoding(df)
df = pd.get_dummies(df)
print('Les nouvelles dimensions du dataframes sont :\n', df.shape)
df_columns=df.columns

df=df.dropna()
X =df.drop('TARGET',axis=1)

y = df['TARGET']  



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(fit_intercept=True,intercept_scaling=1,max_iter=200,tol=0.0001,random_state=None)

logisticRegr.fit(X_train, y_train)
#ERROR

error = (1 - logisticRegr.score(X_test, y_test))*100

print('Score  = ',logisticRegr.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(n_components=None)

lda.fit(X_train, y_train)
#ERROR

error = (1 - lda.score(X_test, y_test))*100

print('Score  = ',lda.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0)

rf.fit(X_train,y_train)
error = (1 - rf.score(X_test, y_test))*100

print('Score  = ',rf.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)
error = (1 - clf.score(X_test, y_test))*100

print('Score  = ',clf.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn.metrics import classification_report, confusion_matrix

predictions = logisticRegr.predict(X_test)



print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import cross_val_score



scores = cross_val_score(estimator = logisticRegr , 

                         X=X_train, 

                         y=y_train, 

                         cv=3)

print('Cross-validation scores de réussite: %s' %(scores))

print('CV précision: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
print('Taux de réussite par modèle:\n\nRégression Logistique:',logisticRegr.score(X_test, y_test)*100,'%','\n\nLDA:',lda.score(X_test, y_test)*100,'%','\n\nRandom Forest Classifier:',rf.score(X_test, y_test)*100,'%','\n\nDecision Tree Classifier:',clf.score(X_test, y_test)*100,'%')