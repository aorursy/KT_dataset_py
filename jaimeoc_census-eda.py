#First, we import the libraries we need



%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns
df_census = pd.read_csv("../input/census.csv")

df_census_test = pd.read_csv("../input/census_test.csv")

df_census.head(10)
df_census.dtypes
df_census.greater_than_50k.unique()
#For df_census

df_census_selected = df_census.select_dtypes(include = ["object"])

df_census[df_census_selected.columns] = df_census_selected.apply(lambda x: x.str.strip())



#For df_cesus_test



df_census_selected = df_census_test.select_dtypes(include = ["object"])

df_census_test[df_census_selected.columns] = df_census_selected.apply(lambda x: x.str.strip())
df_census = df_census.astype(

{

        "workclass" : "category", 

        "education" : "category",

        "marital_status" : "category",

        "occupation" : "category",

        "relationship" : "category",

        "race" : "category",

        "native_country" : "category"

}

)



df_census_test = df_census_test.astype(

{

        "workclass" : "category", 

        "education" : "category",

        "marital_status" : "category",

        "occupation" : "category",

        "relationship" : "category",

        "race" : "category",

        "native_country" : "category"

}

)
for col_name in df_census.columns:

    if(isinstance(df_census[col_name].dtype, pd.core.dtypes.dtypes.CategoricalDtype)):

        print("Catergories for {} are:".format(col_name))

        for category in list(df_census[col_name].cat.categories):

            print("  -", category) 
pd.concat([df_census[["age", "hours_per_week"]].describe(), df_census_test[["age", "hours_per_week"]].describe() ], axis = 1)
sns.distplot(df_census["age"])
import scipy.stats as stats

print("H0 hypothesis test:",stats.normaltest(df_census["age"]))
sns.distplot(df_census_test["age"])
sns.distplot(df_census["hours_per_week"])
print("H0 hypothesis test:",stats.normaltest(df_census["hours_per_week"]))
sns.distplot(df_census_test["hours_per_week"])
print("Proportions of 1s for census: ", df_census.greater_than_50k.mean())

print("Proportions of 1s for census test: ", df_census_test.greater_than_50k.mean())
df_1 = pd.DataFrame({ "census": (df_census.workclass.value_counts()/df_census.workclass.count())})

df_2 = pd.DataFrame({ "census": (df_census_test.workclass.value_counts()/df_census_test.workclass.count())})

df_2 =df_2.rename(columns = {"census" : "census_test"})

pd.concat([df_1, df_2], axis = 1)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_test[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].value_counts()
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["education",  "greater_than_50k"]].groupby(["education"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["education",  "greater_than_50k"]].groupby(["education"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_test[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].value_counts()
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["occupation",  "greater_than_50k"]].groupby(["occupation"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["occupation",  "greater_than_50k"]].groupby(["occupation"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census["occupation"].value_counts()
gr_occupation = sns.barplot(y = "occupation", x ="greater_than_50k", data=df_census, estimator = np.mean)
gr_occupation = sns.barplot(y = "occupation", x ="greater_than_50k", data=df_census_test, estimator = np.mean)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["relationship",  "greater_than_50k"]].groupby(["relationship"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["relationship",  "greater_than_50k"]].groupby(["relationship"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["race",  "greater_than_50k"]].groupby(["race"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["race",  "greater_than_50k"]].groupby(["race"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["gender",  "greater_than_50k"]].groupby(["gender"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["gender",  "greater_than_50k"]].groupby(["gender"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["native_country",  "greater_than_50k"]].groupby(["native_country"])["greater_than_50k"].mean()})

df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["native_country",  "greater_than_50k"]].groupby(["native_country"])["greater_than_50k"].mean()})

pd.concat([df_census_mean, df_census_mean_test], axis = 1)
sns.countplot(y='occupation', hue='greater_than_50k', data=df_census)
sns.countplot(y='occupation', hue='greater_than_50k', data=df_census_test)
%matplotlib inline

import scipy.stats as stats

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split
sns.heatmap(df_census.isnull())
df_census.dropna(inplace = True)
from sklearn.model_selection import train_test_split

from warnings import simplefilter



simplefilter(action='ignore', category=FutureWarning)

df_census_dummies = pd.get_dummies(df_census[['workclass', 'education', 'marital_status',

       'occupation', 'relationship', 'race', 'gender',

       'native_country']])
df_census.drop(['workclass', 'education', 'marital_status',

       'occupation', 'relationship', 'race', 'gender',

       'native_country'], axis = 1, inplace=True)
#We get together original dataset with dummies



df_census_train = pd.concat([df_census_dummies, df_census], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df_census_train.drop('greater_than_50k',axis=1), 

                                                    df_census_train['greater_than_50k'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
# Our accuracy is:

logmodel_score = logmodel.score(X_test, y_test)

print("Model Score: " ,logmodel_score)
pd.DataFrame(metrics.confusion_matrix(y_test, predictions), columns = ["PREDICTED_FALSE","PREDICTED_TRUE" ], index = ["ACTUAL_FALSE", "ACTUAL_TRUE"])
a = logmodel.predict_proba(X_train)[:,1]

b = y_train



tot_bads=1.0*sum(b)

tot_goods=1.0*(len(b)-tot_bads)

elements_df = pd.DataFrame({'probability': a,'gbi': b})

pivot_elements_df = pd.pivot_table(elements_df, values='gbi', index=['probability'], aggfunc=[sum,len]).fillna(0)

max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0

cum_perc_bads_list = [0.0]

cum_perc_goods_list = [0.0]

cum_cp_minus = [0.0]

cum_cp_plus = [0.0]



for i in range(len(pivot_elements_df)):

    perc_goods =  ((pivot_elements_df['len'].iloc[i]['gbi'] - pivot_elements_df['sum'].iloc[i]['gbi']) / tot_goods)

    perc_bads = float(pivot_elements_df['sum']['gbi'].iloc[i]/ tot_bads)

    cum_perc_goods += perc_goods   

    cum_perc_bads += perc_bads



    

    cum_perc_bads_list.append(cum_perc_bads)

    cum_perc_goods_list.append(cum_perc_goods)

    cum_diff = cum_perc_bads-cum_perc_goods



    cum_cp_minus.append(0.0)    

    cum_cp_minus[-1] = cum_perc_bads_list[-1] - cum_perc_bads_list[-2]



    cum_cp_plus.append(0.0)

    cum_cp_plus[-1] = cum_perc_goods_list[-1] + cum_perc_goods_list[-2]

    

    

    if abs(cum_diff) > max_ks:

        max_ks = abs(cum_diff)



print('KS=',max_ks)
z_score = 0

for i in range(len(cum_cp_plus)):

    try:

        z_score +=  cum_cp_minus[i] * cum_cp_plus[i]

    except:

        pass

print('GINI=',1- z_score/100.0)
a = logmodel.predict_proba(X_test)[:,1]

b = y_test



tot_bads=1.0*sum(b)

tot_goods=1.0*(len(b)-tot_bads)

elements_df = pd.DataFrame({'probability': a,'gbi': b})

pivot_elements_df = pd.pivot_table(elements_df, values='gbi', index=['probability'], aggfunc=[sum,len]).fillna(0)

max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0

cum_perc_bads_list = [0.0]

cum_perc_goods_list = [0.0]

cum_cp_minus = [0.0]

cum_cp_plus = [0.0]



for i in range(len(pivot_elements_df)):

    perc_goods =  ((pivot_elements_df['len'].iloc[i]['gbi'] - pivot_elements_df['sum'].iloc[i]['gbi']) / tot_goods)

    perc_bads = float(pivot_elements_df['sum']['gbi'].iloc[i]/ tot_bads)

    cum_perc_goods += perc_goods   

    cum_perc_bads += perc_bads



    

    cum_perc_bads_list.append(cum_perc_bads)

    cum_perc_goods_list.append(cum_perc_goods)

    cum_diff = cum_perc_bads-cum_perc_goods



    cum_cp_minus.append(0.0)    

    cum_cp_minus[-1] = cum_perc_bads_list[-1] - cum_perc_bads_list[-2]



    cum_cp_plus.append(0.0)

    cum_cp_plus[-1] = cum_perc_goods_list[-1] + cum_perc_goods_list[-2]

    

    

    if abs(cum_diff) > max_ks:

        max_ks = abs(cum_diff)



print('KS=',max_ks)
z_score = 0

for i in range(len(cum_cp_plus)):

    try:

        z_score +=  cum_cp_minus[i] * cum_cp_plus[i]

    except:

        pass

print('GINI=',1- z_score/100.0)