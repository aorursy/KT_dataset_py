import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import itertools
application_data = pd.read_csv(r'/kaggle/input/application_data.csv')

previous_application = pd.read_csv(r'/kaggle/input/previous_application.csv')

columns_description = pd.read_csv(r'/kaggle/input/columns_description.csv',skiprows=1)
print ("application_data     :",application_data.shape)

print ("previous_application :",previous_application.shape)

print ("columns_description  :",columns_description.shape)
pd.set_option("display.max_rows", None, "display.max_columns", None)

display("application_data")

display(application_data.head(3))
display("previous_application ")

display(previous_application.head(3))
display("columns_description")

pd.set_option('display.max_colwidth', -1)

columns_description=columns_description.drop(['1'],axis=1)

display(columns_description)
fig = plt.figure(figsize=(18,6))

miss_previous_application = pd.DataFrame((previous_application.isnull().sum())*100/previous_application.shape[0]).reset_index()

miss_previous_application["type"] = "previous_application"

ax = sns.pointplot("index",0,data=miss_previous_application,hue="type")

plt.xticks(rotation =90,fontsize =7)

plt.title("Percentage of Missing values in previous_application")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")

fig.set_facecolor("lightgrey")
round(100*(previous_application.isnull().sum()/len(previous_application.index)),2)
previous_application=previous_application.drop([ 'AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',

       "RATE_INTEREST_PRIVILEGED"],axis=1)
fig = plt.figure(figsize=(18,6))

miss_previous_application = pd.DataFrame((previous_application.isnull().sum())*100/previous_application.shape[0]).reset_index()

miss_previous_application["type"] = "previous_application"

ax = sns.pointplot("index",0,data=miss_previous_application,hue="type")

plt.xticks(rotation =90,fontsize =7)

plt.title("Percentage of Missing values in previous_application")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")

fig.set_facecolor("lightgrey")
round(100*(previous_application.isnull().sum()/len(previous_application.index)),2)
print("AMT_ANNUITY NULL COUNT:" ,previous_application['AMT_ANNUITY'].isnull().sum())
previous_application['AMT_ANNUITY'].describe()
sns.set_style('whitegrid') 

sns.distplot(previous_application['AMT_ANNUITY']) 

plt.show()

print("AMT_GOODS_PRICE NULL COUNT:" ,previous_application['AMT_GOODS_PRICE'].isnull().sum())
previous_application['AMT_GOODS_PRICE'].describe()
sns.set_style('whitegrid') 

sns.distplot(previous_application['AMT_GOODS_PRICE']) 

plt.show()

print("NAME_TYPE_SUITE NULL COUNT:" ,previous_application['NAME_TYPE_SUITE'].isnull().sum())
previous_application['NAME_TYPE_SUITE'].value_counts()
print("CNT_PAYMENT NULL COUNT:" ,previous_application['CNT_PAYMENT'].isnull().sum())
previous_application['CNT_PAYMENT'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['CNT_PAYMENT']) 

plt.show()
print("DAYS_FIRST_DRAWING :" ,previous_application['CNT_PAYMENT'].isnull().sum())
previous_application['DAYS_FIRST_DRAWING'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['DAYS_FIRST_DRAWING']) 

plt.show()
print("DAYS_FIRST_DUE :" ,previous_application['DAYS_FIRST_DUE'].isnull().sum())
previous_application['DAYS_FIRST_DUE'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['DAYS_FIRST_DUE']) 

plt.show()
print("DAYS_LAST_DUE_1ST_VERSION :" ,previous_application['DAYS_LAST_DUE_1ST_VERSION'].isnull().sum())
previous_application['DAYS_LAST_DUE_1ST_VERSION'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['DAYS_LAST_DUE_1ST_VERSION']) 

plt.show()
print("DAYS_LAST_DUE:" ,previous_application['DAYS_LAST_DUE'].isnull().sum())
previous_application['DAYS_LAST_DUE'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['DAYS_LAST_DUE']) 

plt.show()
print("DAYS_TERMINATION :" ,previous_application['DAYS_TERMINATION'].isnull().sum())
previous_application['DAYS_TERMINATION'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['DAYS_TERMINATION']) 

plt.show()
print("NFLAG_INSURED_ON_APPROVAL:" ,previous_application['NFLAG_INSURED_ON_APPROVAL'].isnull().sum())
previous_application['NFLAG_INSURED_ON_APPROVAL'].value_counts()
previous_application.isnull().sum()
print("AMT_CREDIT :" ,previous_application['AMT_CREDIT'].isnull().sum())
previous_application['AMT_CREDIT'].describe()
sns.set_style('whitegrid') 

sns.boxplot(previous_application['AMT_CREDIT']) 

plt.show()

print("PRODUCT_COMBINATION :" ,previous_application['PRODUCT_COMBINATION'].isnull().sum())
previous_application['PRODUCT_COMBINATION'].value_counts()
class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
obj_dtypes = [i for i in previous_application.select_dtypes(include=np.object).columns if i not in ["type"] ]

num_dtypes = [i for i in previous_application.select_dtypes(include = np.number).columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]
print(color.BOLD + color.PURPLE + 'Categorical Columns' + color.END, "\n")

for x in range(len(obj_dtypes)): 

    print(obj_dtypes[x])

print(color.BOLD + color.PURPLE + 'Numerical' + color.END, "\n")

for x in range(len(obj_dtypes)): 

    print(obj_dtypes[x])
fig = plt.figure(figsize=(18,6))

miss_application_data = pd.DataFrame((application_data.isnull().sum())*100/application_data.shape[0]).reset_index()

miss_application_data["type"] = "application_data"

ax = sns.pointplot("index",0,data=miss_application_data,hue="type")

plt.xticks(rotation =90,fontsize =7)

plt.title("Percentage of Missing values in application_data")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")

fig.set_facecolor("lightgrey")
round(100*(application_data.isnull().sum()/len(application_data.index)),2)
application_data=application_data.drop([ 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',

       'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',

       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',

       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',

       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',

       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',

       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',

       'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',

       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',

       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',

       'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',

       'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',

       'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',

       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',

       'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',

       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',"OWN_CAR_AGE","OCCUPATION_TYPE"],axis=1)
fig = plt.figure(figsize=(18,6))

miss_application_data = pd.DataFrame((application_data.isnull().sum())*100/application_data.shape[0]).reset_index()

miss_application_data["type"] = "application_data"

ax = sns.pointplot("index",0,data=miss_application_data,hue="type")

plt.xticks(rotation =90,fontsize =7)

plt.title("Percentage of Missing values in application_data")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")

fig.set_facecolor("lightgrey")



round(100*(application_data.isnull().sum()/len(application_data.index)),2)
print("AMT_REQ_CREDIT_BUREAU_DAY NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_DAY'].isnull().sum())
application_data['AMT_REQ_CREDIT_BUREAU_DAY'].describe()
print("AMT_REQ_CREDIT_BUREAU_HOUR NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_HOUR'].isnull().sum())
application_data['AMT_REQ_CREDIT_BUREAU_HOUR'].describe()
print("AMT_REQ_CREDIT_BUREAU_MON NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_MON'].isnull().sum())
application_data['AMT_REQ_CREDIT_BUREAU_MON'].describe()
print("AMT_REQ_CREDIT_BUREAU_QRT NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_QRT'].isnull().sum())
print("AMT_REQ_CREDIT_BUREAU_WEEK NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_WEEK'].isnull().sum())
application_data['AMT_REQ_CREDIT_BUREAU_WEEK'].describe()
print("AMT_REQ_CREDIT_BUREAU_YEAR NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum())
application_data['AMT_REQ_CREDIT_BUREAU_YEAR'].describe()
print("DEF_30_CNT_SOCIAL_CIRCLE NAN COUNT :" ,application_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull().sum())
application_data['DEF_30_CNT_SOCIAL_CIRCLE'].describe()
print("DEF_30_CNT_SOCIAL_CIRCLE :" ,application_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull().sum())
application_data['DEF_30_CNT_SOCIAL_CIRCLE'].describe()
print("OBS_60_CNT_SOCIAL_CIRCLE :" ,application_data['OBS_60_CNT_SOCIAL_CIRCLE'].isnull().sum())
application_data['OBS_60_CNT_SOCIAL_CIRCLE'].describe()
print("DEF_60_CNT_SOCIAL_CIRCLE :" ,application_data['DEF_60_CNT_SOCIAL_CIRCLE'].isnull().sum())
application_data['DEF_60_CNT_SOCIAL_CIRCLE'].describe()
application_data.isnull().sum()
print("AMT_ANNUITY  :" ,application_data['AMT_ANNUITY'].isnull().sum())
application_data['AMT_ANNUITY'].describe()
sns.set_style('whitegrid') 

sns.distplot(application_data['AMT_ANNUITY']) 

plt.show()
print("AMT_GOODS_PRICE   :" ,application_data['AMT_GOODS_PRICE'].isnull().sum())
application_data['AMT_GOODS_PRICE'].describe()
sns.set_style('whitegrid') 

sns.distplot(application_data['AMT_GOODS_PRICE']) 

plt.show()
print("NAME_TYPE_SUITE :" ,application_data['NAME_TYPE_SUITE'].isnull().sum())
application_data['NAME_TYPE_SUITE'].value_counts()
print("CNT_FAM_MEMBERS :" ,application_data['CNT_FAM_MEMBERS'].isnull().sum())
application_data['CNT_FAM_MEMBERS'].describe()
sns.set_style('whitegrid') 

sns.distplot(application_data['CNT_FAM_MEMBERS']) 

plt.show()
print("DAYS_LAST_PHONE_CHANGE :" ,application_data['DAYS_LAST_PHONE_CHANGE'].isnull().sum())
application_data['DAYS_LAST_PHONE_CHANGE'].describe()
import statistics 

statistics.mode(application_data['DAYS_LAST_PHONE_CHANGE'])
print(type(application_data.info()))
application_data['DAYS_BIRTH'] = abs(application_data['DAYS_BIRTH'])

application_data['DAYS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH'])

application_data['DAYS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH'])

application_data['DAYS_LAST_PHONE_CHANGE'] = abs(application_data['DAYS_LAST_PHONE_CHANGE'])



display("application_data")

display(application_data.head())
obj_dtypes = [i for i in application_data.select_dtypes(include=np.object).columns if i not in ["type"] ]

num_dtypes = [i for i in application_data.select_dtypes(include = np.number).columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]
print(color.BOLD + color.PURPLE + 'Categorical Columns' + color.END, "\n")

for x in range(len(obj_dtypes)): 

    print(obj_dtypes[x])
print(color.BOLD + color.PURPLE +"Numerical Columns" + color.END, "\n")

for x in range(len(num_dtypes)): 

    print(num_dtypes[x])
fig = plt.figure(figsize=(13,6))

plt.subplot(121)

application_data["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["red","yellow"],startangle = 60,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)

plt.title("Distribution of gender")

plt.show()
plt.figure(figsize=(14,7))

plt.subplot(121)

application_data["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=["repayer","defaulter"],

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)

plt.title("Distribution of target variable")



plt.subplot(122)

ax = application_data["TARGET"].value_counts().plot(kind="barh")



for i,j in enumerate(application_data["TARGET"].value_counts().values):

    ax.text(.7,i,j,weight = "bold",fontsize=20)



plt.title("Count of target variable")

plt.show()
application_data_x = application_data[[x for x in application_data.columns if x not in ["TARGET"]]]

previous_application_x = previous_application[[x for x in previous_application.columns if x not in ["TARGET"]]]

application_data_x["type"] = "application_data"

previous_application_x["type"] = "previous_application"

data = pd.concat([application_data_x,previous_application_x],axis=0) 
plt.figure(figsize=(14,7))

plt.subplot(121)

data[data["type"] == "application_data"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["orange","red"],startangle = 60,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("distribution of contract types in application_data")



plt.subplot(122)

data[data["type"] == "previous_application"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.2f%%",colors = ["red","yellow","green",'BLACK'],startangle = 60,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.ylabel("")

plt.title("distribution of contract types in previous_application")

plt.show()



plt.show()
fig = plt.figure(figsize=(13,6))

plt.subplot(121)

data[data["type"] == "application_data"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["red","yellow"],startangle = 60,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)

plt.title("distribution of gender in application_data")

plt.show()
fig  = plt.figure(figsize=(13,6))

plt.subplot(121)

ax = sns.countplot("NAME_CONTRACT_TYPE",hue="CODE_GENDER",data=data[data["type"] == "application_data"],palette=["r","b","g"])

ax.set_facecolor("lightgrey")

ax.set_title("Distribution of Contract type by gender -application_data")





plt.show()

fig = plt.figure(figsize=(13,6))



plt.subplot(121)

data["FLAG_OWN_CAR"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["gold","orangered"],startangle = 60,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)

plt.title("distribution of client owning a car")



plt.subplot(122)

data[data["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["b","orangered"],startangle = 90,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)

plt.title("distribution of client owning a car by gender")



plt.show()
plt.figure(figsize=(13,6))

plt.subplot(121)

data["FLAG_OWN_REALTY"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["skyblue","gold"],startangle = 90,

                                              wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[0.05,0],shadow =True)

plt.title("Distribution of client owns a house or flat")



plt.subplot(122)

data[data["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["orangered","b"],startangle = 90,

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)

plt.title("Distribution of client owning a house or flat by gender")

plt.show()
fig = plt.figure(figsize=(12,10))

plt.subplot(211)

sns.countplot(application_data["CNT_CHILDREN"],palette="Set1",hue=application_data["TARGET"])

plt.legend(loc="upper center")

plt.title(" Distribution of Number of children client has  by repayment status")

plt.subplot(212)

sns.countplot(application_data["CNT_FAM_MEMBERS"],palette="Set1",hue=application_data["TARGET"])

plt.legend(loc="upper center")

plt.title(" Distribution of Number of family members client has  by repayment status")

fig.set_facecolor("lightblue")
default = application_data[application_data["TARGET"]==1][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

non_default = application_data[application_data["TARGET"]==0][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]



d_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

d_length = len(d_cols)



fig = plt.figure(figsize=(16,4))

for i,j in itertools.zip_longest(d_cols,range(d_length)):

    plt.subplot(1,4,j+1)

    default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,

                                        wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)

    circ = plt.Circle((0,0),.7,color="white")

    plt.gca().add_artist(circ)

    plt.ylabel("")

    plt.title(i+"-Defaulter")





fig = plt.figure(figsize=(16,4))

for i,j in itertools.zip_longest(d_cols,range(d_length)):

    plt.subplot(1,4,j+1)

    non_default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,

                                           wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)

    circ = plt.Circle((0,0),.7,color="white")

    plt.gca().add_artist(circ)

    plt.ylabel("")

    plt.title(i+"-Repayer")
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

length = len(cols)

cs = ["r","b","g","k"]



ax = plt.figure(figsize=(18,18))

ax.set_facecolor("lightgrey")

for i,j,k in itertools.zip_longest(cols,range(length),cs):

    plt.subplot(2,2,j+1)

    sns.distplot(data[data[i].notnull()][i],color=k)

    plt.axvline(data[i].mean(),label = "mean",linestyle="dashed",color="k")

    plt.legend(loc="best")

    plt.title(i)

    plt.subplots_adjust(hspace = .2)
df = application_data.groupby("TARGET")[cols].describe().transpose().reset_index()

df = df[df["level_1"].isin([ 'mean', 'std', 'min', 'max'])] 

df_x = df[["level_0","level_1",0]]

df_y = df[["level_0","level_1",1]]

df_x = df_x.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 0:"amount"})

df_x["type"] = "REPAYER"

df_y = df_y.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 1:"amount"})

df_y["type"] = "DEFAULTER"

df_new = pd.concat([df_x,df_y],axis = 0)



stat = df_new["statistic"].unique().tolist()

length = len(stat)



plt.figure(figsize=(13,15))



for i,j in itertools.zip_longest(stat,range(length)):

    plt.subplot(2,2,j+1)

    fig = sns.barplot(df_new[df_new["statistic"] == i]["amount_type"],df_new[df_new["statistic"] == i]["amount"],

                hue=df_new[df_new["statistic"] == i]["type"],palette=["g","r"])

    plt.title(i + "--Defaulters vs Non defaulters")

    plt.subplots_adjust(hspace = .4)

    fig.set_facecolor("lightgrey")
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']



df1 = data.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()



df_f   = df1[["index","F"]]

df_f   = df_f.rename(columns={'index':"amt_type", 'F':"amount"})

df_f["gender"] = "FEMALE"

df_m   = df1[["index","M"]]

df_m   = df_m.rename(columns={'index':"amt_type", 'M':"amount"})

df_m["gender"] = "MALE"

df_xna = df1[["index","XNA"]]

df_xna = df_xna.rename(columns={'index':"amt_type", 'XNA':"amount"})

df_xna["gender"] = "XNA"



df_gen = pd.concat([df_m,df_f,df_xna],axis=0)



plt.figure(figsize=(12,5))

ax = sns.barplot("amt_type","amount",data=df_gen,hue="gender",palette="Set1")

plt.title("Average Income,credit,annuity & goods_price by gender")

plt.show()
fig = plt.figure(figsize=(10,8))

plt.scatter(application_data[application_data["TARGET"]==0]['AMT_ANNUITY'],application_data[application_data["TARGET"]==0]['AMT_CREDIT'],s=35,

            color="b",alpha=.5,label="REPAYER",linewidth=.5,edgecolor="k")

plt.scatter(application_data[application_data["TARGET"]==1]['AMT_ANNUITY'],application_data[application_data["TARGET"]==1]['AMT_CREDIT'],s=35,

            color="r",alpha=.2,label="DEFAULTER",linewidth=.5,edgecolor="k")

plt.legend(loc="best",prop={"size":15})

plt.xlabel("AMT_ANNUITY")

plt.ylabel("AMT_CREDIT")

plt.title("Scatter plot between credit amount and annuity amount")

plt.show()
amt = application_data[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',

                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]

amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]

sns.pairplot(amt,hue="TARGET",palette=["b","r"])

plt.show()
plt.figure(figsize=(18,12))

plt.subplot(121)

sns.countplot(y=data["NAME_TYPE_SUITE"],

              palette="Set2",

              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])

plt.title("Distribution of Suite type")



plt.subplot(122)

sns.countplot(y=data["NAME_TYPE_SUITE"],

              hue=data["CODE_GENDER"],palette="Set2",

              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])

plt.ylabel("")

plt.title("Distribution of Suite type by gender")

plt.subplots_adjust(wspace = .4)
plt.figure(figsize=(18,12))

plt.subplot(121)

sns.countplot(y=data["NAME_INCOME_TYPE"],

              palette="Set2",

              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])

plt.title("Distribution of client income type")



plt.subplot(122)

sns.countplot(y=data["NAME_INCOME_TYPE"],

              hue=data["CODE_GENDER"],

              palette="Set2",

              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])

plt.ylabel("")

plt.title("Distribution of client income  type by gender")

plt.subplots_adjust(wspace = .4)
plt.figure(figsize=(25,25))

plt.subplot(121)

application_data[application_data["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=12,autopct = "%1.0f%%",

                                                                                                 colors = sns.color_palette("Set1"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("Distribution of Education type for Repayers",color="b")



plt.subplot(122)

application_data[application_data["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=12,autopct = "%1.0f%%",

                                                                                                 colors = sns.color_palette("Set1"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("Distribution of Education type for Defaulters",color="b")

plt.ylabel("")

plt.show()
edu = data.groupby(['NAME_EDUCATION_TYPE','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL',ascending=False)

fig = plt.figure(figsize=(13,7))

ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL',data=edu,hue='NAME_EDUCATION_TYPE',palette="seismic")

ax.set_facecolor("k")

plt.title(" Average Earnings by different professions and education types")

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(121)

application_data[application_data["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",

                                                             startangle=120,colors = sns.color_palette("Set2",7),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0,0])



plt.title("Distribution of Family status for Repayers",color="b")



plt.subplot(122)

application_data[application_data["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",

                                                    startangle=120,colors = sns.color_palette("Set2",7),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0])





plt.title("Distribution of Family status for Defaulters",color="b")

plt.ylabel("")

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(121)

application_data[application_data["TARGET"]==0]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=10,

                                                             colors = sns.color_palette("Spectral"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)



plt.title("Distribution of housing type  for Repayer",color="b")



plt.subplot(122)

application_data[application_data["TARGET"]==1]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=10,

                                                    colors = sns.color_palette("Spectral"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)





plt.title("Distribution of housing type for Defaulters",color="b")

plt.ylabel("")

plt.show()
fig = plt.figure(figsize=(13,8))



plt.subplot(121)

sns.violinplot(y=application_data[application_data["TARGET"]==0]["REGION_POPULATION_RELATIVE"]

               ,x=application_data[application_data["TARGET"]==0]["NAME_CONTRACT_TYPE"],

               palette="Set1")

plt.title("Distribution of region population for Non Default loans",color="b")

plt.subplot(122)

sns.violinplot(y = application_data[application_data["TARGET"]==1]["REGION_POPULATION_RELATIVE"]

               ,x=application_data[application_data["TARGET"]==1]["NAME_CONTRACT_TYPE"]

               ,palette="Set1")

plt.title("Distribution of region population  for  Default loans",color="b")



plt.subplots_adjust(wspace = .2)

fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(13,15))



plt.subplot(221)

sns.distplot(application_data[application_data["TARGET"]==0]["DAYS_BIRTH"],color="b")

plt.title("Age Distribution of repayers")



plt.subplot(222)

sns.distplot(application_data[application_data["TARGET"]==1]["DAYS_BIRTH"],color="r")

plt.title("Age Distribution of defaulters")



plt.subplot(223)

sns.lvplot(application_data["TARGET"],application_data["DAYS_BIRTH"],hue=application_data["CODE_GENDER"],palette=["b","grey","m"])

plt.axhline(application_data["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")

plt.legend(loc="lower right")

plt.title("Client age vs Loan repayment status(hue=gender)")



plt.subplot(224)

sns.lvplot(application_data["TARGET"],application_data["DAYS_BIRTH"],hue=application_data["NAME_CONTRACT_TYPE"],palette=["r","g"])

plt.axhline(application_data["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")

plt.legend(loc="lower right")

plt.title("Client age vs Loan repayment status(hue=contract type)")



plt.subplots_adjust(wspace = .2,hspace = .3)



fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(13,5))



plt.subplot(121)

sns.distplot(application_data[application_data["TARGET"]==0]["DAYS_EMPLOYED"],color="b")

plt.title("days employed distribution of repayers")



plt.subplot(122)

sns.distplot(application_data[application_data["TARGET"]==1]["DAYS_EMPLOYED"],color="r")

plt.title("days employed distribution of defaulters")



fig.set_facecolor("ghostwhite")
fig = plt.figure(figsize=(13,5))



plt.subplot(121)

sns.distplot(application_data[application_data["TARGET"]==0]["DAYS_REGISTRATION"],color="b")

plt.title("registration days distribution of repayers")



plt.subplot(122)

sns.distplot(application_data[application_data["TARGET"]==1]["DAYS_REGISTRATION"],color="r")

plt.title("registration days distribution of defaulter")



fig.set_facecolor("ghostwhite")
x   = application_data[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',

       'FLAG_PHONE', 'FLAG_EMAIL',"TARGET"]]

x["TARGET"] = x["TARGET"].replace({0:"repayers",1:"defaulters"})

x  = x.replace({1:"YES",0:"NO"})



cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',

       'FLAG_PHONE', 'FLAG_EMAIL']

length = len(cols)



fig = plt.figure(figsize=(15,12))

fig.set_facecolor("lightgrey")



for i,j in itertools.zip_longest(cols,range(length)):

    plt.subplot(2,3,j+1)

    sns.countplot(x[i],hue=x["TARGET"],palette=["r","g"])

    plt.title(i,color="b")
fig = plt.figure(figsize=(13,13))

plt.subplot(221)

application_data[application_data["TARGET"]==0]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=12,

                                                             colors = sns.color_palette("Pastel1"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)



plt.title("Distribution of region rating  for Repayers",color="b")



plt.subplot(222)

application_data[application_data["TARGET"]==1]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=12,

                                                    colors = sns.color_palette("Pastel1"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)





plt.title("Distribution of region rating  for Defaulters",color="b")

plt.ylabel("")



plt.subplot(223)

application_data[application_data["TARGET"]==0]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=12,

                                                             colors = sns.color_palette("Paired"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)



plt.title("Distribution of city region rating   for Repayers",color="b")



plt.subplot(224)

application_data[application_data["TARGET"]==1]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=12,

                                                    colors = sns.color_palette("Paired"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)





plt.title("Distribution of city region rating  for Defaulters",color="b")

plt.ylabel("")

fig.set_facecolor("ivory")
day = application_data.groupby("TARGET").agg({"WEEKDAY_APPR_PROCESS_START":"value_counts"})

day = day.rename(columns={"WEEKDAY_APPR_PROCESS_START":"value_counts"})

day = day.reset_index()

day_0 = day[:7]

day_1 = day[7:]

day_0["percentage"] = day_0["value_counts"]*100/day_0["value_counts"].sum()

day_1["percentage"] = day_1["value_counts"]*100/day_1["value_counts"].sum()

days = pd.concat([day_0,day_1],axis=0)

days["TARGET"] = days.replace({1:"defaulters",0:"repayers"})



fig = plt.figure(figsize=(13,15))

plt.subplot(211)

order = ['SUNDAY', 'MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY', 'FRIDAY', 'SATURDAY']

ax= sns.barplot("WEEKDAY_APPR_PROCESS_START","percentage",data=days,

                hue="TARGET",order=order,palette="prism")

ax.set_facecolor("k")

ax.set_title("Peak days for applying loans (defaulters vs repayers)")



hr = application_data.groupby("TARGET").agg({"HOUR_APPR_PROCESS_START":"value_counts"})

hr = hr.rename(columns={"HOUR_APPR_PROCESS_START":"value_counts"}).reset_index()

hr_0 = hr[hr["TARGET"]==0]

hr_1 = hr[hr["TARGET"]==1]

hr_0["percentage"] = hr_0["value_counts"]*100/hr_0["value_counts"].sum()

hr_1["percentage"] = hr_1["value_counts"]*100/hr_1["value_counts"].sum()

hrs = pd.concat([hr_0,hr_1],axis=0)

hrs["TARGET"] = hrs["TARGET"].replace({1:"defaulters",0:"repayers"})

hrs = hrs.sort_values(by="HOUR_APPR_PROCESS_START",ascending=True)



plt.subplot(212)

ax1 = sns.pointplot("HOUR_APPR_PROCESS_START","percentage",

                    data=hrs,hue="TARGET",palette="prism")

ax1.set_facecolor("k")

ax1.set_title("Peak hours for applying loans (defaulters vs repayers)")

fig.set_facecolor("snow")
org = application_data.groupby("TARGET").agg({"ORGANIZATION_TYPE":"value_counts"})

org = org.rename(columns = {"ORGANIZATION_TYPE":"value_counts"}).reset_index()

org_0 = org[org["TARGET"] == 0]

org_1 = org[org["TARGET"] == 1]

org_0["percentage"] = org_0["value_counts"]*100/org_0["value_counts"].sum()

org_1["percentage"] = org_1["value_counts"]*100/org_1["value_counts"].sum()



organization = pd.concat([org_0,org_1],axis=0)

organization = organization.sort_values(by="ORGANIZATION_TYPE",ascending=True)



organization["TARGET"] = organization["TARGET"].replace({0:"repayers",1:"defaulters"})



organization

plt.figure(figsize=(13,7))

ax = sns.pointplot("ORGANIZATION_TYPE","percentage",

                   data=organization,hue="TARGET",palette=["b","r"])

plt.xticks(rotation=90)

plt.grid(True,alpha=.3)

ax.set_facecolor("k")

ax.set_title("Distribution in organization types for repayers and defaulters")

plt.show()
fig = plt.figure(figsize=(20,20))

plt.subplot(421)

sns.boxplot(data=application_data,x='TARGET',y='OBS_30_CNT_SOCIAL_CIRCLE',

            hue="TARGET", palette="Set3")

plt.title("Client's social surroundings with observable 30 DPD (days past due) def",color="b")

plt.subplot(422)

sns.boxplot(data=application_data,x='TARGET',y='DEF_30_CNT_SOCIAL_CIRCLE',

            hue="TARGET", palette="Set3")

plt.title("Client's social surroundings defaulted on 30 DPD (days past due)",color="b")

plt.subplot(423)

sns.boxplot(data=application_data,x='TARGET',y='OBS_60_CNT_SOCIAL_CIRCLE',

            hue="TARGET", palette="Set3")

plt.title("Client's social surroundings with observable 60 DPD (days past due) default",color="b")

plt.subplot(424)

sns.boxplot(data=application_data,x='TARGET',y='DEF_60_CNT_SOCIAL_CIRCLE',

            hue="TARGET", palette="Set3")

plt.title("Client's social surroundings defaulted on 60 DPD (days past due)",color="b")

fig.set_facecolor("ghostwhite")
plt.figure(figsize=(13,7))

plt.subplot(121)

ax = sns.violinplot(application_data["TARGET"],

                    application_data["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])

ax.set_facecolor("oldlace")

ax.set_title("days before application client changed phone -violin plot")

plt.subplot(122)

ax1 = sns.lvplot(application_data["TARGET"],

                 application_data["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])

ax1.set_facecolor("oldlace")

ax1.set_ylabel("")

ax1.set_title("days before application client changed phone -box plot")

plt.subplots_adjust(wspace = .2)
cols = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',

       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',

       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',

       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',

       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']



df_flag = application_data[cols+["TARGET"]]



length = len(cols)



df_flag["TARGET"] = df_flag["TARGET"].replace({1:"defaulter",0:"repayer"})



fig = plt.figure(figsize=(13,24))

fig.set_facecolor("lightgrey")

for i,j in itertools.zip_longest(cols,range(length)):

    plt.subplot(5,4,j+1)

    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","b"])

    plt.yticks(fontsize=5)

    plt.xlabel("")

    plt.title(i)

    ax.set_facecolor("k")
cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',

       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',

       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

application_data.groupby("TARGET")[cols].max().transpose().plot(kind="barh",

                                                                 figsize=(10,5),width=.8)

plt.title("Maximum enquries made by defaulters and repayers")

application_data.groupby("TARGET")[cols].mean().transpose().plot(kind="barh",

                                                                  figsize=(10,5),width=.8)

plt.title("average enquries made by defaulters and repayers")

application_data.groupby("TARGET")[cols].std().transpose().plot(kind="barh",

                                                                 figsize=(10,5),width=.8)

plt.title("standard deviation in enquries made by defaulters and repayers")

plt.show()
x = previous_application.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()

plt.figure(figsize=(13,7))

ax = sns.distplot(x["SK_ID_PREV"],color="orange")

plt.axvline(x["SK_ID_PREV"].mean(),linestyle="dashed",color="r",label="average")

plt.axvline(x["SK_ID_PREV"].std(),linestyle="dashed",color="b",label="standard deviation")

plt.axvline(x["SK_ID_PREV"].max(),linestyle="dashed",color="g",label="maximum")

plt.legend(loc="best")

plt.title("Current loan id having previous loan applications")

ax.set_facecolor("k")
cnts = previous_application["NAME_CONTRACT_TYPE"].value_counts()

import squarify

plt.figure(figsize=(8,6))

squarify.plot(cnts.values,label=cnts.keys(),value=cnts.values,linewidth=2,edgecolor="k",alpha=.8,color=sns.color_palette("Set1"))

plt.axis("off")

plt.title("Contaract types in previous applications")

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(211)

ax = sns.kdeplot(previous_application["AMT_APPLICATION"],color="b",linewidth=3)

ax = sns.kdeplot(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"],color="r",linewidth=3)

plt.axvline(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"].mean(),color="r",linestyle="dashed",label="AMT_APPLICATION_MEAN")

plt.axvline(previous_application["AMT_APPLICATION"].mean(),color="b",linestyle="dashed",label="AMT_APPLICATION_MEAN")

plt.legend(loc="best")

plt.title("Previous loan amounts applied and loan amounts credited.")

ax.set_facecolor("k")



plt.subplot(212)

diff = (previous_application["AMT_CREDIT"] - previous_application["AMT_APPLICATION"]).reset_index()

diff = diff[diff[0].notnull()]

ax1 = sns.kdeplot(diff[0],color="g",linewidth=3,label = "difference in amount requested by client and amount credited")

plt.axvline(diff[0].mean(),color="white",linestyle="dashed",label = "mean")

plt.title("difference in amount requested by client and amount credited")

ax1.legend(loc="best")

ax1.set_facecolor("k")
mn = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].mean().stack().reset_index()

tt = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].sum().stack().reset_index()

fig = plt.figure(figsize=(10,13))

fig.set_facecolor("ghostwhite")

plt.subplot(211)

ax = sns.barplot(0,"NAME_CONTRACT_TYPE",data=mn[:6],hue="level_1",palette="inferno")

ax.set_facecolor("k")

ax.set_xlabel("average amounts")

ax.set_title("Average amounts by contract types")



plt.subplot(212)

ax1 = sns.barplot(0,"NAME_CONTRACT_TYPE",data=tt[:6],hue="level_1",palette="magma")

ax1.set_facecolor("k")

ax1.set_xlabel("total amounts")

ax1.set_title("total amounts by contract types")

plt.subplots_adjust(hspace = .2)

plt.show()
plt.figure(figsize=(14,5))

plt.subplot(121)

previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].sum().plot(kind="bar")

plt.xticks(rotation=0)

plt.title("Total annuity amount by contract types in previous applications")

plt.subplot(122)

previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].mean().plot(kind="bar")

plt.title("average annuity amount by contract types in previous applications")

plt.xticks(rotation=0)

plt.show()
ax = pd.crosstab(previous_application["NAME_CONTRACT_TYPE"],previous_application["NAME_CONTRACT_STATUS"]).plot(kind="barh",figsize=(10,7),stacked=True)

plt.xticks(rotation =0)

plt.ylabel("count")

plt.title("Count of application status by application type")

ax.set_facecolor("k")
hr = pd.crosstab(previous_application["WEEKDAY_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).stack().reset_index()

plt.figure(figsize=(12,8))

ax = sns.pointplot(hr["WEEKDAY_APPR_PROCESS_START"],hr[0],hue=hr["NAME_CONTRACT_STATUS"],palette=["g","r","b","orange"],scale=1)

ax.set_facecolor("k")

ax.set_ylabel("count")

ax.set_title("Contract status by weekdays")

plt.grid(True,alpha=.2)



hr = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).stack().reset_index()

plt.figure(figsize=(12,8))

ax = sns.pointplot(hr["HOUR_APPR_PROCESS_START"],hr[0],hue=hr["NAME_CONTRACT_STATUS"],palette=["g","r","b","orange"],scale=1)

ax.set_facecolor("k")

ax.set_ylabel("count")

ax.set_title("Contract status by day hours.")

plt.grid(True,alpha=.2)
hr = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["WEEKDAY_APPR_PROCESS_START"]).stack().reset_index()

plt.figure(figsize=(12,8))

ax = sns.pointplot(hr["HOUR_APPR_PROCESS_START"],hr[0],hue=hr["WEEKDAY_APPR_PROCESS_START"],palette=["g","r","b","orange"],scale=1)

ax.set_facecolor("k")

ax.set_ylabel("count")

ax.set_title("Peak hours for week days")

plt.grid(True,alpha=.2)
previous_application[["NAME_CASH_LOAN_PURPOSE","NAME_CONTRACT_STATUS"]]

purpose = pd.crosstab(previous_application["NAME_CASH_LOAN_PURPOSE"],previous_application["NAME_CONTRACT_STATUS"])

purpose["a"] = (purpose["Approved"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])

purpose["c"] = (purpose["Canceled"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])

purpose["r"] = (purpose["Refused"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])

purpose["u"] = (purpose["Unused offer"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])

purpose_new = purpose[["a","c","r","u"]]

purpose_new = purpose_new.stack().reset_index()

purpose_new["NAME_CONTRACT_STATUS"] = purpose_new["NAME_CONTRACT_STATUS"].replace({"a":"accepted_percentage","c":"cancelled_percentage",

                                                               "r":"refused_percentage","u":"unused_percentage"})



lst = purpose_new["NAME_CONTRACT_STATUS"].unique().tolist()

length = len(lst)

cs = ["lime","orange","r","b"]



fig = plt.figure(figsize=(14,18))

fig.set_facecolor("lightgrey")

for i,j,k in itertools.zip_longest(lst,range(length),cs):

    plt.subplot(2,2,j+1)

    dat = purpose_new[purpose_new["NAME_CONTRACT_STATUS"] == i]

    ax = sns.barplot(0,"NAME_CASH_LOAN_PURPOSE",data=dat.sort_values(by=0,ascending=False),color=k)

    plt.ylabel("")

    plt.xlabel("percentage")

    plt.title(i+" by purpose")

    plt.subplots_adjust(wspace = .7)

    ax.set_facecolor("k")

plt.figure(figsize=(13,6))

sns.violinplot(y= previous_application["DAYS_DECISION"],

               x = previous_application["NAME_CONTRACT_STATUS"],palette=["r","g","b","y"])

plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"]["DAYS_DECISION"].mean(),

            color="r",linestyle="dashed",label="accepted_average")

plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]["DAYS_DECISION"].mean(),

            color="g",linestyle="dashed",label="refused_average")

plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Cancelled"]["DAYS_DECISION"].mean(),color="b",

            linestyle="dashed",label="cancelled_average")

plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Unused offer"]["DAYS_DECISION"].mean(),color="y",

            linestyle="dashed",label="un used_average")

plt.legend(loc="best")



plt.title("Contract status relative to decision made about previous application.")

plt.show()
plt.figure(figsize=(8,12))

plt.subplot(211)

rej = previous_application["CODE_REJECT_REASON"].value_counts().reset_index()

ax = sns.barplot("CODE_REJECT_REASON","index",data=rej[:6],palette="husl")

for i,j in enumerate(np.around((rej["CODE_REJECT_REASON"][:6].values*100/(rej["CODE_REJECT_REASON"][:6].sum())))):

    ax.text(.7,i,j,weight="bold")

plt.xlabel("Top as percentage & Bottom as Count")

plt.ylabel("CODE_REJECT_REASON")

plt.title("Reasons for application rejections")



plt.subplot(212)

pay = previous_application["NAME_PAYMENT_TYPE"].value_counts().reset_index()

ax1 = sns.barplot("NAME_PAYMENT_TYPE","index",data=pay,palette="husl")

for i,j in enumerate(np.around((pay["NAME_PAYMENT_TYPE"].values*100/(pay["NAME_PAYMENT_TYPE"].sum())))):

    ax1.text(.7,i,j,weight="bold")

plt.xlabel("pTop as percentage & Bottom as Count")

plt.ylabel("NAME_PAYMENT_TYPE")

plt.title("Clients payment methods")

plt.subplots_adjust(hspace = .3)
plt.figure(figsize=(20,20))

plt.subplot(121)

previous_application["NAME_TYPE_SUITE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=12,

                                                             colors = sns.color_palette("inferno"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("NAME_TYPE_SUITE")



plt.subplot(122)

previous_application["NAME_CLIENT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=12,

                                                             colors = sns.color_palette("inferno"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("NAME_CLIENT_TYPE")

plt.show()
goods = previous_application["NAME_GOODS_CATEGORY"].value_counts().reset_index()

goods["percentage"] = round(goods["NAME_GOODS_CATEGORY"]*100/goods["NAME_GOODS_CATEGORY"].sum(),2)

fig = plt.figure(figsize=(12,5))

ax = sns.pointplot("index","percentage",data=goods,color="yellow")

plt.xticks(rotation = 80)

plt.xlabel("NAME_GOODS_CATEGORY")

plt.ylabel("percentage")

plt.title("popular goods for applying loans")

ax.set_facecolor("k")

fig.set_facecolor('lightgrey')
plt.figure(figsize=(20,20))

plt.subplot(121)

previous_application["NAME_PORTFOLIO"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=12,

                                                             colors = sns.color_palette("prism",5),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},

                                                               shadow =True)

plt.title("previous applications portfolio")

plt.subplot(122)

previous_application["NAME_PRODUCT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=12,

                                                             colors = sns.color_palette("prism",3),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},

                                                                  shadow =True)

plt.title("previous applications product types")

plt.show()
app = pd.crosstab(previous_application["CHANNEL_TYPE"],previous_application["NAME_CONTRACT_STATUS"])

app1 = app

app1["approval_rate"] = app1["Approved"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])

app1["refused_rate"]  = app1["Refused"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])

app1["cacelled_rate"] = app1["Canceled"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])

app2 = app[["approval_rate","refused_rate","cacelled_rate"]]

ax = app2.plot(kind="barh",stacked=True,figsize=(10,7))

ax.set_facecolor("k")

ax.set_xlabel("percentage")

ax.set_title("approval,cancel and refusal rates by channel types")

plt.show()
fig = plt.figure(figsize=(13,5))

plt.subplot(121)

are = previous_application.groupby("SELLERPLACE_AREA")["AMT_CREDIT"].sum().reset_index()

are = are.sort_values(by ="AMT_CREDIT",ascending = False)

ax = sns.barplot(y= "AMT_CREDIT",x ="SELLERPLACE_AREA",data=are[:15],color="r")

ax.set_facecolor("k")

ax.set_title("Highest amount credited seller place areas")



plt.subplot(122)

sell = previous_application.groupby("NAME_SELLER_INDUSTRY")["AMT_CREDIT"].sum().reset_index().sort_values(by = "AMT_CREDIT",ascending = False)

ax1=sns.barplot(y = "AMT_CREDIT",x = "NAME_SELLER_INDUSTRY",data=sell,color="b")

ax1.set_facecolor("k")

ax1.set_title("Highest amount credited seller industrys")

plt.xticks(rotation=90)

plt.subplots_adjust(wspace = .5)

fig.set_facecolor("lightgrey")
plt.figure(figsize=(13,5))

ax = sns.countplot(previous_application["CNT_PAYMENT"],palette="Set1",order=previous_application["CNT_PAYMENT"].value_counts().index)

ax.set_facecolor("k")

plt.xticks(rotation = 90)

plt.title("popular terms of previous credit at application")

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(y = previous_application["PRODUCT_COMBINATION"],order=previous_application["PRODUCT_COMBINATION"].value_counts().index)

plt.title("Detailed product combination of the previous application -count")

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(121)

previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,

                                                             colors = sns.color_palette("prism",4),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("client requesting insurance")



plt.subplot(122)

previous_application["NAME_YIELD_GROUP"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,

                                                             colors = sns.color_palette("prism",4),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("interest rates")

plt.show()
cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']

plt.figure(figsize=(12,6))

sns.heatmap(previous_application[cols].describe()[1:].transpose(),

            annot=True,linewidth=2,linecolor="k",cmap=sns.color_palette("inferno"))

plt.show()
corrmat = application_data.corr() 

  

f, ax = plt.subplots(figsize =(8, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="rainbow") 

plt.show()
corrmat = previous_application.corr() 

  

f, ax = plt.subplots(figsize =(8, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="rainbow") 

plt.show()
corrmat = previous_application.corr() 

corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

corrdf = corrdf.unstack().reset_index()

corrdf.columns = ['Var1', 'Var2', 'Correlation']

corrdf.dropna(subset = ['Correlation'], inplace = True)

corrdf['Correlation'] = round(corrdf['Correlation'], 2)

corrdf['Correlation'] = abs(corrdf['Correlation'])

corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
df_repayer = application_data[application_data['TARGET'] == 0]

df_defaulter = application_data[application_data['TARGET'] == 1]
corrmat = df_repayer.corr() 

corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

corrdf = corrdf.unstack().reset_index()

corrdf.columns = ['Var1', 'Var2', 'Correlation']

corrdf.dropna(subset = ['Correlation'], inplace = True)

corrdf['Correlation'] = round(corrdf['Correlation'], 2)

corrdf['Correlation'] = abs(corrdf['Correlation'])

corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
corrmat = df_defaulter.corr() 

corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

corrdf = corrdf.unstack().reset_index()

corrdf.columns = ['Var1', 'Var2', 'Correlation']

corrdf.dropna(subset = ['Correlation'], inplace = True)

corrdf['Correlation'] = round(corrdf['Correlation'], 2)

corrdf['Correlation'] = abs(corrdf['Correlation'])

corrdf.sort_values(by = 'Correlation', ascending = False).head(10)
mergeddf =  pd.merge(application_data,previous_application,on='SK_ID_CURR')

mergeddf.head()
y = mergeddf.groupby('SK_ID_CURR').size()

dfA = mergeddf.groupby('SK_ID_CURR').agg({'TARGET': np.sum})

dfA['count'] = y

display(dfA.head(10))
dfA.sort_values(by = 'count',ascending=False).head(10)
df_repayer = dfA[dfA['TARGET'] == 0]

df_defaulter = dfA[dfA['TARGET'] == 1]
df_repayer.sort_values(by = 'count',ascending=False).head(10)
df_defaulter.sort_values(by = 'count',ascending=False).head(10)
mergeddf.isnull().sum()
round(100*(mergeddf.isnull().sum()/len(mergeddf.index)), 2)
mergeddf.head()
#dropping SK_ID_CURR since it all unique values



mergeddf.drop(['SK_ID_CURR'], 1, inplace = True)

mergeddf.head()
round(100*(mergeddf.isnull().sum()/len(mergeddf.index)), 2)
enq_cs =['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',

       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',

       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']

for i in enq_cs:

    mergeddf[i] = mergeddf[i].fillna(0)
amt_cs = ["AMT_ANNUITY_y","AMT_GOODS_PRICE_y"]

for i in amt_cs:

    mergeddf[i] = mergeddf[i].fillna(mergeddf[i].mean())

    
cols = ["DAYS_FIRST_DRAWING","DAYS_FIRST_DUE","DAYS_LAST_DUE_1ST_VERSION",

        "DAYS_LAST_DUE","DAYS_TERMINATION",'CNT_PAYMENT']

for i in cols :

    mergeddf[i]  = mergeddf[i].fillna(mergeddf[i].median())
cols = ["NAME_TYPE_SUITE_y","NFLAG_INSURED_ON_APPROVAL"]

for i in cols :

    mergeddf[i]  = mergeddf[i].fillna(mergeddf[i].mode()[0])
# Rest missing values are under 1.5% so we can drop these rows.

mergeddf.dropna(inplace = True)
round(100*(mergeddf.isnull().sum()/len(mergeddf.index)), 2)
mergeddf.isnull().sum()
mergeddf.head()
# List of variables to map



varlist =  ['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_LAST_APPL_PER_CONTRACT']



# Defining the map function

def binary_map(x):

    return x.map({'Y': 1, "N": 0})



# Applying the function to the housing list

mergeddf[varlist] = mergeddf[varlist].apply(binary_map)

mergeddf.head()
#dropping SK_ID_PREV since non required technical field 



mergeddf.drop(['SK_ID_PREV'], 1, inplace = True)

mergeddf.head()
mergeddf[['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL',

 'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',

 'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT',

 'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',

 'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

 'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2',

 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',

 'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',

 'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

 'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',

 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',

 'NFLAG_INSURED_ON_APPROVAL']]= mergeddf[['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL',

                                        'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',

                                        'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT',

                                        'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',

                                        'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

                                        'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2',

                                        'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',

                                        'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',

                                        'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

                                        'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',

                                        'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',

                                        'NFLAG_INSURED_ON_APPROVAL']].astype('category')
obj_dtypes = [i for i in mergeddf.select_dtypes(include=np.object).columns if i not in ["type"] ]

num_dtypes = [i for i in mergeddf.select_dtypes(include = np.number).columns if i not in [ 'TARGET']]
num_dtypes
obj_dtypes
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(mergeddf[['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL',

                                  'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',

                                  'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT',

                                  'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',

                                  'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

                                  'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2',

                                  'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',

                                  'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',

                                  'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

                                  'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',

                                  'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',

                                  'NFLAG_INSURED_ON_APPROVAL','NAME_CONTRACT_TYPE_x','CODE_GENDER',

                                  'NAME_TYPE_SUITE_x','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',

                                  'NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START_x','ORGANIZATION_TYPE','NAME_CONTRACT_TYPE_y',

                                  'WEEKDAY_APPR_PROCESS_START_y','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',

                                  'CODE_REJECT_REASON','NAME_TYPE_SUITE_y','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY',

                                  'NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY',

                                  'NAME_YIELD_GROUP','PRODUCT_COMBINATION']], drop_first=True)

dummy1.head()
# Adding the results to the master dataframe

mergeddf = pd.concat([mergeddf, dummy1], axis=1)

mergeddf.head()
mergeddf = mergeddf.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL',

                          'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',

                          'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT',

                          'REGION_RATING_CLIENT_W_CITY','REG_REGION_NOT_LIVE_REGION',

                          'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

                          'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2',

                          'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',

                          'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',

                          'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

                          'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',

                          'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',

                          'NFLAG_INSURED_ON_APPROVAL','NAME_CONTRACT_TYPE_x','CODE_GENDER',

                          'NAME_TYPE_SUITE_x','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',

                          'NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START_x','ORGANIZATION_TYPE','NAME_CONTRACT_TYPE_y',

                          'WEEKDAY_APPR_PROCESS_START_y','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',

                          'CODE_REJECT_REASON','NAME_TYPE_SUITE_y','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY',

                          'NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY',

                          'NAME_YIELD_GROUP','PRODUCT_COMBINATION'], axis = 1)

mergeddf.head()
mergeddf.shape
mergeddfs=mergeddf.sample(n = 7000) 
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = mergeddfs.drop(['TARGET'], axis=1)
X.head()
X.shape
# Putting response variable to y

y = mergeddfs['TARGET']
y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=70)
X_train.head()
X_train.shape
X_test.head()
X_test.shape
y_train.head()
y_train.shape
y_test.head()
y_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT_x','AMT_ANNUITY_x',

         'AMT_GOODS_PRICE_x','REGION_POPULATION_RELATIVE','DAYS_BIRTH',

         'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','CNT_FAM_MEMBERS',

         'HOUR_APPR_PROCESS_START_x','LIVE_CITY_NOT_WORK_CITY','OBS_30_CNT_SOCIAL_CIRCLE',

         'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',

         'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',

         'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT',

         'AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY_y','AMT_APPLICATION','AMT_CREDIT_y',

         'AMT_GOODS_PRICE_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT',

         'NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','SELLERPLACE_AREA','CNT_PAYMENT',

         'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE',

         'DAYS_TERMINATION']] = scaler.fit_transform(X_train[['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT_x',

                                                               'AMT_ANNUITY_x','AMT_GOODS_PRICE_x','REGION_POPULATION_RELATIVE',

                                                               'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',

                                                               'CNT_FAM_MEMBERS','HOUR_APPR_PROCESS_START_x','LIVE_CITY_NOT_WORK_CITY',

                                                               'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',

                                                               'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR',

                                                               'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',

                                                               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY_y','AMT_APPLICATION',

                                                               'AMT_CREDIT_y','AMT_GOODS_PRICE_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT',

                                                               'NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','SELLERPLACE_AREA','CNT_PAYMENT','DAYS_FIRST_DRAWING',

                                                               'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']])



X_train.head()
X_test[['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT_x','AMT_ANNUITY_x',

         'AMT_GOODS_PRICE_x','REGION_POPULATION_RELATIVE','DAYS_BIRTH',

         'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','CNT_FAM_MEMBERS',

         'HOUR_APPR_PROCESS_START_x','LIVE_CITY_NOT_WORK_CITY','OBS_30_CNT_SOCIAL_CIRCLE',

         'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',

         'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',

         'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT',

         'AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY_y','AMT_APPLICATION','AMT_CREDIT_y',

         'AMT_GOODS_PRICE_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT',

         'NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','SELLERPLACE_AREA','CNT_PAYMENT',

         'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE',

         'DAYS_TERMINATION']] = scaler.transform(X_test[['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT_x',

                                                               'AMT_ANNUITY_x','AMT_GOODS_PRICE_x','REGION_POPULATION_RELATIVE',

                                                               'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',

                                                               'CNT_FAM_MEMBERS','HOUR_APPR_PROCESS_START_x','LIVE_CITY_NOT_WORK_CITY',

                                                               'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',

                                                               'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR',

                                                               'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',

                                                               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY_y','AMT_APPLICATION',

                                                               'AMT_CREDIT_y','AMT_GOODS_PRICE_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT',

                                                               'NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','SELLERPLACE_AREA','CNT_PAYMENT','DAYS_FIRST_DRAWING',

                                                               'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']])



X_test.head()
# Checking the Converted Rate

Target = round((sum(mergeddf['TARGET'])/len(mergeddf['TARGET'].index))*100,2)

print("We have almost {} %  Converted rate after successful data manipulation".format(Target))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

model = KNeighborsClassifier()
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.tail()
features_to_remove = vif.loc[vif['VIF'] >= 4.99,'Features'].values

features_to_remove = list(features_to_remove)

print(features_to_remove)
X_train = X_train.drop(columns=features_to_remove, axis = 1)

X_train.head()
X_test = X_test.drop(columns=features_to_remove, axis = 1)

X_test.head()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features_to_remove = vif.loc[vif['VIF'] >= 4.99,'Features'].values

features_to_remove = list(features_to_remove)

print(features_to_remove)
X_train = X_train.drop(columns=features_to_remove, axis = 1)

X_train.head()
X_test = X_test.drop(columns=features_to_remove, axis = 1)

X_test.head()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train, predict_train )

print(confusion)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our model

trainsensitivity= TP / float(TP+FN)

trainsensitivity
# Let us calculate specificity

trainspecificity= TN / float(TN+FP)

trainspecificity
# Calculate false postive rate - predicting Defaulted when customer does not have Defaulted

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print(TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train,predict_train)
#Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score

precision_score(y_train,predict_train)
recall_score(y_train,predict_train)  
# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data\n\n',predict_test)
confusion2 = metrics.confusion_matrix(y_test, predict_test )

print(confusion2)
# Let's check the overall accuracy.

testaccuracy= accuracy_score(y_test,predict_test)

testaccuracy
# Let's see the sensitivity of our lmodel

testsensitivity=TP / float(TP+FN)

testsensitivity
# Let us calculate specificity

testspecificity= TN / float(TN+FP)

testspecificity
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))

print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))

print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))

print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))