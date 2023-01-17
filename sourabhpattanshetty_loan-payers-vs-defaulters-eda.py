import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt,itertools
curr_app=pd.read_csv("../input/bank-loans-dataset/application_data.csv")

prev_app=pd.read_csv("../input/bank-loans-dataset/previous_application.csv")
print ("Current Application     :",curr_app.shape)

print ("Previous Application    :",prev_app.shape)
pd.set_option('display.max_columns', 100)

display("Current Application")

display(curr_app.head(3))

display("Previous Application")

display(prev_app.head(3))
fig = plt.figure(figsize=(12,6))

fig.set_facecolor("lightgrey")

missing=pd.DataFrame((curr_app.isnull().sum())*100/curr_app.shape[0]).reset_index()

ax = sns.pointplot("index",0,data=missing)

plt.xticks(rotation =90,fontsize =5)

plt.title("Percentage of Missing values in current application")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")
curr_app[missing['index'][missing[0]>40]].shape
curr_app=curr_app[missing['index'][missing[0]<40]]
curr_app.shape
missing[(missing[0]>1) & (missing[0]<40)]
#Looking at credit bureau data which has missing data just above 13%:



plt.figure(figsize=(15,10))

plt.subplot(231)

curr_app["AMT_REQ_CREDIT_BUREAU_HOUR"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_HOUR")





plt.subplot(232)

curr_app["AMT_REQ_CREDIT_BUREAU_DAY"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_DAY")



plt.subplot(233)

curr_app["AMT_REQ_CREDIT_BUREAU_WEEK"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_WEEK")



plt.subplot(234)

curr_app["AMT_REQ_CREDIT_BUREAU_MON"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_MON")



plt.subplot(235)

curr_app["AMT_REQ_CREDIT_BUREAU_QRT"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_QRT")



plt.subplot(236)

curr_app["AMT_REQ_CREDIT_BUREAU_YEAR"].value_counts().plot(kind="barh")

plt.title("AMT_REQ_CREDIT_BUREAU_YEAR")





plt.show()

#In the above graphs, If we see the values distributions, 5 out of 6 columns can have null values filled with 

#respective Modes of the distribution which is 0.0. Lets fill those in!!





for i in curr_app.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_QRT'].columns:

    curr_app[i].fillna(0,inplace=True)
#Lets check out the missing values across the data again!!



fig = plt.figure(figsize=(15,10))

fig.set_facecolor("lightgrey")

missing=pd.DataFrame((curr_app.isnull().sum())*100/curr_app.shape[0]).reset_index()

ax = sns.pointplot("index",0,data=missing)

plt.xticks(rotation =90,fontsize =5)

plt.title("Percentage of Missing values in current application")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

ax.set_facecolor("k")
curr_app.info(verbose=True)
curr_app.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
curr_app.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
curr_app[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']].head(5)
for i in ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:

    curr_app[i]=abs(curr_app[i])



curr_app[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']].head(5)
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app[curr_app['AMT_INCOME_TOTAL']<np.percentile(curr_app['AMT_INCOME_TOTAL'],99)]['AMT_INCOME_TOTAL'])

plt.subplot(122)

sns.distplot(curr_app[curr_app['AMT_INCOME_TOTAL']<np.percentile(curr_app['AMT_INCOME_TOTAL'],99)]['AMT_INCOME_TOTAL'])

plt.show()
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app['AMT_INCOME_TOTAL'])

plt.subplot(122)

sns.distplot(curr_app['AMT_INCOME_TOTAL'])

plt.show()
curr_app[curr_app['AMT_INCOME_TOTAL']>np.percentile(curr_app['AMT_INCOME_TOTAL'],99)].shape
np.percentile(curr_app['AMT_INCOME_TOTAL'],99)
curr_app[curr_app['AMT_CREDIT'] > np.percentile(curr_app['AMT_CREDIT'],99)].shape
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app[curr_app['AMT_ANNUITY']<58000]['AMT_ANNUITY'])

plt.subplot(122)

sns.distplot(curr_app[curr_app['AMT_ANNUITY']<58000]['AMT_ANNUITY'])

plt.show()
curr_app[curr_app['AMT_ANNUITY'] > 58000].shape
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app[curr_app['AMT_GOODS_PRICE']<1300000]['AMT_GOODS_PRICE'])

plt.subplot(122)

sns.distplot(curr_app[curr_app['AMT_GOODS_PRICE']<1300000]['AMT_GOODS_PRICE'])

plt.show()
curr_app[curr_app['AMT_GOODS_PRICE']>1300000].shape
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app[curr_app['DAYS_EMPLOYED']<350000]['DAYS_EMPLOYED'])

plt.subplot(122)

sns.distplot(curr_app[curr_app['DAYS_EMPLOYED']<350000]['DAYS_EMPLOYED'])

plt.show()
curr_app[curr_app['DAYS_EMPLOYED']>365000].shape
plt.figure(figsize=(15,7))



plt.subplot(121)

sns.boxplot(curr_app['DAYS_EMPLOYED'])

plt.subplot(122)

sns.distplot(curr_app['DAYS_EMPLOYED'])

plt.show()
#Lets check out Flag document columns according to the distribution percentages here 



plt.figure(figsize=(15,10))



for i in range(2,11):

    plt.subplot(int("33"+str(i-1)))

    (100*curr_app["FLAG_DOCUMENT_"+str(i)].value_counts(normalize=True)).plot(kind="barh")

    plt.title("FLAG_DOCUMENT_"+str(i))
curr_app1=curr_app[curr_app['TARGET']==1]

curr_app0=curr_app[curr_app['TARGET']==0]
sns.heatmap(curr_app1[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED']].corr(),annot=True)
corr=curr_app.corr()

corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))

corr_df=corr.abs().unstack().reset_index()

corr_df.columns=['var1','var2','correlation']

corr_df.correlation=round(corr_df.correlation*100,2)

corr_df.dropna(subset=['correlation'],inplace=True)

corr_df.sort_values('correlation',ascending=False).head(10)
##Correlation of Repayers:



corr=curr_app0.corr()

corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))

corr_df=corr.abs().unstack().reset_index()

corr_df.columns=['var1','var2','correlation']

corr_df.correlation=round(corr_df.correlation*100,2)

corr_df.dropna(subset=['correlation'],inplace=True)

corr_df.sort_values('correlation',ascending=False).head(10)
##Correlation of Defaulters:



corr=curr_app1.corr()

corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))

corr_df=corr.abs().unstack().reset_index()

corr_df.columns=['var1','var2','correlation']

corr_df.correlation=round(corr_df.correlation*100,2)

corr_df.dropna(subset=['correlation'],inplace=True)

corr_df.sort_values('correlation',ascending=False).head(10)
plt.figure(figsize=(14,7))

plt.subplot(121)

curr_app["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=["repayer","defaulter"],

                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)

plt.title("Distribution of target variable")



plt.subplot(122)

ax = curr_app["TARGET"].value_counts().plot(kind="barh")



for i,j in enumerate(curr_app["TARGET"].value_counts().values):

    ax.text(.7,i,j,weight = "bold",fontsize=20)



plt.title("Count of target variable")

plt.show()



#TARGET :Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in sample, 0 - all other cases)

#8% out of total client population have difficulties in repaying loans.

cols = [ 'AMT_INCOME_TOTAL','AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_CREDIT','AMT_ANNUITY', 'AMT_ANNUITY','AMT_GOODS_PRICE','AMT_GOODS_PRICE']

length = len(cols)

cs = ["r","b","g","k"]



ax = plt.figure(figsize=(15,30))

ax.set_facecolor("lightgrey")

count=1

for i in cols:

    plt.subplot(4,2,count)

    if count%2==1:

        sns.distplot(curr_app1[curr_app1[i].notnull()][i],color='r')

        plt.axvline(curr_app1[i].mean(),label = "mean",linestyle="dashed",color='k')

        plt.title(i+ "for customers who have defaulted")

    else:

        sns.distplot(curr_app0[curr_app0[i].notnull()][i],color='b')

        plt.axvline(curr_app0[i].mean(),label = "mean",linestyle="dashed",color='k')

        plt.title(i + "for customers who have not defaulted")

    count+=1

    plt.legend(loc="best")

    plt.subplots_adjust(hspace = .2)



    
cols = [ 'AMT_INCOME_TOTAL']



df = curr_app.groupby("TARGET")[cols].describe().transpose().reset_index()

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



plt.figure(figsize=(10,10))



for i,j in itertools.zip_longest(stat,range(length)):

    plt.subplot(2,2,j+1)

    fig = sns.barplot(df_new[df_new["statistic"] == i]["amount_type"],df_new[df_new["statistic"] == i]["amount"],

                hue=df_new[df_new["statistic"] == i]["type"],palette=["g","r"])

    plt.title(i + "--Defaulters vs Non defaulters")

   

    fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(13,15))



plt.subplot(221)

sns.boxplot(abs(curr_app0["REGION_POPULATION_RELATIVE"]),color ="b")

plt.title("REGION_POPULATION_RELATIVE Distribution of repayers")



plt.subplot(222)

sns.boxplot(abs(curr_app1["REGION_POPULATION_RELATIVE"]),color="r")

plt.title("REGION_POPULATION_RELATIVE Distribution of defaulters")
fig = plt.figure(figsize=(13,13))



plt.subplot(221)

sns.distplot(curr_app0["DAYS_BIRTH"],color="b")

plt.title("Age Distribution of repayers")



plt.subplot(222)

sns.distplot(curr_app1["DAYS_BIRTH"],color="r")

plt.title("Age Distribution of defaulters")
cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',

       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',

       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

curr_app.groupby("TARGET")[cols].max().transpose().plot(kind="barh",

                                                                 figsize=(10,5),width=.8)

plt.title("Maximum enquries made by defaulters and repayers")

curr_app.groupby("TARGET")[cols].mean().transpose().plot(kind="barh",

                                                                  figsize=(10,5),width=.8)

plt.title("average enquries made by defaulters and repayers")

curr_app.groupby("TARGET")[cols].std().transpose().plot(kind="barh",

                                                                 figsize=(10,5),width=.8)

plt.title("standard deviation in enquries made by defaulters and repayers")

plt.show()

d_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

d_length = len(d_cols)



fig = plt.figure(figsize=(16,4))

for i,j in itertools.zip_longest(d_cols,range(d_length)):

    plt.subplot(1,4,j+1)

    curr_app1[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,

                                        wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)

    circ = plt.Circle((0,0),.7,color="white")

    plt.gca().add_artist(circ)

    plt.ylabel("")

    plt.title(i+"-Defaulter")





fig = plt.figure(figsize=(16,4))

for i,j in itertools.zip_longest(d_cols,range(d_length)):

    plt.subplot(1,4,j+1)

    curr_app0[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,

                                           wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)

    circ = plt.Circle((0,0),.7,color="white")

    plt.gca().add_artist(circ)

    plt.ylabel("")

    plt.title(i+"-Repayer")
plt.figure(figsize=(15,10))

plt.subplot(121)

curr_app0["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",colors = sns.color_palette("Set1"),

wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("Distribution of Education type for Repayers",color="b")



plt.subplot(122)

curr_app1["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",

                                                                                                 colors = sns.color_palette("Set1"),

                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

circ = plt.Circle((0,0),.7,color="white")

plt.gca().add_artist(circ)

plt.title("Distribution of Education type for Defaulters",color="b")

plt.ylabel("")

plt.show()
fig = plt.figure(figsize=(7,6))

occ = curr_app0["OCCUPATION_TYPE"].value_counts().reset_index()

occ = occ.sort_values(by = "index",ascending=True)

occ1 = curr_app1["OCCUPATION_TYPE"].value_counts().reset_index()

occ1 = occ1.sort_values(by = "index",ascending=True)

occ["percentage"]  = (occ["OCCUPATION_TYPE"]*100/occ["OCCUPATION_TYPE"].sum())

occ1["percentage"] = (occ1["OCCUPATION_TYPE"]*100/occ1["OCCUPATION_TYPE"].sum())

occ["type"]        = "Repayers"

occ1["type"]       = "defaulters"

occupation = pd.concat([occ,occ1],axis=0)



ax = sns.barplot("index","percentage",data=occupation,hue="type",palette=["b","r"])

plt.xticks(rotation = 70)

plt.xlabel("occupation")

ax.set_facecolor("k")

fig.set_facecolor("ghostwhite")

plt.title("Occupation percentage in data with respect to repayment status")

plt.show()
org = curr_app.groupby("TARGET").agg({"ORGANIZATION_TYPE":"value_counts"})

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
fig = plt.figure(figsize=(8,6))

plt.scatter(curr_app0['AMT_ANNUITY'],curr_app0['AMT_CREDIT'],s=35,

            color="b",alpha=.5,label="REPAYER",linewidth=.5,edgecolor="k")

plt.scatter(curr_app1['AMT_ANNUITY'],curr_app1['AMT_CREDIT'],s=35,

            color="r",alpha=.2,label="DEFAULTER",linewidth=.5,edgecolor="k")

plt.legend(loc="best",prop={"size":15})

plt.xlabel("AMT_ANNUITY")

plt.ylabel("AMT_CREDIT")

plt.title("Scatter plot between credit amount and annuity amount")

plt.show()
fig = plt.figure(figsize=(7,7))

amt = curr_app[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',

                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]

amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]

sns.pairplot(amt,hue="TARGET",palette=["b","r"])

plt.show()
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']







plt.figure(figsize=(15,5))

plt.subplot(121)

df1 = curr_app0.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()



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

ax = sns.barplot("amt_type","amount",data=df_gen,hue="gender",palette="Set1")

plt.title("Average Income,credit,annuity & goods_price by gender in Repayers")



plt.subplot(122)

df1 = curr_app1.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()



df_f   = df1[["index","F"]]

df_f   = df_f.rename(columns={'index':"amt_type", 'F':"amount"})

df_f["gender"] = "FEMALE"

df_m   = df1[["index","M"]]

df_m   = df_m.rename(columns={'index':"amt_type", 'M':"amount"})

df_m["gender"] = "MALE"





df_gen = pd.concat([df_m,df_f,df_xna],axis=0)

ax = sns.barplot("amt_type","amount",data=df_gen,hue="gender",palette="Set1")

plt.title("Average Income,credit,annuity & goods_price by gender In Defaulters")

plt.show() 
plt.figure(figsize=(15,10))

plt.subplot(211)

edu = curr_app1.groupby(['NAME_EDUCATION_TYPE','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index()

ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL',data=edu,hue='NAME_EDUCATION_TYPE')

ax.set_facecolor("k")

plt.title(" Average Earnings by different professions and education types for defaulters ")



plt.subplot(212)

edu = curr_app0.groupby(['NAME_EDUCATION_TYPE','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index()

ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL',data=edu,hue='NAME_EDUCATION_TYPE')

ax.set_facecolor("k")

plt.title(" Average Earnings by different professions and education types for repayers")



plt.show()
plt.figure(figsize=(12,7))

plt.subplot(121)

sns.countplot(y=curr_app0["NAME_INCOME_TYPE"],

              hue=curr_app0["CODE_GENDER"],

              palette="Set2",

              order=curr_app0["NAME_INCOME_TYPE"].value_counts().index[:4])

plt.title("Distribution of client income type by gender for repayers")



plt.subplot(122)

sns.countplot(y=curr_app1["NAME_INCOME_TYPE"],

              hue=curr_app1["CODE_GENDER"],

              palette="Set2",

              order=curr_app1["NAME_INCOME_TYPE"].value_counts().index[:4])

plt.ylabel("")

plt.title("Distribution of client income  type by gender for defaulters")

plt.subplots_adjust(wspace = .4)
plt.figure(figsize=(12,7))

plt.subplot(121)

sns.countplot(y=curr_app0["NAME_TYPE_SUITE"],

              hue=curr_app0["CODE_GENDER"],palette="Set2",

              order=curr_app0["NAME_TYPE_SUITE"].value_counts().index[:5])

plt.title("Distribution of NAME_TYPE_SUITE by gender for repayers")



plt.subplot(122)

sns.countplot(y=curr_app1["NAME_TYPE_SUITE"],

              hue=curr_app1["CODE_GENDER"],palette="Set2",

              order=curr_app1["NAME_TYPE_SUITE"].value_counts().index[:5])

plt.ylabel("")

plt.title("Distribution of NAME_TYPE_SUITE by gender for defaulters")

plt.subplots_adjust(wspace = .4)
curr_app.shape
prev_app.shape
all_applications=pd.merge(curr_app,prev_app,how='inner',on='SK_ID_CURR')
all_applications.shape
all_applications.head(3)
plt.figure(figsize=(12,7))



plt.subplot(121)

sns.countplot(all_applications[all_applications['TARGET']==1].NAME_CONTRACT_STATUS)

plt.xlabel("Contract Status")

plt.ylabel("Count of Contract Status")

plt.title("Distribution of Contract Status for defaulters")



plt.subplot(122)

sns.countplot(all_applications[all_applications['TARGET']==0].NAME_CONTRACT_STATUS)

plt.xlabel("Contract Status")

plt.ylabel("Count of Contract Status")

plt.title("Distribution of Contract Status for Repayers")

plt.show()
def plot_3charts(var, label_rotation,horizontal_layout):

    if(horizontal_layout):

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10,5))

    else:

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10,5))

    

    s1=sns.countplot(ax=ax1,x=refused[var], data=refused, order= refused[var].value_counts().index,)

    ax1.set_title("Refused", fontsize=10)

    ax1.set_xlabel('%s' %var)

    ax1.set_ylabel("Count of Loans")

    if(label_rotation):

        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)

    

    s2=sns.countplot(ax=ax2,x=approved[var], data=approved, order= approved[var].value_counts().index,)

    if(label_rotation):

        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)

    ax2.set_xlabel('%s' %var)

    ax2.set_ylabel("Count of Loans")

    ax2.set_title("Approved", fontsize=10)

    

    

    s3=sns.countplot(ax=ax3,x=canceled[var], data=canceled, order= canceled[var].value_counts().index,)

    ax3.set_title("Canceled", fontsize=10)

    ax3.set_xlabel('%s' %var)

    ax3.set_ylabel("Count of Loans")

    if(label_rotation):

        s3.set_xticklabels(s3.get_xticklabels(),rotation=90)

    plt.show()
approved=all_applications[all_applications.NAME_CONTRACT_STATUS=='Approved']

refused=all_applications[all_applications.NAME_CONTRACT_STATUS=='Refused']

canceled=all_applications[all_applications.NAME_CONTRACT_STATUS=='Canceled']

unused=all_applications[all_applications.NAME_CONTRACT_STATUS=='Unused Offer']
plot_3charts('PRODUCT_COMBINATION', label_rotation=True,horizontal_layout=True)
prev_app
plt.figure(figsize=(15,10))

plt.subplot(211)

edu = all_applications[all_applications['TARGET']==1].groupby(['NAME_CONTRACT_TYPE_x','NAME_CONTRACT_STATUS'])['AMT_INCOME_TOTAL'].mean().reset_index()

ax = sns.barplot('NAME_CONTRACT_STATUS','AMT_INCOME_TOTAL',data=edu,hue='NAME_CONTRACT_TYPE_x')

ax.set_facecolor("k")

plt.title(" Average Earnings by different contract types and statuses for defaulters ")



plt.subplot(212)

edu = all_applications[all_applications['TARGET']==0].groupby(['NAME_CONTRACT_TYPE_x','NAME_CONTRACT_STATUS'])['AMT_INCOME_TOTAL'].mean().reset_index()

ax = sns.barplot('NAME_CONTRACT_STATUS','AMT_INCOME_TOTAL',data=edu,hue='NAME_CONTRACT_TYPE_x')

ax.set_facecolor("k")

plt.title(" Average Earnings by different contract types and statuses for repayers")



plt.show()