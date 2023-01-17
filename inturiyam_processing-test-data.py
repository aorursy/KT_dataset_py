import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
#Read data from CSV input
df_test = pd.read_csv("../input/titanic/test.csv")
print(df_test.dtypes)
df_test.info()
df_test.describe()
df_test.isna().sum()
#In this step, we replace missing values of Age with their average values
#df_test["Age"].describe()
#avg = np.average(df_test["Age"].fillna(value=0))
#print(avg)
#df_test["Age"].fillna(value = avg, inplace = True)
#df_test["Age"].describe()
#df_test.describe()
#As in the Training data representation, we will try to infer the missing values of Age based on the individual's title.

df_test["title"]=df_test["Name"].str.lower().str.extract('([a-z]*\.)', expand=True)
#df_test["title"].head()
#Passengers in each title group whose age is missing
df_test[df_test["Age"].isnull()].groupby(by = ["title"])["PassengerId"].count() 

avg_master = df_test[((df_test["title"]=="master.") & (df_test["Age"].isnull()==False))]["Age"].median()  
avg_miss = df_test[((df_test["title"]=="miss.") & (df_test["Age"].isnull()==False))]["Age"].median()  
avg_mr = df_test[((df_test["title"]=="mr.") & (df_test["Age"].isnull()==False))]["Age"].median()  
avg_mrs = df_test[((df_test["title"]=="mrs.") & (df_test["Age"].isnull()==False))]["Age"].median()  

#We will now replace the missing age values in each group with the corresponding average values 
# Refer - https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
df_test.loc[((df_test["title"]=="ms.") & (df_test["Age"].isnull()==True)).tolist(),'Age']=avg_miss #there is only 1 ms. in the whole test set. 
df_test.loc[((df_test["title"]=="master.") & (df_test["Age"].isnull()==True)).tolist(),'Age']=avg_master
df_test.loc[((df_test["title"]=="miss.") & (df_test["Age"].isnull()==True)).tolist(),'Age']=avg_miss
df_test.loc[((df_test["title"]=="mr.") & (df_test["Age"].isnull()==True)).tolist(),'Age']=avg_mr
df_test.loc[((df_test["title"]=="mrs.") & (df_test["Age"].isnull()==True)).tolist(),'Age']=avg_mrs

df_test.describe()
#We will now scale the Age column and add it as a new column to the dataframe. 
#For this we need to use the same mean and Std.Dev. as the Training Set. I am copying the values from my other notebook here manually.  
age_mean = 29.390202020202018
age_std = 13.265321985344817
df_test["age_norm"]=((df_test["Age"]-age_mean)/age_std)
df_test["age_norm"].hist()

df_test["is_male"] = pd.get_dummies(df_test["Sex"], drop_first=True) #we use drop_first to avoid creating another correlated column is_female
df_test.info()
#We will bin the passengers into few age groups just to see if children and older passengers had any higher survival probability
bins = [0,15,25,50,100]
df_test["age_group"]=pd.cut(df_test["Age"],bins)
#print(pd.get_dummies(df_train["age_group"]))
df_test[["age15","age25","age50","age100"]]=pd.get_dummies(df_test["age_group"], dtype="uint8")
#print(df_test["age15"])
namelen = []
for i in range(len(df_test["Name"])):
    namelen.append(len(df_test["Name"][i]))
df_test["len_name"]=namelen
#df_train.hist("len_name", by=["Survived", "Pclass"] , bins=10,layout=[4,3], figsize = [15,15])
df_test["len_name"].describe()
#df_train[df_train["len_name"] >= 30]

len_name_avg = 26.9652076318743
len_name_std = 9.28160688314506

df_test["norm_len_name"]=(df_test["len_name"]-len_name_avg)/len_name_std
df_test["norm_len_name"].hist()

#df_train.pivot(index="PassengerId",columns = "title", values = "Survived")
df_test.groupby(["title"])["PassengerId"].count()
#df_train.groupby(["title"])["Survived"].sum()/df_train.groupby(["title"])["PassengerId"].count()
#Approach 1
lookfor = np.array(['mrs.','sir.','countess.', 'lady.', 'master.', 'miss.', 'mlle.', 'mme.','mrs.','ms.', 'sir.'])
#s = pd.Series(lookfor)
df_test["high_prob_group"]=df_test["title"].isin(lookfor).astype('uint8')
df_test["high_prob_group"].sum()
#Approach 2
#use of x.astype('uint8') helps convert the Boolean output of isin() to an integer (0,1) representation 
df_test["title_ms"] = df_test["title"].isin(["miss.","ms."]).astype('uint8')
df_test["title_mrs"] = df_test["title"].isin(["mrs.","mme.","mlle."]).astype('uint8')
df_test["title_mr"] = df_test["title"].isin(["mr."]).astype('uint8')
df_test["title_others"]=df_test["title"].isin(['countess.', 'lady.', 'master.', 'dr.', 'don.','jonkheer.','rev.','major.','sir.','col.','capt.']).astype('uint8')
median_fare=df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S')]['Fare'].median()
print(median_fare)
#We start with a boxplot to figure out the range of Fare values
df_test.boxplot("Fare", by=["Embarked","Pclass"], figsize = [8,8])
df_test[df_test["Fare"].isna()] 
median_fare= df_test[(df_test["Embarked"]=="S") & (df_test["Pclass"]==3)]["Fare"].median()
df_test["Fare"].fillna(value = median_fare, inplace = True)
df_test["Fare"].describe()
#df_test.describe()
#Normalize fare values
fare_mean = 32.204208
fare_std = 49.693429
df_test["norm_fare"]= (df_test["Fare"]-fare_mean)/fare_std
df_test.hist("Fare", by=["Embarked", "Pclass"],layout=[4,3], figsize = [15,15], bins=10)
df_test[df_test["Embarked"].isna()] 
df_test["Embarked"].fillna(value = "C", inplace = True)
#df_test.hist("Fare", by=["Embarked", "Pclass"],layout=[4,3], figsize = [15,15], bins=10)
#df_train[df_train["Embarked"]=="C"]
df_test[["embC","embQ","embS"]]=pd.get_dummies(df_test["Embarked"], dtype="uint8")
#df_train.hist("Embarked",by=["Survived","Pclass"],layout=[2,3], figsize = [10,8]) #Just ran this to see if there is any significant pattern in data
#FAMILY SIZE
#Normalizing with same Mean and Std as Training data set
df_test["tot_family_size"] = df_test["Parch"]+df_test["SibSp"]
df_test["norm_family_size"] = (df_test["tot_family_size"]-0.9046015712682379)/(1.6134585413550788)
#One-hot encode passenger class
df_test["Pclass"].hist()
df_test[["P2","P3"]]=pd.get_dummies(df_test["Pclass"],drop_first=True)
#Encode cabin information

df_test["cab"] = df_test["Cabin"].str.lower().str.get(0)

df_test["cab"].fillna(value="z",inplace=True)
df_test["cab"].unique()
df_test[["cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_z"]] =pd.get_dummies(df_test["cab"],drop_first=True)

#It appears that the file gets stored to a folder called working. See below. 
df_test.info()
df_test.to_csv(path_or_buf="test_processed.csv")
print(os.listdir("../"))
print(os.listdir("../working"))
import seaborn as sns
corr = df_test[["P2","P3","norm_len_name","title_ms","title_mrs","title_mr","title_others","is_male","age_norm","norm_family_size","norm_fare",
                    "cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_z","embQ","embS"]].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(0, 50, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

