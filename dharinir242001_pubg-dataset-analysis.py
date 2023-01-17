import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import os
print(os.listdir("../input"))

#Reading the csv files
df = pd.read_csv("../input/train_V2.csv")
df1 = pd.read_csv("../input/test_V2.csv")


#summary
df.head()
df1.head()
#Finding pairwise correlation between columns
df_new = df.corr()
#Lineplot between Weapons Acquired and Win Percent
sns.lineplot(x=df['weaponsAcquired'],y=df['winPlacePerc'])

#Line plot between Walk Distance and Win Percent
sns.lineplot(x=df['walkDistance'],y=df['winPlacePerc'])


#Line plot between kills and Win Percent
sns.lineplot(x=df['kills'],y=df['winPlacePerc'])
#Applying OLS model and below is the R squared result. It shows an R squared score of 0.971
target = df_new[['winPlacePerc']]
start_features = df_new[['kills','weaponsAcquired','walkDistance']]
model = sm.OLS(target,start_features).fit()
model.summary()
