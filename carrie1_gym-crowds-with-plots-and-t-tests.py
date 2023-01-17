import pandas as pd

df = pd.read_csv('../input/data.csv')

df.info() 
df.head() 
df.describe() #descriptive statistics
import seaborn as sns 

import matplotlib.pyplot as plt

sns.set_style("white")

sns.set_palette('Reds_d')

%matplotlib inline

sns.violinplot(y=df["number_people"])
sns.distplot(df["number_people"])
sns.pointplot(x="day_of_week", y="number_people",data=df)
sns.pointplot(x="month", y="number_people",hue='is_during_semester',data=df)
sns.pointplot(x="hour", y="number_people",data=df)
sns.jointplot(x="number_people", y="temperature", kind="kde",data=df)
sns.violinplot(x="is_weekend", y="number_people", data=df)
df.groupby(by='is_weekend').mean()
from scipy import stats

stats.ttest_ind(df['number_people'],df['is_weekend'], equal_var = False)
sns.violinplot(x="is_start_of_semester", y="number_people", data=df)
stats.ttest_ind(df['number_people'],df['is_start_of_semester'], equal_var = False)
sns.violinplot(x="is_during_semester", y="number_people", data=df)
stats.ttest_ind(df['number_people'],df['is_during_semester'], equal_var = False)
RF = df[["is_during_semester",'is_start_of_semester','is_weekend','hour',

         'day_of_week','is_holiday','temperature','month','number_people']]

features = RF.columns[:8]

features
from sklearn.ensemble import RandomForestRegressor

sns.set_style("white")

colors = ['silver',"black", 'charcoal']

sns.set_palette(sns.xkcd_palette(colors))

y = RF['number_people']

rgr1 = RandomForestRegressor(n_jobs=5)

rgr1.fit(RF[features], y)



features_out = pd.DataFrame({'Feature Name':features,'Importance': rgr1.feature_importances_})

features_out.sort_values(by='Importance',inplace=True,ascending=False)



sns.factorplot(x="Feature Name", y="Importance",data=features_out, kind='bar',size=9)
rgr1.score(RF[features], y)
start = df[df['is_start_of_semester'].isin([1])]

RF_start = start[['is_weekend','hour','day_of_week','number_people']]

features = RF_start.columns[:3]

y = RF_start['number_people']

rgr2 = RandomForestRegressor(n_jobs=2)

rgr2.fit(RF_start[features], y)



features_out2 = pd.DataFrame({'Feature Name':features,'Importance': rgr2.feature_importances_})

features_out2.sort_values(by='Importance',inplace=True,ascending=False)



sns.factorplot(x="Feature Name", y="Importance",data=features_out2, kind='bar',size=9)
start_and_morn = start[start['hour'].isin([6,7,8,9,10,11])]

RF_start_and_morn = start_and_morn[['is_weekend','hour','day_of_week','number_people']]

features = RF_start_and_morn.columns[:3]

y = RF_start_and_morn['number_people']

rgr3 = RandomForestRegressor(n_jobs=5)

rgr3.fit(RF_start_and_morn[features], y)



features_out3 = pd.DataFrame({'Feature Name':features,'Importance': rgr3.feature_importances_})

features_out3.sort_values(by='Importance',inplace=True,ascending=False)



sns.factorplot(x="Feature Name", y="Importance",data=features_out3, kind='bar',size=9)
start_and_aft = start[start['hour'].isin([12,1,2,3,4])]

RF_start_and_aft = start_and_aft[['is_weekend','hour','day_of_week','number_people']]

features = RF_start_and_aft.columns[:3]

y = RF_start_and_aft['number_people']

rgr4 = RandomForestRegressor(n_jobs=5)

rgr4.fit(RF_start_and_aft[features], y)



features_out4 = pd.DataFrame({'Feature Name':features,'Importance': rgr4.feature_importances_})

features_out4.sort_values(by='Importance',inplace=True,ascending=False)



sns.factorplot(x="Feature Name", y="Importance",data=features_out4, kind='bar',size=9)