import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/accidental-drug/Accidental_Drug.csv')
df.info(memory_usage='deep')
df = df.drop(['DeathCityGeo','ResidenceCityGeo','InjuryCityGeo','DeathCity','DeathCounty'],axis = 1)
df['Age'].fillna(df['Age'].median(), inplace = True)
df.head()
df['DateType'].value_counts()
df['Death'] = df.DateType.map(lambda x : 0 if x == 'DateofDeath' else 1)
#Let’s see how many of each Race is in our data set
df.Race.fillna('Unknown', inplace=True)
df['Race'].value_counts()
df['Race'].value_counts().plot.bar()
df['Sex'].value_counts()
#Let’s see how many of each sex is in our data set
df['Sex'].value_counts().plot.bar()
df['Female'] = df.Sex.map(lambda x : 1 if x == 'Female' else 0)
df['Male'] = df.Sex.map(lambda x : 1 if x == 'Male' else 0)
#Let’s see types of drugs
df.columns[18:36]
for drug in df.columns[18:36]:
    df[drug] = df[drug].map(lambda x : 1 if x == 'Y' else 0)   
df['NDrugs'] = df.apply(lambda x: x[18:36].sum(), axis='columns')
Drugs = df.groupby('NDrugs').ID.count()
Drugs.plot.bar(x=Drugs.index, y=Drugs.values)
list_ = []
list_of_drugs = df.columns[18:36]
for i in list_of_drugs:
    list_.append(df[i].sum())
Drugs_df = pd.DataFrame(list_,list_of_drugs)
Drugs_df.sort_values(by=[0],ascending=False).plot.bar()
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df_source = df
df_output = df['Death']
df_plt = df.drop(['ID','Date','DateType','OtherSignifican', 'Sex','Race','ResidenceCity','ResidenceCounty','ResidenceState','Location','LocationifOther','DescriptionofInjury','InjuryPlace','InjuryCity','InjuryCounty','InjuryState','COD','OtherSignifican'],axis = 1) 
df = df.drop(['Death','ID','Date','DateType','OtherSignifican', 'Sex','Race','ResidenceCity','ResidenceCounty','ResidenceState','Location','LocationifOther','DescriptionofInjury','InjuryPlace','InjuryCity','InjuryCounty','InjuryState','COD','OtherSignifican'],axis = 1)
inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(df, df_output, test_size = 0.33, random_state = 42)
rf = RandomForestClassifier (n_estimators=100)
rf.fit(inputs_train, expected_output_train)
accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))
f_={}
for feature, importance in zip(df.columns, rf.feature_importances_):
    f_[feature] = importance
f_
filter1 = df_source["Age"]>20 
filter2 =  df_source["Age"]<70
df_source.where(filter1 & filter2, inplace = True)
sns.relplot(x="Age", y="NDrugs",hue = 'Death', kind="line", data=df_source[['Death','Age','NDrugs']]);
sns.relplot(x="Age", y="Fentanyl",hue = 'Death', kind="line", data=df_source[['Death','Age','Fentanyl']]);
sns.relplot(x="Age", y="Heroin",hue = 'Death', kind="line", data=df_source[['Death','Age','Heroin']]);
sns.relplot(x="Age", y="AnyOpioid",hue = 'Death', kind="line", data=df_source[['Death','Age','AnyOpioid']]);
df.info()
rf.predict([[30,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,3]])
rf.predict([[65,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,4]])