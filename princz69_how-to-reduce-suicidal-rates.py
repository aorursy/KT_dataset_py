# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
suicide_rate_df = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

suicide_rate_df.head(5)
suicide_rate_df.describe()
suicide_rate_df=suicide_rate_df.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.

suicide_rate_df.isnull().sum()
#As you can see, most of the HDIForYear value is empty. That's why I want this value deleted.

suicide_rate_df.drop(['HDIForYear','CountryYear'],axis=1,inplace=True)
#1985 min year,2016 max year.



#Now start analysis, min year and max year will find them

min_year=min(suicide_rate_df.Year)

max_year=max(suicide_rate_df.Year)
suicideGender1985=suicide_rate_df.groupby(['Country','Gender']).SuicidesNo.sum()

suicideGender1985
suicidesNo=[]

for country in suicide_rate_df.Country.unique():

    suicidesNo.append(sum(suicide_rate_df[suicide_rate_df['Country']==country].SuicidesNo))

    

suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])

country=pd.DataFrame(suicide_rate_df.Country.unique(),columns=['country'])

data_suicide_countr=pd.concat([suicidesNo,country],axis=1)



data_suicide_countr=data_suicide_countr.sort_values(by='suicidesNo',ascending=False)



sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])

plt.show()
group_data=suicide_rate_df.groupby(['Age','Gender'])['SuicidesNo'].sum().unstack()

group_data=group_data.reset_index().melt(id_vars='Age')



group_data_female=group_data.iloc[:6,:]

group_data_male=group_data.iloc[6:,:]



group_data_female
group_data_male
female_=[175437,208823,506233,16997,430036,221984]

male_=[633105,915089,1945908,35267,1228407,431134]

plot_id = 0

for i,age in enumerate(['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']):

    plot_id += 1

    plt.subplot(3,2,plot_id)

    plt.title(age)

    fig, ax = plt.gcf(), plt.gca()

    sns.barplot(x=['female','male'],y=[female_[i],male_[i]],color='red')

    plt.tight_layout()

    fig.set_size_inches(10, 15)

plt.show() 
suicide_no_1985 = suicide_rate_df[(suicide_rate_df['Year']==min_year)].SuicidesNo.sum()

suicide_no_1985
suicide_no_2015 = suicide_rate_df[(suicide_rate_df['Year']==2015)].SuicidesNo.sum()

suicide_no_2015
sns.barplot(x=['1985','2015'],y=[suicide_no_1985, suicide_no_2015], palette=['blue', 'grey'])
fig, ax = plt.gcf(), plt.gca()

sns.barplot(x=suicide_rate_df.groupby('Age')['SuicidesNo'].sum().index,y=suicide_rate_df.groupby('Age')['SuicidesNo'].sum().values)

fig.set_size_inches(10,5)

plt.show()
# Plot sepal with as a function of sepal_length across days

g = sns.lmplot(x="Year", y="SuicidesNo", hue="Country",

               truncate=True, height=5, data=suicide_rate_df)



# Use more informative axis labels than are provided by default

g.set_axis_labels("Year", "Suicides No")

plt.show()
%matplotlib inline

sns.FacetGrid(suicide_rate_df,hue='Country',size=5).map(plt.scatter,'GdpPerCapitalMoney','SuicidesNo').add_legend()

plt.show()
foreveralone_df = pd.read_csv("../input/the-demographic-rforeveralone-dataset/foreveralone.csv")

foreveralone_df.head(5)
foreveralone_df.describe()
foreveralone_df.info()
foreveralone_df.isnull().sum()
foreveralone_df['gender'] = foreveralone_df['gender'].map({'Transgender female': 'Female', 'Transgender male' : 'Male', 'Male': 'Male', 'Female': 'Female'})
foreveralone_df
columns = ['gender','sexuallity','income','bodyweight','virgin','friends','social_fear','depressed','employment']

target = foreveralone_df['attempt_suicide']
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
le = LabelEncoder()

foreveralone_df['gender'] = le.fit_transform(foreveralone_df['gender'])

foreveralone_df['sexuallity'] = le.fit_transform(foreveralone_df['sexuallity'])

foreveralone_df['income'] = le.fit_transform(foreveralone_df['income'])

foreveralone_df['bodyweight'] = le.fit_transform(foreveralone_df['bodyweight'])

foreveralone_df['virgin'] = le.fit_transform(foreveralone_df['virgin'])

foreveralone_df['social_fear'] = le.fit_transform(foreveralone_df['social_fear'])

foreveralone_df['depressed'] = le.fit_transform(foreveralone_df['depressed'])

foreveralone_df['employment'] = le.fit_transform(foreveralone_df['employment'])
pca = PCA()

foreveralone_df[columns] = pca.fit_transform(foreveralone_df[columns])
clf = DecisionTreeClassifier()

clf.fit(foreveralone_df[columns], target)
clf.feature_importances_
target_pred = cross_val_predict(clf,foreveralone_df[columns],target,cv=3)

print('accuracy: ', accuracy_score(target, target_pred))

print('confusion matrix:')

print(confusion_matrix(target, target_pred))

print('classification report:')

print(classification_report(target, target_pred))