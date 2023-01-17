import pandas as pd
pd.set_option('display.max_columns', None)
from pandas.api.types import CategoricalDtype
import numpy as np   
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
#Task 1: Data Preparation
df=pd.read_csv("../input/star-wars-survey-data/star_wars.csv",encoding='cp1252')
#lets see what are we dealing with
df.head(10)
df.shape
df.columns
#The column name does not make much sense right now 
#we can see that all unnamed is the option what are the relevent options lets see
df.iloc[0]
#lets form appropriate columns . 1st row show all options replacing all non informative unnamed shit aand add column which makes sense
columns_dict = {'RespondentID':'RespondentID','Have you seen any of the 6 films in the Star Wars franchise?':'Q1','Do you consider yourself to be a fan of the Star Wars film franchise?':'Q2',
          'Which of the following Star Wars films have you seen? Please select all that apply.':'Q3/O1','Unnamed: 4':'Q3/O2','Unnamed: 5':'Q3/O3','Unnamed: 6':'Q3/O4','Unnamed: 7':'Q3/O5','Unnamed: 8':'Q3/O6',
          'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'Q4/O1','Unnamed: 10':'Q4/O2','Unnamed: 11':'Q4/O3','Unnamed: 12':'Q4/O4',
          'Unnamed: 13':'Q4/O5','Unnamed: 14':'Q4/O6','Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'Q5/O1','Unnamed: 16':'Q5/O2', 'Unnamed: 17':'Q5/O3', 'Unnamed: 18':'Q5/O4', 'Unnamed: 19':'Q5/O5',
          'Unnamed: 20':'Q5/O6', 'Unnamed: 21':'Q5/O7', 'Unnamed: 22':'Q5/O8', 'Unnamed: 23':'Q5/O9','Unnamed: 24':'Q5/O10', 'Unnamed: 25':'Q5/O11', 'Unnamed: 26':'Q5/O12', 'Unnamed: 27':'Q5/O13','Unnamed: 28':'Q5/O14',
          'Which character shot first?':'Q6','Are you familiar with the Expanded Universe?':'Q7','Do you consider yourself to be a fan of the Expanded Universe?ÂŒÃ¦':'Q8',
          'Do you consider yourself to be a fan of the Star Trek franchise?':'Q9','Gender':'Gender','Age':'Age','Household Income':'Household Income','Education':'Education','Location (Census Region)':'Location (Census Region)'}
df.columns=df.columns.to_series().map(columns_dict)
df.columns#new column names
a=pd.Series(columns_dict,name='new')
a.index.name='old'
a.reset_index()
#make something to track our impute question imputation
q={'questions':['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9'],'Details':['Have you seen any of the 6 films in the Star Wars franchise?','Do you consider yourself to be a fan of the Star Wars film franchise?','Which of the following Star Wars films have you seen? Please select all that apply.','Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.','Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.','Which character shot first?','Are you familiar with the Expanded Universe?','Do you consider yourself to be a fan of the Expanded Universe?','Do you consider yourself to be a fan of the Star Trek franchise?']}
quesions=pd.DataFrame(data=q)
pd.options.display.max_colwidth = 150
quesions
#making dict to keep track of imputed options
options_dict=['Q3/O1', 'Q3/O2', 'Q3/O3', 'Q3/O4', 'Q3/O5',
       'Q3/O6', 'Q4/O1', 'Q4/O2', 'Q4/O3', 'Q4/O4', 'Q4/O5', 'Q4/O6', 'Q5/O1',
       'Q5/O2', 'Q5/O3', 'Q5/O4', 'Q5/O5', 'Q5/O6', 'Q5/O7', 'Q5/O8', 'Q5/O9',
       'Q5/O10', 'Q5/O11', 'Q5/O12', 'Q5/O12', 'Q5/O13']

options={}
for col in df.columns:
    if col in options_dict:
        options.update({col:df.loc[0,col]})
        
options 
df=df.iloc[1:]
df.head()
#by general observation if someone did not responded to q2 all other values are nan
df.loc[(df['Q2'].isnull())&(df['Q1']=='Yes')]
#removing all these values i am not removing No values as it gives some demographics info on ehat kind of people don't like movies
df=df.drop(df[(df.Q2.isnull())&(df.Q1=='Yes')].index)
df.loc[df['Q1']=='No'].shape[0]
df['Q2'].isnull().sum()#all remaning value na in q2 comes from people who have not watched any movies
#now we are sorted with columns and preserve all info so lets find out what's wrong
df['Q1'].value_counts().index#we can see whitespace
df['Q1']=df['Q1'].str.strip()
df['Q2'].value_counts()
col_q2={'No':0,'Noo':0,'Yes':1,'Yess':1}
df['Q2']=df['Q2'].replace(col_q2)
#now we are just going to treat yes data as all no in q2 leve rest of column as na
df_yes=df.loc[df['Q1']=='Yes']
df_yes.shape
#for question 3 we have name of series as person have watched it and nan if not
#writing a finction to replace it with 1 and 0

def impute_val(col):
    df_yes.loc[df[col].notnull(),col]=1
    df_yes.loc[df[col].isnull(),col]=0

q3_options=['Q3/O1','Q3/O2','Q3/O3','Q3/O4','Q3/O5','Q3/O6']
#writing a loop to automate
for col in q3_options:
    impute_val(col)

#for question 4
#levels are 1,2,3,4,5,6 and we will make 0 for nan values(noooooo)
#since it is ordinal so imputing 0 is not logical

#MANUAL IMPUTE

df_yes.loc[(df_yes['Q4/O3'].isnull()),'Q4/O3']=6  
df_yes.loc[(df_yes['Q4/O1'].isnull()),'Q4/O1']=6  
# o3 and o1 have na value but it is in order like 1,2,3,4,5 and for say 6 is missing so imputing the remaning shit
df_yes['Q4/O1']=df_yes['Q4/O1'].astype(int)
df_yes['Q4/O2']=df_yes['Q4/O2'].astype(int)
df_yes['Q4/O3']=df_yes['Q4/O3'].astype(int)
df_yes['Q4/O4']=df_yes['Q4/O4'].astype(int)
df_yes['Q4/O5']=df_yes['Q4/O5'].astype(int)
df_yes['Q4/O6']=df_yes['Q4/O6'].astype(int)
df_yes['Q5/O1'].value_counts()
df_yes['Q8'].isnull().sum()
df_yes['Q8'].value_counts()
df_yes['Q8']=df_yes['Q8'].fillna('Not Answred')
df_yes['Q8']=df_yes['Q8'].replace({'Yess':'Yes'})
df_yes['Q9'].value_counts()
df_yes['Q9']=df_yes['Q9'].replace({'yes':'Yes','Noo':'No'})
df_yes['Gender'].value_counts()
gender_dict={"Male":"Male","Female":"Female","female":"Female","F":"Female","male":"Male"}
df_yes['Gender']=df_yes['Gender'].replace(gender_dict)
df_yes['Age'].value_counts()
#500 dosent makes sense so remove
df_yes['Age']=df_yes['Age'].drop(df_yes[df_yes.Age=='500'].index)
#NULL VALUES
for col in df_yes.columns:
    print(col, df_yes[col].isnull().sum())
df_knn=df_yes.iloc[:,2:]
df_topredict=df_knn.loc[df_yes['Household Income'].isnull(),]
df_topredict=df_topredict.drop('Household Income',axis=1)
df_topredict=df_topredict.dropna()
df_knn=df_knn.dropna()
num_encode_q5={'Unfamiliar (N/A)':0,'Very unfavorably':1,'Somewhat unfavorably':2,'Neither favorably nor unfavorably (neutral)':3,'Somewhat favorably':4,'Very favorably':5}

df_knn.loc[:,'Q5/O1':'Q5/O14']=df_knn.loc[:,'Q5/O1':'Q5/O14'].replace(num_encode_q5)

df_topredict.loc[:,'Q5/O1':'Q5/O14']=df_topredict.loc[:,'Q5/O1':'Q5/O14'].replace(num_encode_q5)
df_knn['Q5/O1'].dtype
df_knn=pd.concat([df_knn,pd.get_dummies(df_knn['Q6'],prefix=['Q6'])],axis=1)#SUCCESS
df_knn=pd.concat([df_knn,pd.get_dummies(df_knn['Q8'],prefix=['Q8'])],axis=1)#SUCCESS
df_knn=pd.concat([df_knn,pd.get_dummies(df_knn['Location (Census Region)'],prefix=['LSR'])],axis=1)#SUCCESS
df_knn=df_knn.drop(['Q6','Q8','Location (Census Region)'],axis=1)#SUCCESS


df_topredict=pd.concat([df_topredict,pd.get_dummies(df_topredict['Q6'],prefix=['Q6'])],axis=1)#SUCCESS
df_topredict=pd.concat([df_topredict,pd.get_dummies(df_topredict['Q8'],prefix=['Q8'])],axis=1)#SUCCESS
df_topredict=pd.concat([df_topredict,pd.get_dummies(df_topredict['Location (Census Region)'],prefix=['LSR'])],axis=1)#SUCCESS
df_topredict=df_topredict.drop(['Q6','Q8','Location (Census Region)'],axis=1)#SUCCESS
df_topredict['Q7'].value_counts()
def binary_conv(col):
    df_knn.loc[df_knn[col]=='No',col]=0
    df_knn.loc[df_knn[col]=='Yes',col]=1
    
def binary_conv_pre(col):
    df_topredict.loc[df_topredict[col]=='No',col]=0
    df_topredict.loc[df_topredict[col]=='Yes',col]=1
    
binary_conv('Q7')#success
binary_conv('Q9')#success

binary_conv_pre('Q7')#success
binary_conv_pre('Q9')#success
df_knn.loc[df_knn['Gender']=='Female','Gender']=0#success
df_knn.loc[df_knn['Gender']=='Male','Gender']=1#success


df_topredict.loc[df_topredict['Gender']=='Female','Gender']=0#success
df_topredict.loc[df_topredict['Gender']=='Male','Gender']=1#success
df_knn['Household Income'].value_counts()
num_encode_age={'18-29':1,'30-44':2,'45-60':3,'> 60':4}
num_encode_edu={'Less than high school degree':1,'High school degree':2,'Some college or Associate degree':3,'Bachelor degree':4,'Graduate degree':5}
num_encode_house_inc={'$0 - $24,999':1,'$25,000 - $49,999':2,'$50,000 - $99,999':3,'$100,000 - $149,999':4,'$150,000+':5}
df_knn['Age']=df_knn['Age'].replace(num_encode_age)#success
df_knn['Education']=df_knn['Education'].replace(num_encode_edu)#success
df_knn['Household Income']=df_knn['Household Income'].replace(num_encode_house_inc)#success


df_topredict['Age']=df_topredict['Age'].replace(num_encode_age)#success
df_topredict['Education']=df_topredict['Education'].replace(num_encode_edu)#success

df_topredict['Q4/O1']=df_topredict['Q4/O1'].astype(int)
df_topredict['Q4/O2']=df_topredict['Q4/O2'].astype(int)
df_topredict['Q4/O3']=df_topredict['Q4/O3'].astype(int)
df_topredict['Q4/O4']=df_topredict['Q4/O4'].astype(int)
df_topredict['Q4/O5']=df_topredict['Q4/O5'].astype(int)
df_topredict['Q4/O6']=df_topredict['Q4/O6'].astype(int)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
y=df_knn.loc[:,'Household Income']
df_knn=df_knn.drop('Household Income',axis=1)
X_train, X_test, y_train, y_test = train_test_split(df_knn, y, test_size=0.33, random_state=42)
parma={'max_depth':[3,4,5,6,7,8,9,10,11,12,13,14,15,16],'criterion':('gini','entropy')}
clf_gini = DecisionTreeClassifier(random_state = 100) 
cv=GridSearchCV(estimator=clf_gini,param_grid=parma,cv=5)
cv.fit(X_train, y_train)
cv.best_estimator_
y_pred = cv.predict(X_test)
accuracy_score(y_test,y_pred)*100
res=y_pred = cv.predict(df_topredict)
df_yes.head()
df_topredict['Household Income']=res
for i in range(129):
    for j in range(835):
        if(df_topredict.index[i]==df_yes.index[j]):
            df_yes.iloc[j,35]=df_topredict.iloc[i,47]
#df_yes['Household Income']=df_yes['Household Income'].replace(num_encode_house_inc)
num_encode_house_inc={'$0 - $24,999':1,'$25,000 - $49,999':2,'$50,000 - $99,999':3,'$100,000 - $149,999':4,'$150,000+':5}
my_dict2 = {y:x for x,y in num_encode_house_inc.items()}
df_yes['Household Income']=df_yes['Household Income'].replace(my_dict2)
df_yes=df_yes.dropna()
df_yes.shape[0]+df.loc[df['Q1']=='No'].shape[0]
((1187-988)/1187)*100#percent reduction
df_yes.head()
#####################################################################3333
#
one=df_yes.loc[:,'Q4/O1':'Q4/O6']
one=one.mean(axis=0)
one=pd.DataFrame(one)
one=one.reset_index()
#one.columns
m=sns.pointplot(x="index", y=one.columns[1], data=one)
m.set(ylabel='RATING')
m.set_xticklabels(['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi'], rotation='vertical', fontsize=10)
loc_1=df_yes.groupby('Location (Census Region)').sum().loc[:,'Q3/O1':'Q3/O6']
loc_1=loc_1.reset_index()
loc_1=pd.melt(loc_1,id_vars=['Location (Census Region)'])
sns.set_style("whitegrid")
sns.set_context("talk")
loc_1=sns.catplot(data=loc_1,kind="bar",x='Location (Census Region)',y='value',hue='variable',orient = "v",legend_out=True)
axes = loc_1.axes.flatten()
axes[0].set_title("Popularity of movies in different region")
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in loc_1.axes.flat]
loc_1.fig.set_figwidth(30)
loc_1.fig.set_figheight(10)
loc_1.despine(left=True)
# title
new_title = 'Movies'
loc_1._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(loc_1._legend.texts, new_labels): t.set_text(l)
loc_1=df_yes.groupby('Location (Census Region)').sum().loc[:,'Q3/O1':'Q3/O6']
loc_1=loc_1.reset_index()
loc_1=pd.melt(loc_1,id_vars=['Location (Census Region)'])
loc_1.groupby('Location (Census Region)').sum().reset_index().plot(kind='pie',y='value',autopct='%1.1f%%',labels=loc_1['Location (Census Region)'],figsize=(20, 20))
loc_1=df_yes.groupby('Location (Census Region)').sum().loc[:,'Q3/O1':'Q3/O6']
loc_1=loc_1.reset_index()

a=loc_1.loc[(loc_1['Location (Census Region)']=='Pacific')|(loc_1['Location (Census Region)']=='East North Central'),]
b=loc_1.loc[~(loc_1['Location (Census Region)']=='Pacific')|(loc_1['Location (Census Region)']=='East North Central'),]

a=pd.melt(a, id_vars=['Location (Census Region)'])
sns.catplot(data=a,kind="bar",x='Location (Census Region)',y='value',hue='variable',orient = "v",legend_out=True)

b=pd.melt(loc_1, id_vars=['Location (Census Region)'])
x=sns.catplot(data=b,kind="bar",x='Location (Census Region)',y='value',hue='variable',orient = "v",legend_out=True)
x.fig.set_figwidth(30)
x.fig.set_figheight(10)
loc_1=df_yes.groupby('Location (Census Region)').mean().loc[:,'Q4/O1':'Q4/O6']
loc_1=loc_1.reset_index()
loc_1=pd.melt(loc_1,id_vars=['Location (Census Region)'])
sns.set_style("whitegrid")
sns.set_context("talk")
loc_1=sns.catplot(data=loc_1,kind="bar",x='Location (Census Region)',y='value',hue='variable',orient = "v",legend_out=True)
axes = loc_1.axes.flatten()
axes[0].set_title("Popularity of movies in different region")
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in loc_1.axes.flat]
loc_1.fig.set_figwidth(30)
loc_1.fig.set_figheight(10)
loc_1.despine(left=True)
# title
new_title = 'Movies'
loc_1._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(loc_1._legend.texts, new_labels): t.set_text(l)
q4=['Q4/O1','Q4/O2','Q4/O3','Q4/O4','Q4/O5','Q4/O6']
one=[]
two=[]
three=[]
four=[]
five=[]
six=[]

for col in q4:
    a=df_yes[col].value_counts()
    a=dict(a)
    one.append(a[1])
    two.append(a[2])
    three.append(a[3])
    four.append(a[4])
    five.append(a[5])
    six.append(a[6])

grouped_bar=pd.DataFrame({'options':q4,'1':one,'2':two,'3':three,'4':four,'5':five,'6':six})
grouped_bar['options']=grouped_bar['options'].replace(options)
x=pd.melt(grouped_bar,id_vars=["options"])

sns.set_style("whitegrid")
sns.set_context("talk")
g=sns.catplot(data=x,kind="bar",x='options',y='value',hue='variable',orient = "v")
axes = g.axes.flatten()
axes[0].set_title("Rating of Movies")
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
g.fig.set_figwidth(22)
g.fig.set_figheight(10)
g.despine(left=True)


# title
new_title = 'Ratings'
g._legend.set_title(new_title)
hhi=df_yes.groupby(['Household Income']).sum().loc[:,'Q3/O1':'Q3/O6']
hhi['Household_Income']=hhi.index
hhi=pd.melt(hhi,id_vars=['Household_Income'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Household_Income',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
hhi=df_yes.groupby(['Household Income']).sum().loc[:,'Q3/O1':'Q3/O6']
hhi['Household_Income']=hhi.index
hhi=pd.melt(hhi,id_vars=['Household_Income'])
hhi=hhi.drop('variable',axis=1)
hhi.groupby('Household_Income').sum().reset_index().plot(kind='pie',y='value',autopct='%1.1f%%',labels=hhi['Household_Income'],figsize=(20, 20))
hhi=df_yes.groupby(['Household Income']).mean().loc[:,'Q4/O1':'Q4/O6']
hhi['Household_Income']=hhi.index
hhi=pd.melt(hhi,id_vars=['Household_Income'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Household_Income',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
age_avr_rat=df_yes.groupby(['Age']).sum().loc[:,'Q3/O1':'Q3/O6']
age_avr_rat=age_avr_rat.reset_index()
age_avr_rat=pd.melt(age_avr_rat,id_vars=['Age'])

age_avr_rat=age_avr_rat.drop('variable',axis=1)
age_avr_rat.groupby('Age').sum().reset_index().plot(kind='pie',y='value',autopct='%1.1f%%',labels=age_avr_rat['Age'],figsize=(20, 20))      


hhi=df_yes.groupby(['Age']).sum().loc[:,'Q3/O1':'Q3/O6']
hhi['Age']=hhi.index
hhi=pd.melt(hhi,id_vars=['Age'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Age',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
hhi=df_yes.groupby(['Age']).mean().loc[:,'Q4/O1':'Q4/O6']
hhi['Age']=hhi.index
hhi=pd.melt(hhi,id_vars=['Age'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Age',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
age_avr_rat=df_yes.groupby(['Age']).mean().loc[:,'Q4/O1':'Q4/O6']
age_avr_rat=age_avr_rat.reset_index()
age_avr_rat=pd.melt(age_avr_rat,id_vars=['Age'])

ax=sns.pointplot(x="variable", y="value", hue="Age",data=age_avr_rat, dodge=True)
ax.legend()
sns.set(rc={'figure.figsize':(21.7,8.27)})
ax.set(ylabel='RATING')
ax.set_xticklabels(['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi'], rotation=20, fontsize=10)
age_avr_rat=df_yes.groupby(['Education']).sum().loc[:,'Q3/O1':'Q3/O6']
age_avr_rat=age_avr_rat.reset_index()
age_avr_rat=pd.melt(age_avr_rat,id_vars=['Education'])

age_avr_rat=age_avr_rat.drop('variable',axis=1)
age_avr_rat.groupby('Education').sum().reset_index().plot(kind='pie',y='value',autopct='%1.1f%%',labels=age_avr_rat['Education'],figsize=(20, 20),legend=False)      

hhi=df_yes.groupby(['Education']).sum().loc[:,'Q3/O1':'Q3/O6']
hhi['Education']=hhi.index
hhi=pd.melt(hhi,id_vars=['Education'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Education',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
hhi=df_yes.groupby(['Education']).mean().loc[:,'Q4/O1':'Q4/O6']
hhi['Education']=hhi.index
hhi=pd.melt(hhi,id_vars=['Education'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Education',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
age_avr_rat=df_yes.groupby(['Education']).mean().loc[:,'Q4/O1':'Q4/O6']
age_avr_rat=age_avr_rat.reset_index()
age_avr_rat=pd.melt(age_avr_rat,id_vars=['Education'])

ax=sns.pointplot(x="variable", y="value", hue="Education",data=age_avr_rat, dodge=True)
ax.legend()
sns.set(rc={'figure.figsize':(21.7,8.27)})
ax.set(ylabel='RATING')
ax.set_xticklabels(['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi'], rotation=20, fontsize=10)
asum=df_yes.groupby(['Age','Gender']).sum().loc[:,'Q3/O1':'Q3/O6']
m=asum.reset_index()
m['value']=m.sum(axis=1)
des_col=['Age','Gender','value']
m=m[des_col]
m = m.pivot(columns='Gender',index='Age').fillna(0)
m.plot.bar(stacked=True)
#a=df_yes.loc[:,['Gender','Q4/O1']]
#sns.stripplot(x="Gender", y="Q4/O1", data=a, jitter=0.05)
df_no=df.loc[df['Q1']=='No']
df_no['Age'].value_counts().plot.bar()
hhi=df_yes.groupby(['Gender']).mean().loc[:,'Q4/O1':'Q4/O6']
hhi['Gender']=hhi.index
hhi=pd.melt(hhi,id_vars=['Gender'])
sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=hhi,kind="bar",x='Gender',y='value',hue='variable',orient = "v")
m.fig.set_figwidth(30)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Movie viewership as per household income")

# title
new_title = 'Movies'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
age_avr_rat=df_yes.groupby(['Gender']).mean().loc[:,'Q4/O1':'Q4/O6']
age_avr_rat=age_avr_rat.reset_index()
age_avr_rat=pd.melt(age_avr_rat,id_vars=['Gender'])

ax=sns.pointplot(x="variable", y="value", hue='Gender',data=age_avr_rat, dodge=True)
ax.legend()
sns.set(rc={'figure.figsize':(21.7,8.27)})
ax.set(ylabel='RATING')
ax.set_xticklabels(['Star Wars: Episode I  The Phantom Menace', 'Star Wars: Episode II  Attack of the Clones','Star Wars: Episode III  Revenge of the Sith','Star Wars: Episode IV  A New Hope','Star Wars: Episode V The Empire Strikes Back','Star Wars: Episode VI Return of the Jedi'], rotation=20, fontsize=10)
#3333333333333333333333333333333333333333333333333 1999, 2002, 2005, 1977, 1980, 1983
co=pd.concat([df_yes.loc[:,'Q3/O1':'Q3/O6'].sum().reset_index(),df_yes.loc[:,'Q4/O1':'Q4/O6'].mean().reset_index()],axis=1)
co.columns=['index','number','index2','rating']
co['number'].corr(co['rating'])

df_full_1 = pd.concat([df_yes.iloc[:,1:], df.loc[df['Q1']=='No','Q1':]])
df_full_1=df_full_1.reset_index()
df_full_1['Gender']=df_full_1['Gender'].replace(gender_dict)
data_crosstab = pd.crosstab(df_full_1['Q1'], 
                            df_full_1['Gender'],  
                               margins = False)
data_crosstab.plot.bar(stacked=True)
data_crosstab
scipy.stats.chi2_contingency(data_crosstab)#do the writeup


x=df_yes.loc[:,['Education','Q4/O3']]
x.columns=['Education', 'value']
import statsmodels.api as sm
from statsmodels.formula.api import ols
lm=ols('value ~ C(Education)',data=x).fit()
table=sm.stats.anova_lm(lm)
print(table)
x=x.groupby('Education').mean().reset_index()
x
x.groupby('Education').count()
x.plot.bar()
labels=x['Education']
plt.xticks(x.index,labels, rotation=0)

x=[]
aa=df_yes.loc[:,'Q5/O1':'Q5/O14']
for c in aa.columns:
    x.append(aa[c].value_counts().sort_index().values)

x=pd.DataFrame(x)
x=x.transpose()
x['relationship']=['Neither favorably nor unfavorably (neutral)','Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Somewhat unfavorably']
x.columns=['Han Solo','Luke Skywalker','Princess Leia Organa','Anakin Skywalker','Obi Wan Kenobi','Emperor Palpatine','Darth Vader','Lando Calrissian','Boba Fett','C-3P0','R2 D2','Jar Jar Binks','Padme Amidala','Yoda','relationship']
x
#being of diffiernt gender makes you like or dislike hime
columns=['Q5/O1','Gender']
x=df_yes.loc[:,columns]

data_crosstab = pd.crosstab(x['Q5/O1'], 
                            df_full_1['Gender'],  
                               margins = False)
scipy.stats.chi2_contingency(data_crosstab)
data_crosstab.plot.bar(stacked=True)
plt.xticks(rotation=20)

figure(num=None, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')

x=[]
aa=df_yes.loc[df_yes['Education']=='Some college or Associate degree','Q5/O1':'Q5/O14']
for c in aa.columns:
    x.append(aa[c].value_counts().sort_index().values)

x=pd.DataFrame(x)
x=x.transpose()
x['relationship']=['Neither favorably nor unfavorably (neutral)','Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Somewhat unfavorably']
x.columns=['Han Solo','Luke Skywalker','Princess Leia Organa','Anakin Skywalker','Obi Wan Kenobi','Emperor Palpatine','Darth Vader','Lando Calrissian','Boba Fett','C-3P0','R2 D2','Jar Jar Binks','Padme Amidala','Yoda','relationship']
x=pd.melt(x,id_vars=['relationship'])
plt.scatter(x="variable", y="relationship", s="value", data=x)
plt.xticks(rotation=70)

def long_pot(col_car,col):
    figure(num=None, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
    x=[]
    aa=df_yes.loc[df_yes[col]==col_car,'Q5/O1':'Q5/O14']
    for c in aa.columns:
        x.append(aa[c].value_counts().sort_index().values)
    x=pd.DataFrame(x)
    x=x.transpose()
    x['relationship']=['Neither favorably nor unfavorably (neutral)','Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Somewhat unfavorably']
    x.columns=['Han Solo','Luke Skywalker','Princess Leia Organa','Anakin Skywalker','Obi Wan Kenobi','Emperor Palpatine','Darth Vader','Lando Calrissian','Boba Fett','C-3P0','R2 D2','Jar Jar Binks','Padme Amidala','Yoda','relationship']
    x=pd.melt(x,id_vars=['relationship'])
    plt.scatter(x="variable", y="relationship", s="value", data=x)
    plt.xticks(rotation=70)
#1999, 2002, 2005, 1977, 1980, 1983
df_yes['Age'].value_counts().sort_index().index
plot=df_yes['Age'].value_counts().sort_index().index

for do in plot:
    long_pot(do,'Age')
c=['Q5/O1','Q5/O2','Q5/O3','Q5/O4','Q5/O5','Q5/O6','Q5/O7','Q5/O8','Q5/O9','Q5/O10','Q5/O11','Q5/O12','Q5/O13','Q5/O14','Age']
dfh=df_yes.loc[:,c]
dfh=pd.melt(dfh,id_vars=['Age'])
dfh=dfh.groupby(['Age','value']).count().reset_index()

so=dfh.groupby(['Age','value']).agg({'variable':'sum'})
dfh=so.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=dfh,kind="bar",x='Age',y='variable',hue='value',orient = "v")
m.fig.set_figwidth(50)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Familarity proprtions with characters")

# title
new_title = 'Familarity'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Neither favorably nor unfavorably (neutral)', 'Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Very unfavorably']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)

df_yes['Household Income'].value_counts().sort_index().index
plot=df_yes['Household Income'].value_counts().sort_index().index

for do in plot:
    long_pot(do,'Household Income')
c=['Q5/O1','Q5/O2','Q5/O3','Q5/O4','Q5/O5','Q5/O6','Q5/O7','Q5/O8','Q5/O9','Q5/O10','Q5/O11','Q5/O12','Q5/O13','Q5/O14','Household Income']
dfh=df_yes.loc[:,c]
dfh=pd.melt(dfh,id_vars=['Household Income'])
dfh=dfh.groupby(['Household Income','value']).count().reset_index()

so=dfh.groupby(['Household Income','value']).agg({'variable':'sum'})
dfh=so.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=dfh,kind="bar",x='Household Income',y='variable',hue='value',orient = "v")
m.fig.set_figwidth(50)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Familarity proprtions with characters")

# title
new_title = 'Familarity'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Neither favorably nor unfavorably (neutral)', 'Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Very unfavorably']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)





df_yes.loc[df_yes['Education']=='Bachelor degree',:]['Age'].value_counts()
plot=['Bachelor degree', 'Graduate degree', 'High school degree', 'Some college or Associate degree']

for do in plot:
    long_pot(do,'Education')
c=['Q5/O1','Q5/O2','Q5/O3','Q5/O4','Q5/O5','Q5/O6','Q5/O7','Q5/O8','Q5/O9','Q5/O10','Q5/O11','Q5/O12','Q5/O13','Q5/O14','Education']
dfh=df_yes.loc[:,c]
dfh=pd.melt(dfh,id_vars=['Education'])
dfh=dfh.groupby(['Education','value']).count().reset_index()

so=dfh.groupby(['Education','value']).agg({'variable':'sum'})
dfh=so.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=dfh,kind="bar",x='Education',y='variable',hue='value',orient = "v")
m.fig.set_figwidth(90)
m.fig.set_figheight(30)

axes = m.axes.flatten()
axes[0].set_title("Familarity proprtions with characters")

# title
new_title = 'Familarity'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Neither favorably nor unfavorably (neutral)', 'Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Very unfavorably']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
plot=df_yes['Location (Census Region)'].value_counts().sort_index().index

for do in plot:
    long_pot(do,'Location (Census Region)')
c=['Q5/O1','Q5/O2','Q5/O3','Q5/O4','Q5/O5','Q5/O6','Q5/O7','Q5/O8','Q5/O9','Q5/O10','Q5/O11','Q5/O12','Q5/O13','Q5/O14','Location (Census Region)']
dfh=df_yes.loc[:,c]
dfh=pd.melt(dfh,id_vars=['Location (Census Region)'])
dfh=dfh.groupby(['Location (Census Region)','value']).count().reset_index()

so=dfh.groupby(['Location (Census Region)','value']).agg({'variable':'sum'})
dfh=so.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

sns.set_style("whitegrid")
sns.set_context("talk")
m=sns.catplot(data=dfh,kind="bar",x='Location (Census Region)',y='variable',hue='value',orient = "v")
m.fig.set_figwidth(70)
m.fig.set_figheight(10)

axes = m.axes.flatten()
axes[0].set_title("Familarity proprtions with characters")

# title
new_title = 'Familarity'
m._legend.set_title(new_title)
# replace labels
new_labels = ['Neither favorably nor unfavorably (neutral)', 'Somewhat favorably','Somewhat unfavorably','Unfamiliar (N/A)','Very favorably','Very unfavorably']
for t, l in zip(m._legend.texts, new_labels): t.set_text(l)
#THANKYOU HOPE YOU LIKED IT




