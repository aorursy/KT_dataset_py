# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  ## plot library

from sklearn import preprocessing  ## preprocessing form sklearn to deal with type object

import seaborn as sns  #import seaborn for correlation matrix

import xgboost as xgb #import xgboost to train missing values









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## quick check if import is working correctly and vizualize head of data



train_df=pd.read_csv("../input/basic_income_dataset_dalia.csv",encoding ='utf-8')

train_df.head()
## Explore object type in columns

dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

print(dtype_df.groupby("Column Type").aggregate('count').reset_index())
## explore missing values in columns

missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(6,8))

rects = ax.barh(ind, missing_df.missing_count.values, color='y')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()


## Transform columns with types object into labels



for f in train_df.columns:

    if train_df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        
###Simplify data. Simplify column names

#rename the columns to be less wordy

train_df.rename(columns = {'question_bbi_2016wave4_basicincome_awareness':'awareness',

            'question_bbi_2016wave4_basicincome_vote':'vote',

            'question_bbi_2016wave4_basicincome_effect':'effect',

            'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',

            'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against'},

           inplace = True)





##Remove the shade between "would probably" and "would" so we have only 3 cases

train_df.vote.replace([3, 4], [1, 2], inplace=True)

train_df['vote'].value_counts().plot(kind = 'bar')



#Drop weight columns for now as i don't get why there are 1 dot sometimes and 4 rest of the time

## drop uuid too as it doesn't matter in feature and correlation

train_df = train_df.drop(["weight","uuid"], axis=1)
train_y= train_df.vote.values

train_X=train_df.drop(['vote'], axis=1)



## quick xgb to use in imprtant features



xgb_params = {

    'eta': 0.05,

    'max_depth': 3,

    'subsample': 0.5,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
## Correlation Matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=1, square=True, annot=True);
# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, height=0.8, ax=ax)

plt.show()
df=pd.read_csv("../input/basic_income_dataset_dalia.csv",encoding ='utf-8')



###Simplify data. Simplify column names

#rename the columns to be less wordy

df.rename(columns = {'question_bbi_2016wave4_basicincome_awareness':'awareness',

            'question_bbi_2016wave4_basicincome_vote':'vote',

            'question_bbi_2016wave4_basicincome_effect':'effect',

            'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',

            'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against'},

           inplace = True)

## Merge 2 kind of favorables people and filrter only favorable people 

df.vote.replace('I would probably vote for it','I would vote for it', inplace=True)

yes_df=df[df['vote'] =='I would vote for it' ]







##plot count of effect on themselves mentionned for yes_vote only



yes_df['effect'].value_counts().plot(kind='bar',title='Basic income effect importance. People favorable')

##same for people against

df.vote.replace('I would probably vote against it','I would vote against it', inplace=True)

no_df=df[df['vote'] =='I would vote against it' ]



##same for no

no_df['effect'].value_counts().plot(kind='bar',title='Basic income effect importance. People against')
## plot arguments for mentionned for yes_vote only

## several arguments possible 

## need additional work to split the list of arguments and then count them. Split on |

#yes_df['arg_for'].value_counts().plot(kind='bar',title='Basic income argument for importance')

test=df.loc[df['uuid']=='d468c670-da40-0133-de39-0a81e8b09a82'].arg_for.values[0].split('|')

#print(test)



arg_for=['It reduces anxiety about financing basic needs ',' It creates more equality of opportunity','It encourages financial independence and self-responsibility',' It increases solidarity, because it is funded by everyone ','It reduces bureaucracy and administrative expenses','It increases appreciation for household work and volunteering','None of the above']

#print (arg_for)

arg_for_count=[0,0,0,0,0,0,0]



for x in yes_df.iterrows():

    for i in range(0,len(arg_for)):

        if arg_for[i] in x[1]['arg_for'].split('|'):

            arg_for_count[i]=arg_for_count[i]+1



#print (arg_for_count)

arg_for_simp=['No anxiety','equality','independance','solidarity','less bureaucracy','volunteering','none']

D = {}

for i in range(0,len(arg_for)):

    D[arg_for_simp[i]]=arg_for_count[i]





D_df=pd.DataFrame(list(D.items()),columns=['Arguments for', 'Count'])

D_df.plot(x='Arguments for', y='Count', kind='bar', title ='Arguments for mentionned by proponents ',legend=False)
## plot arguments for regarding answers of people AGAINST basic income



arg_for=['It reduces anxiety about financing basic needs ',' It creates more equality of opportunity','It encourages financial independence and self-responsibility',' It increases solidarity, because it is funded by everyone ','It reduces bureaucracy and administrative expenses','It increases appreciation for household work and volunteering','None of the above']

#print (arg_for)

arg_for_count=[0,0,0,0,0,0,0]



for x in no_df.iterrows():

    for i in range(0,len(arg_for)):

        if arg_for[i] in x[1]['arg_for'].split('|'):

            arg_for_count[i]=arg_for_count[i]+1



#print (arg_for_count)

arg_for_simp=['No anxiety','equality','independance','solidarity','less bureaucracy','volunteering','none']

D = {}

for i in range(0,len(arg_for)):

    D[arg_for_simp[i]]=arg_for_count[i]







D_df=pd.DataFrame(list(D.items()),columns=['Arguments for', 'Count'])

D_df.plot(x='Arguments for', y='Count', kind='bar', title ='Arguments for mentionned by opponents ',legend=False)
## plot arguments against regarding answers of people AGAINST basic income



arg_against=['It is impossible to finance',

             'It might encourage people to stop working',

             'Foreigners might come to my country and take advantage of the benefit',

             'It is against the principle of linking merit and reward',

             'Only the people who need it most should get something from the state',

             'It increases dependence on the state',

             'None of the above']

#print (arg_for)

arg_for_count=[0,0,0,0,0,0,0]



for x in no_df.iterrows():

    for i in range(0,len(arg_against)):

        if arg_against[i] in x[1]['arg_against'].split('|'):

            arg_for_count[i]=arg_for_count[i]+1



#print (arg_for_count)

arg_for_simp=['Cost','Lazyness','Foreigners','No Meritocracy','Unecessary','Nanny State','None']

D = {}

for i in range(0,len(arg_for)):

    D[arg_for_simp[i]]=arg_for_count[i]



D_df=pd.DataFrame(list(D.items()),columns=['Arguments Against', 'Count'])

D_df.plot(x='Arguments Against', y='Count', kind='bar', title ='Arguments against mentionned by opponents ',legend=False)
## plot arguments against mentionned by people for basic income



arg_against=['It is impossible to finance',

             'It might encourage people to stop working',

             'Foreigners might come to my country and take advantage of the benefit',

             'It is against the principle of linking merit and reward',

             'Only the people who need it most should get something from the state',

             'It increases dependence on the state',

             'None of the above']

#print (arg_for)

arg_for_count=[0,0,0,0,0,0,0]



for x in yes_df.iterrows():

    for i in range(0,len(arg_against)):

        if arg_against[i] in x[1]['arg_against'].split('|'):

            arg_for_count[i]=arg_for_count[i]+1



#print (arg_for_count)

arg_for_simp=['Cost','Lazyness','Foreigners','No Meritocracy','Unecessary','Nanny State','None']

D = {}

for i in range(0,len(arg_for)):

    D[arg_for_simp[i]]=arg_for_count[i]







D_df=pd.DataFrame(list(D.items()),columns=['Arguments Against', 'Count'])

D_df.plot(x='Arguments Against', y='Count', kind='bar', title ='Arguments against mentionned by proponents ',legend=False)
##Simplified for/against + repartition



sub_df = df.groupby('country_code')['vote'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by country')
## Calculate and plot share of yes per country

## We calculate share on total pop, as if undecided were against



temp= df['country_code'].tolist()

country_code = []

[country_code.append(x) for x in temp if x not in country_code]

#print(country_code)

V={}

for x in country_code:

    V[str(x)]=yes_df[yes_df['country_code']==str(x)].uuid.count()/df[df['country_code']==str(x)].uuid.count()

    

V_df=pd.DataFrame(list(V.items()),columns=['Country_code', 'Proponent %'])

V_df.plot(x='Country_code', y='Proponent %', kind='bar', title ='Basic Income proponents by country',legend=False)
## Not simplified for/against

df=pd.read_csv("../input/basic_income_dataset_dalia.csv",encoding ='utf-8')

###Simplify data. Simplify column names

#rename the columns to be less wordy

df.rename(columns = {'question_bbi_2016wave4_basicincome_awareness':'awareness',

            'question_bbi_2016wave4_basicincome_vote':'vote',

            'question_bbi_2016wave4_basicincome_effect':'effect',

            'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',

            'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against'},

           inplace = True)



sub_df = df.groupby('country_code')['vote'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by country')
## Plotting basic income awareness between countries 



sub_df = df.groupby('country_code')['awareness'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by country')
## Plotting basic income awareness by age 

sub_df = df.groupby('age_group')['awareness'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by age')

##print share of yes simplified

print(V_df.sort_values(by='Proponent %',ascending=True))


## Awareness  visualization

df['awareness'].value_counts().plot(kind='pie',title='Basic income awareness',legend=False)
## Plotting basic income awareness between countries 



sub_df = df.groupby('country_code')['awareness'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by country')
## Plotting basic income awareness by age 

sub_df = df.groupby('age_group')['awareness'].value_counts(normalize=True).unstack()

sub_df.plot(kind='bar',stacked=True,colormap='nipy_spectral',sort_columns=True, title='Basic income awareness by age')
