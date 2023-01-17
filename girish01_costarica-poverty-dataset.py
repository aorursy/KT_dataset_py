import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting

import seaborn as sns #for visualization

from plotly.offline import init_notebook_mode, iplot



import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

from collections import OrderedDict #to make sorted Dictionary



# Set a few plotting defaults

%matplotlib inline

plt.style.use('dark_background')

plt.rcParams['font.size'] = 15

plt.rcParams['patch.edgecolor'] = 'k'









# Suppress warnings from pandas

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')

train.head()
test['Target']=np.nan
train.shape
train.info()
train.select_dtypes(['object']).head()
#Converting Object DataType to Integer

train['dependency']=train['dependency'].replace(('yes','no'),(1,0)).astype(np.float64)

train['edjefe']=train['edjefe'].replace(('yes','no'),(1,0)).astype(np.float64)

train['edjefa']=train['edjefa'].replace(('yes','no'),(1,0)).astype(np.float64)

test['dependency']=test['dependency'].replace(('yes','no'),(1,0)).astype(np.float64)

test['edjefe']=test['edjefe'].replace(('yes','no'),(1,0)).astype(np.float64)

test['edjefa']=test['edjefa'].replace(('yes','no'),(1,0)).astype(np.float64)
train[['dependency','edjefe','edjefa']].describe()

train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color='red',edgecolor='k',linewidth=2)

plt.xlabel('Number of Unique Values')

plt.ylabel('Count')

plt.title('Number of Unique Values in Integer Columns')

print(train.select_dtypes(np.int64).nunique().value_counts())

plt.show()
data=pd.concat([train,test],axis=0)
head=data.loc[data['parentesco1']==1].copy()



#Labels

labels=data.loc[(data['Target'].notnull()) & (data['parentesco1']==1),['Target','idhogar']]



#Value counts of Target

label_counts=labels['Target'].value_counts().sort_index()



# color mapping

colors=OrderedDict({1:'red',2:'orange',3:'blue',4:'green'})

poverty_mapping=OrderedDict({1:'extreme',2:'moderate',3:'vulnerable',4:'non-vulnerable'})



#visualization of labels distribution

label_counts.plot.bar(figsize=(10,8),color=colors.values(),edgecolor='k',linewidth=3)



#Formatting x-axis and y-axis

plt.xlabel('Poverty Level')

plt.ylabel('Count of housholds')

plt.xticks([x-1 for x in poverty_mapping.keys()],list(poverty_mapping.values()),rotation=60)

plt.title('Poverty Level Breakdown')

print(label_counts)

plt.show()
#group by household id and see target label of each individual in household

#categorise into equal labels and non-equal labels

equal_labels=train.groupby('idhogar')['Target'].apply(lambda x:x.nunique()==1)



non_equal_labels=equal_labels[equal_labels ==False]

print('There are {} number of houselholds who have different target labels'.format(len(non_equal_labels)))
#Example

train[train['idhogar'] == non_equal_labels.index[0]][['idhogar', 'parentesco1', 'Target']]
head_household=train.groupby('idhogar')['parentesco1'].sum()

no_head=train.loc[train['idhogar'].isin(head_household[head_household==0].index),:]

print('No. of households with no head are {}'.format(no_head['idhogar'].nunique()))
no_head_label=no_head.groupby('idhogar')['Target'].apply(lambda x:x.nunique()==1)

print('{} Households with no head have different labels.'.format(sum(no_head_label == False)))
for household in non_equal_labels.index:

    #find correct label

    true_target=int(train[(train['idhogar']==household) &(train['parentesco1']==1)]['Target'])

    

    #set correct label for all member of household

    train.loc[train['idhogar']==household,'Target']=true_target

    

#Check if changes has been implemented

all_equal=train.groupby('idhogar')['Target'].apply(lambda x: x.nunique()==1)



not_equal=all_equal[all_equal==False]



print('No of household with different Target variables are {}'.format(len(not_equal)))
train.isnull().sum().sort_values(0,ascending=False).to_frame().head(10)
print(data.loc[data['rez_esc'].notnull()]['age'].describe())

print(train.loc[train['rez_esc'].notnull()]['age'].describe())

print(data.loc[data['rez_esc'].isnull()]['age'].describe())

print(train.loc[train['rez_esc'].isnull()]['age'].describe())
data.loc[((data['age']<7 )| (data['age']>17))&(data['rez_esc'].isnull()),'rez_esc']=0

train.loc[((train['age']<7 )| (train['age']>17))&(train['rez_esc'].isnull()),'rez_esc']=0
data['missing_value']=data['rez_esc'].isnull()

train['missing_value']=train['rez_esc'].isnull()

#explore any data anamoly

data['rez_esc'].describe()

data.loc[data['rez_esc']>5,'rez_esc']=5

train.loc[train['rez_esc']>5,'rez_esc']=5

data['v18q1']=data['v18q1'].fillna(0)

train['v18q1']=train['v18q1'].fillna(0)
# Variables indicating home ownership

own_variables = [x for x in data if x.startswith('tipo')]





# Plot of the home ownership variables for home missing rent payments

data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
# Fill in households that own the house with 0 rent payment

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

data['v2a1-missing'] = data['v2a1'].isnull()



data['v2a1-missing'].value_counts()
train['v2a1'].isna().sum()
sum_tipo=train['tipovivi1'].sum()+train['tipovivi2'].sum()+train['tipovivi4'].sum()+train['tipovivi5'].sum()
rent_miss=train.loc[(train['tipovivi2']==1)&(train['v2a1'].notna()),:].shape[0]
sum_tipo-rent_miss
rent_miss
train.loc[(train['tipovivi1']==1)&(train['v2a1'].isna()),'v2a1']=0

data.loc[(data['tipovivi1'] == 1) & (data['v2a1'].isna()), 'v2a1'] = 0

train.loc[(train['tipovivi2']==1)&(train['v2a1'].isna()),'v2a1']=0

data.loc[(data['tipovivi2']==1)&(data['v2a1'].isna()),'v2a1']=0

train.loc[(train['tipovivi4']==1)&(train['v2a1'].isna()),'v2a1']=0

data.loc[(data['tipovivi4']==1)&(data['v2a1'].isna()),'v2a1']=0

train.loc[(train['tipovivi5']==1)&(train['v2a1'].isna()),'v2a1']=0

data.loc[(data['tipovivi5']==1)&(data['v2a1'].isna()),'v2a1']=0
train['v2a1'].isna().sum()
for cols in data.columns[1:]:

    if cols in ['idhogar', 'dependency', 'edjefe', 'edjefa']:

        continue

    percentile75 = np.percentile(data[cols].fillna(0), 75)

    percentile25 = np.percentile(data[cols].fillna(0), 25)

    threshold = (percentile75 - percentile25) * 1.5

    lower, upper = (percentile25 - threshold), (percentile75 + threshold)

    # identify outliers

    outliers = data.loc[(data[cols] < lower) & (data[cols] > upper)]

    if len(outliers) > 0:

        print('Feature: {}. Identified outliers: {}'.format(cols, len(outliers)))

for cols in train.columns[1:]:

    if cols in ['idhogar', 'dependency', 'edjefe', 'edjefa']:

        continue

    percentile75 = np.percentile(train[cols].fillna(0), 75)

    percentile25 = np.percentile(train[cols].fillna(0), 25)

    threshold = (percentile75 - percentile25) * 1.5

    lower, upper = (percentile25 - threshold), (percentile75 + threshold)

    # identify outliers

    outliers = train.loc[(train[cols] < lower) & (train[cols] > upper)]

    if len(outliers) > 0:

        print('Feature: {}. Identified outliers: {}'.format(cols, len(outliers)))
elec = []



# Assign values

for i, row in head.iterrows():

    if row['noelec'] == 1:

        elec.append(0)

    elif row['coopele'] == 1:

        elec.append(1)

    elif row['public'] == 1:

        elec.append(2)

    elif row['planpri'] == 1:

        elec.append(3)

    else:

        elec.append(np.nan)

        

# Record the new variable and missing flag

head['elec'] = elec

head['elec-missing'] = head['elec'].isnull()



# Remove the electricity columns

head = head.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
heads = head.drop(columns = 'area2')



heads.groupby('area1')['Target'].value_counts(normalize = True)
# Wall ordinal variable

heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]),

                           axis = 1)
# No toilet, no electricity, no floor, no water service, no ceiling

heads['warning'] = 1 * (heads['sanitario1'] + 

                         (heads['elec'] == 0) + 

                         heads['pisonotiene'] + 

                         heads['abastaguano'] + 

                         (heads['cielorazo'] == 0))
plt.figure(figsize = (10, 6))

sns.violinplot(x = 'warning', y = 'Target', data = heads);

plt.title('Target vs Warning Variable');
# Owns a refrigerator, computer, tablet, and television

heads['bonus'] = 1 * (heads['refrig'] + 

                      heads['computer'] + 

                      (heads['v18q1'] > 0) + 

                      heads['television'])



sns.violinplot('bonus', 'Target', data = heads,

                figsize = (10, 6));

plt.title('Target vs Bonus Variable');
# Use only training data

train_heads = heads.loc[heads['Target'].notnull(), :].copy()



pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target': 'pcorr'}).reset_index()

pcorrs = pcorrs.rename(columns = {'index': 'feature'})



print('Most negatively correlated variables:')

print(pcorrs.head())



print('\nMost positively correlated variables:')

print(pcorrs.dropna().tail())
variables = ['Target', 'dependency', 'warning', 'meaneduc'

             , 'r4m1', 'overcrowding']



# Calculate the correlations

corr_mat = train_heads[variables].corr().round(2)



# Draw a correlation heatmap

plt.rcParams['font.size'] = 18

plt.figure(figsize = (12, 12))

sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 

            cmap = plt.cm.RdYlGn_r, annot = True);
id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone']



ind_ordered = ['rez_esc', 'escolari', 'age']





ind = data[id_ + ind_bool + ind_ordered]

ind.shape
# Create correlation matrix

corr_matrix = ind.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
ind = ind.drop(columns = 'male')
ind[[c for c in ind if c.startswith('instl')]].head()
ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)
plt.figure(figsize = (10, 8))

sns.violinplot(x = 'Target', y = 'inst', data = ind);

plt.title('Education Distribution by Target');
# Drop the education columns

ind = ind.drop(columns = [c for c in ind if c.startswith('instlevel')])

ind.shape
ind['escolari/age'] = ind['escolari'] / ind['age']



plt.figure(figsize = (10, 8))

sns.violinplot('Target', 'escolari/age', data = ind);
ind['inst/age'] = ind['inst'] / ind['age']

ind['tech'] = ind['v18q'] + ind['mobilephone']

ind['tech'].describe()
# Define custom function

range_ = lambda x: x.max() - x.min()

range_.__name__ = 'range_'



# Group and aggregate

ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])

ind_agg.head()
# Rename the columns

new_col = []

for c in ind_agg.columns.levels[0]:

    for stat in ind_agg.columns.levels[1]:

        new_col.append(f'{c}-{stat}')

        

ind_agg.columns = new_col

ind_agg.head()
ind_agg = ind_agg.drop(columns = to_drop)

ind_feats = list(ind_agg.columns)



# Merge on the household id

final = heads.merge(ind_agg, on = 'idhogar', how = 'left')



print('Final features shape: ', final.shape)
final.head()
corrs = final.corr()['Target']
corrs.sort_values().head()
corrs.sort_values().dropna().tail()
plt.figure(figsize = (10, 6))

sns.violinplot(x = 'Target', y = 'escolari-max', data = final);

plt.title('Max Schooling by Target');
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'escolari-max', data = final);

plt.title('Max Schooling by Target');
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'meaneduc', data = final);

plt.xticks([0, 1, 2, 3], poverty_mapping.values())

plt.title('Average Schooling by Target');
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'overcrowding', data = final);

plt.xticks([0, 1, 2, 3], poverty_mapping.values())

plt.title('Overcrowding by Target');
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



# Custom scorer for cross validation

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
# Labels for training

train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))



# Extract the training data

train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



# Submission base which is used for making submissions to the competition

submission_base = test[['Id', 'idhogar']].copy()
features = list(train_set.columns)



pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 

                      ('scaler', MinMaxScaler())])



# Fit and transform training data

train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)
model = RandomForestClassifier(n_estimators=100, random_state=10, 

                               n_jobs = -1)

# 10 fold cross validation

cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')