# Library we need 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure Visualization Defaults
mpl.style.use('ggplot')
pylab.rcParams['figure.figsize'] = 12, 8
sns.set_style('white')
pd.set_option('display.max_columns', None)
# Replace the following two directories with those in the following comments
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

# inconsistent columns so we 
# use concat, rather than pd.merge(df_train, df_test, on = [...], how = 'inner')
df_all = pd.concat([df_train, df_test], join='outer', axis=0) 
df_train.name = 'Training data'
df_test.name = 'Test data'

# We have 11 features and 1 target variables
print(df_train.info())
# Dataset Dimensions
print(f'Number of Training Examples: {df_train.shape[0]}')
print(f'Number of Test Examples = {df_test.shape[0]}')
print(f'Shape of Training Examples = {df_train.shape}')
print(f'Shape of Test Examples = {df_test.shape}')
# Column name we have
print(sorted(df_train.columns.tolist())) 
print(sorted(df_test.columns.tolist()))
# Numeric variables in training set
print(df_train.describe(include=[np.number]).T)
print('-' * 30)
# Categorical variables in training set
print(df_train.describe(include=['O']).T)
# Missing values
def displayMissing(df):
    for col in df.columns.tolist():
        print(f'{col:11s} NO. missing values: {df[col].isnull().sum()}')

for i, df in enumerate([df_train, df_test]):
    print(f'{df.name}')
    displayMissing(df)
    if i == 0: print('-' * 30) 
# todo:  Ticket combination is the feature without any missing values.
# We should try to extract any information from it
# although it appears useless at the first glance.
def getTicketPrefixAndNumber(df, col):
    # naming the columns to be created
    col_num = col + '_num'
    col_alp = col + '_alp'

    # get the last group of contiguous digits
    # vectorize string function with str method
    # get any contignuous nuemrical digits from the end
    # return anything that matches the pattern specified inside the parenthesis 
    df[col_num] = df[col].str.extract(r'(\d+)$')
    df[col_num].fillna(-1, inplace=True)

    # get the complete string before a space that is followed by a trailing number group
    df[col_alp] = df[col].str.extract(r'(.*)\ \d+$')
    # sidenote: .replace({'\.': '', '/': ''}, regex=True)
    df[col_alp].fillna('M', inplace=True)
    return df

print(getTicketPrefixAndNumber(df_all, 'Ticket').head())
print(df_all.Ticket_num.describe())
print('-' * 30) 
df_all['Ticket_num'] = pd.to_numeric(df_all['Ticket_num'])
print(df_all.Ticket_num.describe())
# check to see if the string decomposition works as expected.
colnames = ['Ticket' + s for s in ['', '_num', '_alp']]
print(df_all[colnames].head())
islice = df_all.groupby('Ticket_num')['Ticket_alp'].transform(lambda x: x.nunique() > 1)
print(df_all.loc[islice,:].sort_values(by=['Ticket_num']).head(6))
gtb1 = df_all[['Survived', 'Ticket']].groupby(['Ticket'])
# compute the groupwise survival rates (percentage) # pd.count() counts non-NA cells
temp = (gtb1['Survived'].sum() / gtb1['Survived'].count() * 100).sort_values()
# name the resulting column to be used in the merge below
temp.name = 'TeamSurvivalRate'
# one-to-many merge on column Ticket
df_all = pd.merge(df_all, temp, on='Ticket')
df_all.loc[df_all.groupby('Ticket')['Survived'].transform('size') > 5, :]['Ticket'].unique()
print(df_all.loc[df_all.Ticket=='S.O.C. 14879',['Ticket', 'Survived']])
print(df_all.loc[df_all['Ticket'] == 'PC 17608',:])
print(df_all.loc[df_all['Ticket'] == 'CA. 2343',:])
# If use it rather than `pd.size()` to count group size on Survived feature, 
# the resulting group size is not correct.
df_all.loc[df_all.groupby('Ticket')['Survived'].transform('count') > 5, :]['Ticket'].unique()
df_all['TeamSize'] = df_all.groupby('Ticket')['Survived'].transform(lambda x: x.shape[0]) # ~ x.size
# The correct size of a family
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
# To have a sense of the distribution of TeamSize
g = df_all.groupby(['Ticket']).first()['TeamSize'].value_counts()
ans = g / g.sum() * 100 # convert to percentrage
# Tabularize
ans.index = ans.index.astype(int)
ans.index.name = 'Group Size'
print(pd.DataFrame(ans).rename(columns={'TeamSize': 'Percentage'}).T)
df_all.groupby(['TeamSize','Ticket'])['TeamSurvivalRate'].mean().groupby(level=0).mean()
df_all.groupby(['Pclass', 'TeamSize'])['TeamSurvivalRate'].mean()
# This code results in the sorting of Teamsize within passenger class by team survival rate.
print(df_all.groupby(['Pclass', 'TeamSize'])['TeamSurvivalRate'].mean(). \
    reset_index().sort_values(by=['Pclass','TeamSurvivalRate']))
# We can further obtain the class-wise top three groups in terms of team survival rate.
# Just append "groupby(['Pclass']).tail(3)".
print(df_all.groupby(['Pclass', 'TeamSize'])['TeamSurvivalRate'].mean(). \
    reset_index().sort_values(by=['Pclass','TeamSurvivalRate']).groupby(['Pclass']).tail(3))
def getCredibilitySurvivalRate(df):
    # Use `size` or `shape[0]` to get the full length of a series
    # When summing over cells of {0,1,NaN}, 
    # regardless of the use of `pd.notnull()`, the outcome is identical.
    df['SRcredibility'] = pd.notnull(df['Survived']).sum() / df['Survived'].size 
    return df

df_all = df_all.groupby('Ticket').apply(getCredibilitySurvivalRate)
def getRole(df, cutoff=7):
    df['Role'] = 'Man'
    df.loc[df['Sex'] == 'female', 'Role'] = 'Woman'
    df.loc[df['Age'] <= cutoff, 'Role'] = 'Child'
    return df

ans = []
ages = range(1, 30)
for cut in ages:
    getRole(df_all, cutoff=cut)
    g = df_all.groupby(['Role'])
    # [1] I covert the resulting Pandas series to a data frame object and then append
    # it to the `ans` list object so that I can concatenate them in one step. later.
    ans.append((g['Survived'].sum() / g['Survived'].count()).to_frame())
# [2] To concatnate the data frames stored in the `ans` list object.
temp = pd.concat(ans, axis=1)  # 3 by N (=len(ages))
# [3] convert 3 by N table to N by 3 table
tb1 = pd.DataFrame(np.array(temp).T,
                   columns=['Child', 'Man', 'Woman'], index=ages)  # N by 3
tb1.index.name = 'Age'
# [4] melt the table tb1 for drawing a line plot
tb1.reset_index(inplace=True)  # prep for melt
tb2 = pd.melt(tb1, id_vars=['Age'], value_vars=['Child', 'Man', 'Woman'],
              var_name='Role', value_name='Survival')
# [5] FacetGrid and mapping plot functions to each grid
g = sns.FacetGrid(tb2, col='Role', margin_titles=True)
g = g.map(plt.plot, 'Age', 'Survival')

# add vertical line
axes = g.fig.axes
for ax in axes:
    ax.vlines(x=15, ymax=1, ymin=0, linestyles='dashed', alpha=0.3, colors='blue')
plt.show()
getRole(df_all, cutoff=15);
# Use this as an example: df_all.loc[df_all['Ticket_num']==17608,:]
# Step 1. Create a new column for the number of siblings of a child.
df_all['childSibSp'] = np.where(df_all['Role'] == 'Child', df_all['SibSp'], 0)
# Step 2. Is the childSibSp value unique within a travel group? Answer: Yes
logic = df_all['childSibSp']>0 # screen out parents (whose SibSp is at least one) 
df_all.loc[logic,:].groupby('Ticket')['childSibSp'].nunique().value_counts()
# Step 3. Broadcasting: in a group, let every instance of the same group shares the 'childSibSp' value of the youngest child.
df_all['childSibSp'] = df_all.groupby('Ticket')['childSibSp'].transform('max') # 'cz otherwise it's 0 by default.
# Step 4. If an example's SibSp is not equal to the shared value, the instance must be
# a elder child whose age exceeds the age limit for the child definition.
# Condition 1: to prevent updating value of members in travel group without children. Seems redundant (see condition 2).
# Condition 2: 'cz siblings share childSibSp value by construction.
# Condition 3: to prevent the youngest child's value being overwritten.

logic = (df_all['SibSp'] != 0) & \
        (df_all['SibSp'] == df_all['childSibSp']) & \
        (df_all['Role'] != 'Child') 
df_all.loc[logic, 'Role'] = 'olderChild'
print(df_all.loc[df_all['Ticket'] == 'PC 17608',['PassengerId', 'Survived', 'Name', 'Sex', 'Age', 'SibSp', 'Ticket', 'Role', 'childSibSp']])
# Step 1. Identify the FamilySize value of the youngest child in each travel team.
df_all['childFamilySize'] = np.where(df_all['Role'].isin(['Child', 'olderChild']),
                                     df_all['FamilySize'], 
                                     0)
# Step 2. Broadcasting the FamilySize value of the youngest child to
# other members in the same travel team; by doing so, maid and bultler'r role will stay put
# (either Man or Woman as it was).
df_all['childFamilySize'] = df_all.groupby('Ticket')['childFamilySize'].transform('max')
# Step 3. identy people who are parents
def isMotherOrFather(s):
    return 'Father' if s == 'Man' else 'Mother'

# Condition 1: narrow down the scope to family member only; 
# maid and bultlers are going to be screened out.
# Condition 2: narrow down to people whose role is not "child" or "elder child".
slice_logic = ((df_all['FamilySize'] == df_all['childFamilySize']) & \
               (~df_all['Role'].isin(['Child', 'olderChild'])))

# A trick to obtain the index of valid examples after logical operations.
# However, we may as well substitute slice_index with slice_ligic in this case.
slice_index = df_all.loc[slice_logic, :].index
df_all.loc[slice_index, 'Role'] = \
    df_all.loc[slice_logic, :]['Role'].apply(isMotherOrFather)
df_all['ChildWAdult'] = 'Not Applicable'
logic = (df_all['Role'].isin(['Child', 'olderChild']))
df_all.loc[logic, 'ChildWAdult'] = np.where(
    df_all.loc[logic, 'FamilySize'] > df_all.loc[logic, 'childSibSp'] + 1,
    'Yes',
    'No'
)
cols = ['PassengerId', 'Survived', 'Name', 'Sex', 'Age', 'SibSp', 'Ticket', 'Role', 'ChildWAdult']
print(df_all.loc[df_all['Ticket'] == 'PC 17608', cols])
print(df_all.loc[df_all.ChildWAdult=='No', cols].sort_values(by=['Ticket']).head(10))
# age imputation
logic = df_all['PassengerId'] == 241
df_all.loc[logic, 'Age'] = 14.5
logic = df_all['PassengerId'].isin([10, 831])
df_all.loc[logic, 'Role'] = 'Woman'
logic = df_all['PassengerId'].isin([621, 123])
df_all.loc[logic, 'Role'] = 'Man'
print(df_all.loc[df_all['Ticket']=='2666',cols])
# todo: How many children in this group?
# Method 1.
df_all['NumChild'] = df_all.groupby('Ticket')['Role'].\
    transform(lambda x: x.isin(['Child', 'olderChild']).sum())
df_all['NumYoungChild'] = df_all.groupby('Ticket')['Role'].\
    transform(lambda x: x.isin(['Child']).sum())

# # This works as well.
# #Method 2.
# numChildDict = df_all.groupby('Ticket')['Age']\
#     .apply(lambda x: (x <= cutoffChildAge).sum()).reset_index(name='NumChild')
# df_all.join(numChildDict, on='Ticket')

# # Comment out this block 'cz the conclusion seems shaky.

# # Although the survival rate by number of child varies, the highest survival rate estimate
# # falls in family of three children and it holds across classes.
# print(df_all.groupby(['Pclass', 'NumChild'])['Survived'].mean())
# # So, number of children should be another good predictor;
# # In passenger classes 1 and 2, families with three children have higher
# # estimated survival rates.

# Method 1. The usual method for broadcasting simple logical operations.

df_all['hasMother'] = df_all.groupby('Ticket')['Role'].\
    transform(lambda x: (x == 'Mother').sum() > 0)
df_all['hasFather'] = df_all.groupby('Ticket')['Role'].\
    transform(lambda x: (x == 'Father').sum() > 0)
df_all['hasBothParents'] = np.where(
    df_all['hasMother'] & df_all['hasFather'],
    True,
    False
)
# Method 2. Advanced Method: more complicated logical operations
logics = [(df_all['hasFather'] & ~df_all['hasMother'], 'with Father'),
          (df_all['hasFather'] & df_all['hasMother'], 'with Both'),
          (~df_all['hasFather'] & df_all['hasMother'], 'with Mother'),
          (~df_all['hasFather'] & ~df_all['hasMother'], 'without Parents')]
for logic, s in logics:
    ans = df_all.loc[df_all['Age'] <= 17, :].loc[logic, :].\
        groupby('Ticket')['Survived'].mean().mean()
    print(f"{s:>15} {ans: .2f}")
# Sanity check for groups with only male household head.
index = df_all.loc[(df_all['hasFather'])&~(df_all['hasMother']),:].index
df_all.loc[index,:].groupby('Ticket').groups.keys()
print(df_all.loc[df_all['Ticket']=='230080', cols])

# # See this link: https://medium.com/@ODSC/creating-if-elseif-else-variables-in-python-pandas-7900f512f0e4

# # Method 1. Not exactly what I want. It only identify the entry of mother, but
# # I want the result to be broadcasted to everyone in the same group.
# role = 'Mother'
# #Step 1. Identify a mother's life status
# conditions = [(df_all['Role'] != role), # 'Not applicable'
#               (df_all['Survived'].isnull()), # 'Unknown'
#               (df_all['Survived'] == 1.0), # 'Yes'
#               (df_all['Survived'] == 0.0) # 'No'
#               ]
# choices = ['Not applicable', 'Unknown', 'Yes', 'No']
# df_all['isMotherSurvived'] = np.select(conditions, choices)
# len(df_all[df_all['Role']=='Mother']['isMotherSurvived'].values) # 44 mothers on board

# # Step 2. "Braodcast" the group-specific result to other group members
# # Method 1. Use map + dictionary
# index = df_all[df_all['Role']=='Mother']['Ticket_num'] # step 1-1. get the index
# ss = pd.Series(df_all[df_all['Role']=='Mother']['isMotherSurvived'].values, index=index) # step 1-2. get the value
# df_all['isMotherSurvived'] = df_all['Ticket_num'].map(ss.to_dict()).fillna('not applicable') # step 1-3. use dictionary to update the rest

# # Method 2.
# # https://stackoverflow.com/questions/56708924/broadcast-value-to-dataframe-group-by-condition
# df_all['test3'] = df_all['isMotherSurvived'].where(df_all['Role'].eq('Mother'))\
#     .groupby(df_all['Ticket_num']).transform('first').fillna('not applicable')

role = 'Mother'
def isSurvived(df_all, role='Mother'):
    # Step 1. Identify a mother's life status
    conditions = [(df_all['Role'] != role),  # 'Not applicable'
                  (df_all['Survived'].isnull()),  # 'Unknown'
                  (df_all['Survived'] == 1.0),  # 'Yes'
                  (df_all['Survived'] == 0.0)  # 'No'
                  ]
    choices = ['Not applicable', 'Unknown', 'Yes', 'No']
    s = 'is' + role + 'Survived'
    df_all[s] = np.select(conditions, choices)
    df_all[s] = df_all[s].where(df_all['Role'].eq(role)) \
        .groupby(df_all['Ticket_num']).transform('first').fillna('not applicable')
    return df_all
roles = ['Mother', 'Father']
for role in roles:
    isSurvived(df_all, role=role)
def getCabinPrefix(df):
    # 'M' is assigned to missing values
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    return df
getCabinPrefix(df_all)
df_all['Deck'].unique()

# # def imputeCabinePrefix(df_all):
# df_all['Deck'].value_counts()

# # Check to see if the 2nd half of the combined table are all NaN Survived data
# # (1) iloc works with slicing that includes right endpoint.
# # (2) iloc works with index only, so even though I need 'Survived, I use it separately.
# # (3) isnull() to see if there is any missing value
# df_all.iloc[df_train.shape[0]:, ]['Survived'].isnull().all()

# # Cabin numbers have clusters
# df_all['Cabin'].value_counts()
# # For example, 'B57 B59 B63 B66' corresponds to five persons
# # in the Ryerson family. People in the same cabin share the same
# # Ticket_alp and Ticket_num. These three variables should be highly
# # correlated.
# df_all.loc[df_all['Cabin'] == 'B57 B59 B63 B66']
# # 'B57 B59 B63 B66' maps to Ticket_alp = 'PC', which is a much larger group.
# df_all.loc[df_all['Ticket_alp'] == 'PC']['Survived'].sum()

# # We may check later whether each group can be identified or associated with higher servival rate
# # We may also check to see if couples have higher survival rates
# # Check Family Ryerson. The number of SibSp and Parch might have more information.

def getLastNameAndTitle(df):
    # (1) https://docs.python.org/2/library/re.html
    # (2) Why this patterns works? See the [reason](https://shorturl.at/uAEM8).
    # (3) This pattern works as well r'^([^,]*)'
    # See the reference [link](https://shorturl.at/dwJMS)
    df['LastName'] = df['Name'].str.extract(r'^(.+?),')
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].\
        str.split('.', expand=True)[0]
    return df
getLastNameAndTitle(df_all);
print(df_all.loc[df_all['Ticket_num']==33112, cols])

# # Sanity check

# cols = ['Name', 'Title', 'LastName']
# # df_all[cols] works as well
# # colon cannot be ignored in df_all.loc[:,cols]
# df_all.loc[:, cols]

# # finding: People with the same surname may come from different families, for example,
# # check the group of surname 'Davies' we found Ticket #48871 corresponds
# # to three young men; ticket #33112 corresponds to one women of age 48 and
# # and a child with the same surname of age 8. However, an issue is found
# # that just using LastName is not sufficient to locate people of the same
# # family. For example, the record of the woman with Ticket #33112 shows
# # she comes with her two children. By slicing with Ticket #33112, we found the
# # woman indeed has two children whose surnames are different. So, we should only
# # use Ticket_num instead of LastName to identify people traveling together.
# df_all.loc[df_all['LastName'] == 'Davies', :].sort_values(by=['Ticket_num'])

df_all.groupby('Ticket')['Ticket'].\
    apply(lambda x: (x == 'Mrs').sum()).reset_index(name='MrsCount')['MrsCount'].value_counts()
# Example of displaying group results
gs = df_all.groupby('Ticket')
type(gs.indices) 
# Method 1. Peek the grouped data by sampling; so only part of the data
# Use of random.sample() and list(map())

import random
sampled_group_key = random.sample(gs.groups.keys(), 100)
group_list = list(map(lambda gpkey: gs.get_group(gpkey), sampled_group_key))
for i, g in enumerate(group_list):
    if len(g) > 1:
        temp = g.loc[:,cols] # Just treat g as a general data frame
        break
print(temp)
# Method 2. scan through the groups
for i, g in gs.groups.items():
    if len(g) > 1:
        temp = gs.get_group(i).loc[:,cols]
        break
print(temp)
# I found: the record of the Christy indicates there are two children but only one shown.
print(df_all.loc[df_all['LastName'] == 'Christy', cols])        
#Broadcasting
#Identify travelling groups with children among which who are parents?
# 
def addMaxParchMinSibSp(grp):
    return pd.Series(
        [grp['Parch'].max(), grp['SibSp'].min()],
        ['maxParch', 'minSibSp']
    )
# JOIN versue MERGE [link](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
df_all = df_all.join(df_all.groupby('Ticket').apply(addMaxParchMinSibSp), on='Ticket')
print(df_all.loc[:, cols + ['maxParch', 'minSibSp']].sort_values(by=['Ticket']).head(8))


# # LOGICAL OPERATIONS
# # Method 1. dataframe.where()
# df_all['isMother'] = False
# df_all['isMother'] = df_all['isMother'].\
#     where((df_all['Title'] != 'Mrs') | (df_all['hasMaster'] != True), True) # If the condition doesn't hold, then ...

# # Method 2. np.where() # If the condition holds, then ...
# df_all['MotherWithMaster'] = np.where( 
#     (df_all['Title'] == 'Mrs') & (df_all['hasMaster'] == True), 
#     True, 
#     False
# )

# df_all.loc[df_all['Title'] == 'Master', :][['Age', 'Survived']].mean()  # 5.48, 57%

# # Method 3. np.logical_and()
# def MWM(df):
#     return df.apply(lambda x: 1 if 
#         np.logical_and(x['Title'] == 'Mrs', x['Sex'] == 'female') 
#         else 0, axis=1) # axis=1 is key to this implementation
# df_all['test'] = MWM(df_all)
# df_all.head(10)

# # BROADCASTING OPERATIONS 
# # Identify teams travelling with mother and children
# # Method 1. use `transform`

# temp1 = df_all.groupby(['Ticket'])['Title'].\
#     transform(lambda x: x.eq('Master').any())
# temp2 = df_all.groupby(['Ticket'])['MotherWithMaster'].\
#     transform(lambda x: x.eq(True).any())
# df_all['GroupWMomChild'] = temp1 & temp2

# # Method 2-A. use apply-turned `dictionary` and `map` IT WORKS!!

# # temp5 = df_all.groupby('Ticket').apply(
# #       lambda x: x['Title'].eq('Master').any() & x['MotherWithMaster'].eq(True).any())
# # df_all['GroupWMomChild_3'] = df_all['Ticket'].map(temp5)

# # Method 2-B. use apply and merge IT WORKS!!

# # temp3 = df_all.groupby(['Ticket']).apply(lambda x: x['Title'].eq('Master').any())
# # temp4 = df_all.groupby(['Ticket']).apply(lambda x: x['MotherWithMaster'].eq(True).any())
# # df_all.merge((temp3 & temp4).reset_index(), how='left').rename(columns={0: 'GroupWMomChild_2'})

# Impute missing EMBARKED values using K nearest neighbors algorithm
# They more likely board on the ship at port S -- Theory 1.
print(df_all.loc[df_all['Embarked'].isnull(), ['Embarked', 'Fare'] + cols])
print(df_all.loc[df_all['Ticket_num'].between(100000, 125000)]['Embarked'].value_counts())  # S
print(df_all.loc[df_all['Fare'].between(60, 100)]['Embarked'].value_counts())  # S
df_all.groupby(['Pclass', pd.cut(df_all['Fare'],range(50,100,15))])['Embarked'].\
    apply(lambda x: x.value_counts().nlargest(3))
# Step 1. identifying data index corresponding to valid 'Embarked' data.
index = df_all['Embarked'].isnull()

# We are comfortable to only use three features to predict missing value.
_dfAll = df_all.loc[:, ['Embarked', 'Pclass', 'Ticket_alp', 'Ticket_num']].copy()
# Step 2. labeling and normalizing the feature matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
encoder = LabelEncoder()
minmax_scale = MinMaxScaler(feature_range=(0,1))
# Step 2-A. encoding columns 'Pclass' and 'Ticket_alp'
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

for i in range(1,4):
    temp = encoder.fit_transform(_dfAll.iloc[:,i]).reshape(-1,1)
    _dfAll.iloc[:, i] = minmax_scale.fit_transform(temp)
# Our feature matrix consists of `Pclass`, `Ticket_alp`, and `Ticket_num`.
_xtrain = _dfAll.loc[~index,_dfAll.columns[1:4]]
_ytrain = encoder.fit_transform(_dfAll.loc[~index, 'Embarked'])
# Step 3. prediction with k nearest neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knc = KNeighborsClassifier(3, weights='distance')
trained_knc = knc.fit(_xtrain, _ytrain)
predicted_embarked_missing = trained_knc.predict(_dfAll.loc[index, _dfAll.columns[1:4]])
# update the missing value with what we just obtained.
df_all.loc[index,'Embarked'] = encoder.inverse_transform(predicted_embarked_missing) # S
# which instance has missing Fare?
print(df_all.loc[df_all['Fare'].isnull(),:]) # index = 973
plt.figure(figsize=(12,4))
islice = (df_all['Pclass'] == 3)
sns.scatterplot(x='Age', y='Fare', size= 'TeamSize', data=df_all.loc[islice,:]); plt.show()

df_all['Fare'] = df_all.groupby(['Pclass','TeamSize'])['Fare']\
    .transform(lambda x: x.fillna(x.median()))
print(df_all.loc[973, cols + ['Fare']])
cols = ['Survived', 'Pclass', 'Ticket', 'Fare', 'Role', 'Age', 'Name']
# The Davies has two children and two adults (one is maid). The youngest child is alive.
print(df_all.loc[df_all['Ticket_num'] == 33112, cols])
print(df_all.loc[df_all['Ticket_num'] == 2079, cols])
print(df_all.loc[df_all['Ticket_num'] == 36928, cols])  # old family with adult female child
print(df_all.loc[df_all['Ticket_num'] == 236853, cols])  # couples without children
print(df_all.loc[df_all['Ticket_num'] == 17608, cols])  # size of 6
print(df_all.loc[df_all['Ticket_num'] == 3101295, cols]) # travel group with female household head
print(df_all.loc[df_all['Ticket_num'] == 2144, cols]) # Class 3. mother with six children died
print(df_all.loc[df_all['Ticket_num'] == 347742, cols]) # class 3. mother with two children alive
print(df_all.loc[df_all['Ticket_num'] == 347077, cols])  # class 3 mother with five children alive
print(df_all.loc[df_all['Ticket_num'] == 1601, cols]) # class 3. group of eight non-biological-related male members
for i, g in df_all.groupby(['Pclass', 'Sex'])['Age'].groups.items():
    if i == (1, 'female'):
        tb = df_all.groupby(['Pclass', 'Sex'])['Age'].get_group(i)
        break
print(pd.DataFrame(tb).head(10).T)
import string
alpha_list = list(string.ascii_lowercase[:9])
# Seaborn facet grid plot
g = sns.FacetGrid(df_all, col='Pclass', row='Sex')
g = g.map(plt.hist, 'Age', bins=np.arange(0, 80, 5), density=True)

for ax in g.axes.flat:
    # Step 1. Get section
    texts = ax.get_xticklabels()
    locations = ax.get_xticks()
    ylabel = ax.yaxis.get_label()
    # Step 2. Set section
    if ylabel:
        ax.set_ylabel("Percentage")
    ax.set_xticks(np.arange(0,81,10))
    # ax.set_xticklabels(alpha_list) # experimental. works well.
    ax.set_xticklabels(np.arange(0, 81,10))

plt.show()
c = df_all.groupby(['Pclass', 'Sex'])['Survived'].sum().astype(int).rename("count")
c
d = df_all.groupby(['Pclass', 'Sex'])['Survived'].apply(lambda x: pd.notnull(x).sum())
print(d)
print(d.groupby(level=0).sum())
c / d
e = c / d.groupby(level=0).sum()
e
params = {'figure.figsize': (15, 6),
          'axes.labelsize': 18,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'legend.title_fontsize': 12,
          'legend.fontsize': 12,
          'legend.handlelength': 4}
pylab.rcParams.update(params)

fig, ax = plt.subplots()
f = e.unstack(level=-1)
f.plot(kind='bar', stacked=True, fontsize=16, ax=ax)

fig.set_size_inches(4,5)
ax.set_xlabel('Passenger class')
ax.set_ylabel('Percentage')
ax.set_title('Survivor count in each class')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
grouped = df_all.groupby(['Pclass', 'Sex', pd.cut(df_all['Age'],bins=range(0,80,10))])
unpivoted = grouped['Survived'].apply(lambda x: x.sum()).reset_index()
print(unpivoted.head(14).T)
# Part I. major visualization
g = sns.FacetGrid(unpivoted, col='Pclass', hue='Sex', margin_titles=True)
g = g.map(sns.barplot, 'Age', 'Survived', alpha=0.4)
# Part II.  
g.add_legend()
for i, ax in enumerate(g.axes.flat):
    # Step 1. get information
    labels = ax.get_xticklabels() # get x labels
    ylabel = ax.yaxis.get_label()
    # Step 2. update information
    if i==0: ax.set_ylabel('Survivor count')
#     for i,l in enumerate(labels):
#         if(i%2 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
# Step 1. Generate the head count by Pclas, Sex and age group
grouped = df_all.groupby(['Pclass', 'Sex', pd.cut(df_all['Age'],bins=range(0,80,10))])
x = grouped['Survived'].apply(lambda x: x.sum()).reset_index()
x

# Step 2. unstack Sex so that we have male and female columns
y = x.set_index(['Pclass', 'Age', 'Sex']).unstack(level=-1)
# drop the column index before summation
y.columns = y.columns.droplevel()
y['total'] = y['female'] + y['male']
y

# Step 3. drop the intermediate column and then rename
# Note: the value of female column is actually the sum of the original numbers of female and male.
del y['female']
y = y.rename(columns={'total': 'female'})
unpivoted = y.fillna(0)

# Step 4. Transform back to a data frame with Pclass, Age, Sex, Survived columns
unpivoted = unpivoted.stack()
unpivoted = unpivoted.to_frame(name='Survived')
unpivoted
unpivoted = unpivoted.reset_index()
unpivoted;
unpivoted.info()
# Step 5.
g = sns.FacetGrid(unpivoted, col='Pclass', hue='Sex', margin_titles=True)
g = g.map(sns.barplot, 'Age', 'Survived', alpha=0.5)
# Part II.
g.add_legend()
for i, ax in enumerate(g.axes.flat):
    # Step 1. get information
    labels = ax.get_xticklabels() # get x labels
    ylabel = ax.yaxis.get_label()
    # Step 2. update information
    if i==0: ax.set_ylabel('Survivor count')
#     for i,l in enumerate(labels):
#         if(i%2 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.show()
def sbp(*args, **kwargs):
    data = kwargs.pop('data')
    sns.barplot(x = data.Age, y = data.loc[data.Sex == 'female', 'Survived'], color = 'red', label = 'Female')
    sns.barplot(x = data.Age, y = data.loc[data.Sex == 'male', 'Survived'], color = 'blue', label = 'Male') 
    
unpivoted['Age'] = unpivoted['Age'].astype(str) # This is the key statement for the following mapping works.
                                                # It lets you to convert categorical to str.
    
g = sns.FacetGrid(unpivoted, col='Pclass', margin_titles=True, legend_out=True)
g = g.map_dataframe(sbp, Age='Age', Sex='Sex')

g.add_legend()
for i, ax in enumerate(g.axes.flat):
    # Step 1. get information
    labels = ax.get_xticklabels() # get x labels
    ylabel = ax.yaxis.get_label()
    # Step 2. update information
    if i==0: ax.set_ylabel('Survivor count')
#     for i,l in enumerate(labels):
#         if(i%2 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
    ax.set_xlabel('Age')
plt.show()