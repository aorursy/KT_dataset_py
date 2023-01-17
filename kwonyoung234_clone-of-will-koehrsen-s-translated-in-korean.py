import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv")
train.head()
train.info()
test.info()
df_int_cols = train.select_dtypes('int').apply(lambda x:x.nunique()).reset_index()

df_int_cols.columns = ['var_name','count']

df_int_cols.groupby('count')['var_name'].size().plot(kind='bar')



plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
from collections import OrderedDict



colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



fig,ax = plt.subplots(4,2,figsize=[20,16])



for i,col in enumerate(train.select_dtypes('float').columns):

    ax = plt.subplot(4,2,i+1)

    for proverty_level,color in colors.items():

        sns.kdeplot(train.loc[train['Target']==proverty_level,col].dropna(),ax=ax,color=color,label=poverty_mapping[proverty_level])

    

    ax.set(title=f'{col.capitalize()} Distribution',xlabel=f'{col}',ylabel='Density')
train.select_dtypes('object').head()
mapping = {'yes':1,'no':0}



needToChange = train.select_dtypes('object').columns[-1:1:-1]



for df in [train,test]:

    for col in needToChange:

        df[col] = df[col].replace(mapping).astype(np.float64)

        

train.loc[:,needToChange].describe()
fig,ax = plt.subplots(3,1,figsize=[16,18])



for i,col in enumerate(needToChange):

    ax = plt.subplot(3,1,i+1)

    for proverty_level,color in colors.items():

        sns.kdeplot(train.loc[train['Target']==proverty_level,col].dropna(),ax=ax,color=color,label=poverty_mapping[proverty_level])

    

    ax.set(title=f'{col.capitalize()} Distribution',xlabel=f'{col}',ylabel='Density')
# Add null Target column to test

test['Target'] = np.nan

data = train.append(test, ignore_index = True)
heads = data.loc[data['parentesco1'] == 1].copy()



train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1),['Target','idhogar']]



label_counts = train_labels['Target'].value_counts().sort_index()



label_counts.plot(kind='bar',figsize=[8,6],color=colors.values(),edgecolor='k',linewidth = 2)



plt.xlabel('Poverty Level'); plt.ylabel('Count'); 

plt.xticks([x - 1 for x in poverty_mapping.keys()], 

           list(poverty_mapping.values()), rotation = 60)

plt.title('Poverty Level Breakdown');
all_equal = train.groupby('idhogar')['Target'].apply(lambda x:x.nunique() == 1)



not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
train[train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
household_leader = train.groupby('idhogar')['parentesco1'].sum()



household_nohead = train.loc[train['idhogar'].isin(household_leader[household_leader==0].index),:]

print('There are {} households without a head.'.format(household_nohead['idhogar'].nunique()))
household_nohead_equal = household_nohead.groupby('idhogar')['Target'].apply(lambda x:x.nunique() == 1)

print('{} Households with no head have different labels.'.format(sum(household_nohead_equal == False)))
for household in not_equal.index:

    true_label = int(train.loc[(train['idhogar'] == household) & (train['parentesco1'] == 1)]['Target'])

    train.loc[train['idhogar'] == household,'Target'] = true_label

    

all_equal = train.groupby('idhogar')['Target'].apply(lambda x:x.nunique() == 1)

not_equal = all_equal[all_equal != True]



print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
missing = pd.DataFrame(data.isnull().sum()).rename(columns={0:'total'})

missing['percent'] = missing['total'] / len(data)

missing.sort_values(by='percent',ascending=False).head(10).drop('Target')
def plot_value_counts(df,col,head_only=False):

    if head_only:

        df = df.loc[df['parentesco1']==1,col]

    plt.figure(figsize=[8,6])

    df[col].value_counts().sort_index().plot(kind='bar',color='blue',edgecolor='k',linewidth=2)

    plt.xlabel(f'{col}')

    plt.title(f'{col} Value Counts')

    plt.ylabel('Count')

    plt.show()
plot_value_counts(heads,'v18q1')
heads.groupby('v18q')['v18q1'].apply(lambda x:x.isnull().sum())
data['v18q1'] = data['v18q1'].fillna(0)
own_variables = [x for x in data if x.startswith('tipo')]



data.loc[data['v2a1'].isnull(),own_variables].sum().plot(kind='bar',figsize=[10,8],color='green',edgecolor='k',linewidth=2)



plt.xticks([0,1,2,3,4], ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'], rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
data.loc[data['tipovivi1'] == 1,'v2a1'] = 0



data['v2a1-missing'] = data['v2a1'].isnull()

data['v2a1-missing'].value_counts()
data.loc[data['rez_esc'].notnull(),'age'].describe()
data.loc[data['rez_esc'].isnull(),'age'].describe()
# If individual is over 19 or younger than 7 and missing years behind, set it to 0

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc']>5,'rez_esc'] = 5
def plot_categoricals(x, y, data, annotate = True):

    """Plot counts of two categoricals.

    Size is raw count for each grouping.

    Percentages are for a given value of y."""

    

    # Raw counts 

    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))

    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})

    

    # Calculate counts for each group of x and y

    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))

    

    # Rename the column and reset the index

    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()

    counts['percent'] = 100 * counts['normalized_count']

    

    # Add the raw count

    counts['raw_count'] = list(raw_counts['raw_count'])

    

    plt.figure(figsize = (14, 10))

    # Scatter plot sized by percent

    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',

                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',

                alpha = 0.6, linewidth = 1.5)

    

    if annotate:

        # Annotate the plot with text

        for i, row in counts.iterrows():

            # Put text with appropriate offsets

            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 

                               row[y] - (0.15 / counts[y].nunique())),

                         color = 'navy',

                         s = f"{round(row['percent'], 1)}%")

        

    # Set tick marks

    plt.yticks(counts[y].unique())

    plt.xticks(counts[x].unique())

    

    # Transform min and max to evenly space in square root domain

    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))

    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))

    

    # 5 sizes for legend

    msizes = list(range(sqr_min, sqr_max,

                        int(( sqr_max - sqr_min) / 5)))

    markers = []

    

    # Markers for legend

    for size in msizes:

        markers.append(plt.scatter([], [], s = 100 * size, 

                                   label = f'{int(round(np.square(size) / 100) * 100)}', 

                                   color = 'lightgreen',

                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))

        

    # Legend and formatting

    plt.legend(handles = markers, title = 'Counts',

               labelspacing = 3, handletextpad = 2,

               fontsize = 16,

               loc = (1.10, 0.19))

    

    plt.annotate(f'* Size represents raw count while % is for a given y value.',

                 xy = (0, 1), xycoords = 'figure points', size = 10)

    

    # Adjust axes limits

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 

              counts[x].max() + (6 / counts[x].nunique())))

    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 

              counts[y].max() + (4 / counts[y].nunique())))

    plt.grid(None)

    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
plot_categoricals('rez_esc', 'Target', data);
plot_categoricals('escolari', 'Target', data, annotate = False)
plot_value_counts(data[(data['rez_esc-missing'] == 1)], 'Target')
plot_value_counts(data[(data['v2a1-missing'] == 1)], 'Target')
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone', 'rez_esc-missing']



ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_



from collections import Counter



print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))

print('We covered every variable: ', len(x) == data.shape[1])
np.all(np.array(list(Counter(x).values())) == 1)
sns.lmplot('age','SQBage',data=data,fit_reg=False)

plt.title('Squared Age versus Age')
data = data.drop(columns = sqr_)

data.shape
heads = data.loc[data['parentesco1']== 1,:]

heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

heads.shape
corr_matrix = heads.corr()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))



to_drop = [col for col in upper.columns if any(abs(upper[col]) > 0.95)]



to_drop
corr_matrix.loc[corr_matrix['tamhog'].abs()>0.9,corr_matrix['tamhog'].abs()>0.9]
sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],annot=True, cmap = plt.cm.autumn_r, fmt='.3f');
heads = heads.drop(columns = ['tamhog','hogar_total','r4t3'])
sns.lmplot('tamviv','hhsize',data,fit_reg=False,size=8)

plt.title('Household size vs number of persons living in the household');
heads['hhsize-diff'] = heads['tamviv'] - heads['hhsize']

plot_categoricals('hhsize-diff','Target',heads)
corr_matrix.loc[corr_matrix['coopele'].abs() > 0.9,corr_matrix['coopele'].abs() > 0.9]
elec = []



for i,row in heads.iterrows():

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

        

heads['elec'] = elec

heads['elec-missing'] = heads['elec'].isnull()



heads = heads.drop(columns = ['noelec','coopele','public','planpri'])
plot_categoricals('elec','Target',heads)
corr_matrix.loc[corr_matrix['area2'].abs() >0.9, corr_matrix['area2'].abs()>0.9]
heads = heads.drop(columns='area2')



heads.groupby('area1')['Target'].value_counts(normalize=True)
heads['walls'] = np.argmax(np.array(heads[['epared1','epared2','epared3']]),axis=1)

heads.drop(columns=['epared1','epared2','epared3'])



plot_categoricals('walls','Target',heads)
# Roof ordinal variable

heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]),

                           axis = 1)

heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])



# Floor ordinal variable

heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]),

                           axis = 1)

heads = heads.drop(columns = ['eviv1', 'eviv2', 'eviv3'])
heads['wall+roof+floor'] = heads['walls']+heads['roof']+heads['floor']



plot_categoricals('wall+roof+floor','Target',heads,annotate=False)
count = pd.DataFrame(heads.groupby('wall+roof+floor')['Target'].value_counts(normalize=True)).rename(columns={'Target':'Normalized Count'}).reset_index()

count.head()
heads['warning'] = 1 * (heads['sanitario1'] + (heads['elec'] == 0) + heads['pisonotiene'] + heads['abastaguano'] + (heads['cielorazo'] == 0))
plt.figure(figsize = (10, 6))

sns.violinplot(x = 'warning', y = 'Target', data = heads);

plt.title('Target vs Warning Variable');
plot_categoricals('warning', 'Target', data = heads)
# Owns a refrigerator, computer, tablet, and television

heads['bonus'] = 1 * (heads['refrig'] + 

                      heads['computer'] + 

                      (heads['v18q1'] > 0) + 

                      heads['television'])



sns.violinplot('bonus', 'Target', data = heads,

                figsize = (10, 6));

plt.title('Target vs Bonus Variable');
heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']

heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']

heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']

heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
from scipy.stats import spearmanr



def plot_corrs(x,y):

    spr = spearmanr(x,y).correlation

    pcr = np.corrcoef(x,y)[0,1]

    

    data = pd.DataFrame({'x':x,'y':y})

    plt.figure(figsize=[6,4])

    sns.regplot('x','y',data=data,fit_reg=False)

    plt.title(f'Spearman: {round(spr, 2)}; Pearson: {round(pcr, 2)}');
x = np.array(range(100))

y = x ** 2



plot_corrs(x, y)
x = np.array([1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 9])

y = np.array([1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 3, 3, 2, 4, 2, 2, 4])



plot_corrs(x, y)
x = np.array(range(-19, 20))

y = 2 * np.sin(x)



plot_corrs(x, y)
train_heads = heads.loc[heads['Target'].notnull(),:].copy()



pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target':'pcorr'}).reset_index()

pcorrs = pcorrs.rename(columns= {'index':'feature'})



print('Most negatively correlated variables:')

print(pcorrs.head())



print('\nMost positively correlated variables:')

print(pcorrs.dropna().tail())
import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)



feats = []

scorr = []

pvalues = []



# Iterate through each column

for c in heads:

    # Only valid for numbers

    if heads[c].dtype != 'object':

        feats.append(c)

        

        # Calculate spearman correlation

        scorr.append(spearmanr(train_heads[c], train_heads['Target']).correlation)

        pvalues.append(spearmanr(train_heads[c], train_heads['Target']).pvalue)



scorrs = pd.DataFrame({'feature': feats, 'scorr': scorr, 'pvalue': pvalues}).sort_values('scorr')
print('Most negative Spearman correlations:')

print(scorrs.head())

print('\nMost positive Spearman correlations:')

print(scorrs.dropna().tail())
corrs = pcorrs.merge(scorrs, on = 'feature')

corrs['diff'] = corrs['pcorr'] - corrs['scorr']



corrs.sort_values('diff').head()
corrs.sort_values('diff').dropna().tail()
sns.lmplot('dependency', 'Target', fit_reg = True, data = train_heads, x_jitter=0.05, y_jitter=0.05);

plt.title('Target vs Dependency');
sns.lmplot('rooms-per-capita', 'Target', fit_reg = True, data = train_heads, x_jitter=0.05, y_jitter=0.05);

plt.title('Target vs Rooms Per Capita');
variables = ['Target', 'dependency', 'warning', 'wall+roof+floor', 'meaneduc',

             'floor', 'r4m1', 'overcrowding']



# Calculate the correlations

corr_mat = train_heads[variables].corr().round(2)



# Draw a correlation heatmap

plt.rcParams['font.size'] = 18

plt.figure(figsize = (12, 12))

sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 

            cmap = plt.cm.RdYlGn_r, annot = True);
import warnings

warnings.filterwarnings('ignore')



plot_data = train_heads[['Target', 'dependency', 'wall+roof+floor','meaneduc', 'overcrowding']]



grid = sns.PairGrid(data=plot_data,size=4,diag_sharey=False,hue='Target',hue_order=[4,3,2,1],vars = [col for col in list(plot_data.columns) if col != 'Target'])



grid.map_upper(plt.scatter,alpha=0.8,s=20)



grid.map_diag(sns.kdeplot)



grid.map_lower(sns.kdeplot,cmap=plt.cm.OrRd_r)

grid = grid.add_legend()

plt.suptitle('Feature Plots Colored By Target', size = 32, y = 1.05);
household_feats = list(heads.columns)
ind = data[id_ + ind_bool + ind_ordered]

ind.shape
corr_matrix = ind.corr()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))



to_drop = [col for col in upper.columns if any(abs(upper[col]) > 0.95)]



to_drop
ind = ind.drop(columns = 'male')
ind[[col for col in ind.columns if col.startswith('instlevel')]].head()
ind['inst'] = np.argmax(np.array(ind[[col for col in ind.columns if col.startswith('instlevel')]]),axis=1)



plot_categoricals('inst','Target',ind,annotate=False)
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
range_ = lambda x:x.max() - x.min()

range_.__name__ = 'range_'



ind_agg = ind.drop(columns='Target').groupby('idhogar').agg(['min','max','sum','count','std',range_])

ind_agg.head()
new_cols = []



for col in ind_agg.columns.levels[0]:

    for stat in ind_agg.columns.levels[1]:

        new_cols.append(f'{col}-{stat}')

        

ind_agg.columns = new_cols

ind_agg.head()
ind_agg.shape
corr_matrix = ind_agg.corr()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))



to_drop = [col for col in upper.columns if any(abs(upper[col]) > 0.95)]



print(f'There are {len(to_drop)} correlated columns to remove.')
ind_agg = ind_agg.drop(columns=to_drop)

ind_feats = list(ind_agg.columns)



final = heads.merge(ind_agg,on='idhogar',how='left')



print('Final features shape: ', final.shape)
final.head()
corrs = final.corr()['Target']
corrs.sort_values().head()
corrs.sort_values().dropna().tail()
plot_categoricals('escolari-max', 'Target', final, annotate=False);
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
head_gender = ind.loc[ind['parentesco1'] == 1, ['idhogar', 'female']]

final = final.merge(head_gender, on = 'idhogar', how = 'left').rename(columns = {'female': 'female-head'})
final.groupby('female-head')['Target'].value_counts(normalize=True)
sns.violinplot(x = 'female-head', y = 'Target', data = final);

plt.title('Target by Female Head of Household');
plt.figure(figsize = (8, 8))

sns.boxplot(x = 'Target', y = 'meaneduc', hue = 'female-head', data = final);

plt.title('Average Education by Target and Female Head of Household', size = 16);
final.groupby('female-head')['meaneduc'].agg(['mean', 'count'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



# Custom scorer for cross validation

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
train_labels = np.array(list(final.loc[final['Target'].notnull(),'Target'].astype(np.uint8)))



train_set = final[final['Target'].notnull()].drop(columns=['Id','idhogar','Target'])



test_set = final[final['Target'].isnull()].drop(columns=['Id','idhogar','Target'])



submission_base = test[['Id', 'idhogar']].copy()
train_set.head()
features = list(train_set.columns)



pipeline = Pipeline([('imputer',Imputer(strategy='median')),('scaler',MinMaxScaler())])



train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)
model = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs = -1)

# 10 fold cross validation

cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')
model.fit(train_set,train_labels)



indices = np.argsort(model.feature_importances_)[::-1]

importances = model.feature_importances_



for i in range(len(features)):

    print("%d) feature importances of %s %f"%(i,features[indices[i]],importances[indices[i]]))
# def plot_feature_importances(df,feature_label,importance_label,n=10,ascending=False):

    

#     plt.style.use('fivethirtyeight')

    

#     df = df.sort_values(by=importance_label,ascending=ascending).reset_index()

#     df.loc[list(range(n)),][importance_label].plot(kind='bar')

#     plt.xticks(list(range(n)),df.loc[list(range(n)),][feature_label])
def plot_feature_importances(df, n = 10, threshold = None):

    """Plots n most important features. Also plots the cumulative importance if

    threshold is specified and prints the number of features needed to reach threshold cumulative importance.

    Intended for use with any tree-based feature importances. 

    

    Args:

        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

    

        n (int): Number of most important features to plot. Default is 15.

    

        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.

        

    Returns:

        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 

                        and a cumulative importance column

    

    Note:

    

        * Normalization in this case means sums to 1. 

        * Cumulative importance is calculated by summing features from most to least important

        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    

    """

    plt.style.use('fivethirtyeight')

    

    # Sort features with most important at the head

    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    

    # Normalize the feature importances to add up to one and calculate cumulative importance

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    

    plt.rcParams['font.size'] = 12

    

    # Bar plot of n most important features

    df.loc[:n, :].plot.barh(y = 'importance_normalized', 

                            x = 'feature', color = 'darkgreen', 

                            edgecolor = 'k', figsize = (12, 8),

                            legend = False, linewidth = 2)



    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 

    plt.title(f'{n} Most Important Features', size = 18)

    plt.gca().invert_yaxis()

    

    

    if threshold:

        # Cumulative importance plot

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df
temp_df = pd.DataFrame({'feature':features,'importance':importances})
norm_fi = plot_feature_importances(temp_df, threshold=0.95)
def kde_target(df, variable):

    """Plots the distribution of `variable` in `df` colored by the `Target` column"""

    

    colors = {1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}



    plt.figure(figsize = (12, 8))

    

    df = df[df['Target'].notnull()]

    

    for level in df['Target'].unique():

        subset = df[df['Target'] == level].copy()

        sns.kdeplot(subset[variable].dropna(), 

                    label = f'Poverty Level: {level}', 

                    color = colors[int(subset['Target'].unique())])



    plt.xlabel(variable); plt.ylabel('Density');

    plt.title('{} Distribution'.format(variable.capitalize()));
kde_target(final, 'meaneduc')
kde_target(final, 'escolari/age-range_')
# Model imports

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
import warnings 

from sklearn.exceptions import ConvergenceWarning



# Filter out warnings from models

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)



# Dataframe to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(train, train_labels, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
model_results = cv_model(train_set,train_labels,LinearSVC(),'LSVC',model_results)
model_results = cv_model(train_set,train_labels,GaussianNB(),'GNB',model_results)
model_results = cv_model(train_set,train_labels,MLPClassifier(hidden_layer_sizes=(32,64,128,64,32)),'MLP',model_results)
model_results = cv_model(train_set, train_labels, 

                          LinearDiscriminantAnalysis(), 

                          'LDA', model_results)
model_results = cv_model(train_set, train_labels, RidgeClassifierCV(), 'RIDGE', model_results)
for n in [5, 10, 20]:

    print(f'\nKNN with {n} neighbors\n')

    model_results = cv_model(train_set, train_labels, 

                             KNeighborsClassifier(n_neighbors = n),

                             f'knn-{n}', model_results)
from sklearn.ensemble import ExtraTreesClassifier



model_results = cv_model(train_set, train_labels, 

                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),

                         'EXT', model_results)
model_results = cv_model(train_set, train_labels, 

                         RandomForestClassifier(n_estimators = 100, random_state = 10),

                         'RF', model_results)
models = model_results['model']
model_results_ = model_results.set_index('model')

model_results_['cv_mean'].plot(kind='bar',color='orange',figsize=(8,6),edgecolor='k',linewidth=2, yerr=list(model_results['cv_std']))
test_ids = list(final.loc[final['Target'].isnull(),'idhogar'])
def submit(model,train,train_labels,test,test_ids):

    

    model.fit(train,train_labels)

    predictions = model.predict(test)

    predictions = pd.DataFrame({'idhogar':test_ids,'Target':predictions})

    

    submission = submission_base.merge(predictions,on='idhogar',how='left').drop(columns=['idhogar'])

    

    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    

    return submission
rf_submission = submit(RandomForestClassifier(n_estimators = 100, 

                                              random_state=10, n_jobs = -1), 

                         train_set, train_labels, test_set, test_ids)



rf_submission.to_csv('rf_submission.csv', index = False)
train_set = pd.DataFrame(train_set,columns=features)
corr_matrix = train_set.corr()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))



to_drop = [column for column in upper.columns if any(abs(upper[column]) >0.95)]



to_drop
print(train_set.shape)
train_set = train_set.drop(to_drop,axis=1)

print(train_set.shape)
test_set = pd.DataFrame(test_set,columns=features)

print(test_set.shape)

train_set, test_set = train_set.align(test_set,axis=1,join='inner')

features = list(train_set.columns)

print(test_set.shape)
from sklearn.feature_selection import RFECV



estimator = RandomForestClassifier(random_state=10,n_estimators=100,n_jobs=-1)



selector = RFECV(estimator,step=1,cv=5,scoring=scorer,n_jobs=-1)
selector.fit(train_set,train_labels)
plt.plot(selector.grid_scores_)



plt.xlabel('Number of Features')

plt.ylabel('Macro F1 Score')

plt.title('Feature Selection Scores')

selector.n_features_
rankings = pd.DataFrame({'feature':list(train_set.columns),'rank':list(selector.ranking_)}).sort_values(by='rank')

rankings.head(10)
train_selected = selector.transform(train_set)

test_selected = selector.transform(test_set)
train_selected
# np.where(selector.ranking_==1)
selected_features = train_set.columns[np.where(selector.ranking_==1)]

train_selected = pd.DataFrame(train_selected,columns=selected_features)

test_selected = pd.DataFrame(test_selected,columns=selected_features)
train_selected.shape
model_results = cv_model(train_selected,train_labels,model,'RF-SEL',model_results)
model_results.set_index('model', inplace = True)

model_results_ = model_results



model_results_['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                 edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');

model_results.reset_index(inplace = True)
def macro_f1_score(labels, predictions):

    # Reshape the predictions as needed

    # 행으로만 이루어진 결과치를 4열씩 차례대로 배치하고 그 4열에서 max를 각각 추출

    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)

    

    metric_value = f1_score(labels, predictions, average = 'macro')

    

    # Return is name, value, is_higher_better

    return 'macro_f1', metric_value, True
# from sklearn.model_selection import StratifiedKFold

# import lightgbm as lgb



# feature_names = features



# params = {

#     'boosting_type':'dart',

#     'colsample_bytree':0.88,

#     'learning_rate':0.028,

#     'min_child_samples':10,

#     'num_leaves':36, 'reg_alpha':0.76,

#     'reg_lambda':0.43,

#     'subsample_for_bin':40000,

#     'subsample':0.54,

#     'class_weight':'balanced'

# }



# model = lgb.LGBMClassifier(**params,objective='multiclass',n_jobs=-1,n_estimators=1000,random_state=10)



# SKFOLD = StratifiedKFold(n_splits=5,shuffle=True)



# predictions = pd.DataFrame()

# importances = np.zeros(len(feature_names))



# features = np.array(train_set)

# test_features = np.array(test_set)

# labels = np.array(train_labels).reshape((-1))



# valid_scores = []



# for i,(train_indices,valid_indices) in enumerate(SKFOLD.split(features,labels)):

    

#     fold_predictions = pd.DataFrame()

    

#     X_train,X_valid = features[train_indices],features[valid_indices]

#     y_train,y_valid = labels[train_indices],labels[valid_indices]

    

#     model.fit(X_train,y_train,early_stopping_rounds=100,eval_metric=macro_f1_score,eval_set=[(X_train,y_train),(X_valid,y_valid)],eval_names=['train','valid'],verbose=200)

    

#     valid_scores.append(model.best_score_['valid']['macro_f1'])

    

#     fold_probabilities = model.predict_proba(test_features)

    

#     for j in range(4):

#         fold_predictions[(j+1)] = fold_probabilities[:,j]

        

#     fold_predictions['idhogar'] = test_ids

#     fold_predictions['fold'] = (i+1)

    

#     predictions = predictions.append(fold_predictions)

    

#     importances += model.feature_importances_ / 5

    

#     display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')

    

#     predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

#     predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

    

#     submission = submission_base.merge(predictions[['idhogar','Target']],on='idhogar',how='left').drop(columns=['idhogar'])

#     submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

#     break

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from IPython.display import display



def model_gbm(features, labels, test_features, test_ids, 

              nfolds = 5, return_preds = False, hyp = None):

    """Model using the GBM and cross validation.

       Trains with early stopping on each fold.

       Hyperparameters probably need to be tuned."""

    

    feature_names = list(features.columns)



    # Option for user specified hyperparameters

    if hyp is not None:

        # Using early stopping so do not need number of esimators

        if 'n_estimators' in hyp:

            del hyp['n_estimators']

        params = hyp

    

    else:

        # Model hyperparameters

        params = {'boosting_type': 'dart', 

                  'colsample_bytree': 0.88, 

                  'learning_rate': 0.028, 

                   'min_child_samples': 10, 

                   'num_leaves': 36, 'reg_alpha': 0.76, 

                   'reg_lambda': 0.43, 

                   'subsample_for_bin': 40000, 

                   'subsample': 0.54, 

                   'class_weight': 'balanced'}

    

    # Build the model

    model = lgb.LGBMClassifier(**params, objective = 'multiclass', 

                               n_jobs = -1, n_estimators = 10000,

                               random_state = 10)

    

    # Using stratified kfold cross validation

    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)

    

    # Hold all the predictions from each fold

    predictions = pd.DataFrame()

    importances = np.zeros(len(feature_names))

    

    # Convert to arrays for indexing

    features = np.array(features)

    test_features = np.array(test_features)

    labels = np.array(labels).reshape((-1 ))

    

    valid_scores = []

    

    # Iterate through the folds

    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):

        

        # Dataframe for fold predictions

        fold_predictions = pd.DataFrame()

        

        # Training and validation data

        X_train = features[train_indices]

        X_valid = features[valid_indices]

        y_train = labels[train_indices]

        y_valid = labels[valid_indices]

        

        # Train with early stopping

        model.fit(X_train, y_train, early_stopping_rounds = 100, 

                  eval_metric = macro_f1_score,

                  eval_set = [(X_train, y_train), (X_valid, y_valid)],

                  eval_names = ['train', 'valid'],

                  verbose = 200)

        

        # Record the validation fold score

        valid_scores.append(model.best_score_['valid']['macro_f1'])

        

        # Make predictions from the fold as probabilities

        fold_probabilitites = model.predict_proba(test_features)

        

        # Record each prediction for each class as a separate column

        for j in range(4):

            fold_predictions[(j + 1)] = fold_probabilitites[:, j]

            

        # Add needed information for predictions 

        fold_predictions['idhogar'] = test_ids

        fold_predictions['fold'] = (i+1)

        

        # Add the predictions as new rows to the existing predictions

        predictions = predictions.append(fold_predictions)

        

        # Feature importances

        importances += model.feature_importances_ / nfolds   

        

        # Display fold information

        display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')



    # Feature importances dataframe

    feature_importances = pd.DataFrame({'feature': feature_names,

                                        'importance': importances})

    

    valid_scores = np.array(valid_scores)

    display(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')

    

    # If we want to examine predictions don't average over folds

    if return_preds:

        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

        return predictions, feature_importances

    

    # Average the predictions over folds

    predictions = predictions.groupby('idhogar', as_index = False).mean()

    

    # Find the class and associated probability

    predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

    predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

    predictions = predictions.drop(columns = ['fold'])

    

    # Merge with the base to have one prediction for each individual

    submission = submission_base.merge(predictions[['idhogar', 'Target']], on = 'idhogar', how = 'left').drop(columns = ['idhogar'])

        

    # Fill in the individuals that do not have a head of household with 4 since these will not be scored

    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    

    # return the submission and feature importances along with validation scores

    return submission, feature_importances, valid_scores
predictions, gbm_fi = model_gbm(train_set,train_labels,test_set,test_ids,return_preds=True)
predictions.head()
plt.rcParams['font.size'] = 18



# Kdeplot

g = sns.FacetGrid(predictions, row = 'fold', hue = 'Target', size = 3, aspect = 4)

g.map(sns.kdeplot, 'confidence');

g.add_legend();



plt.suptitle('Distribution of Confidence by Fold and Target', y = 1.05);
plt.figure(figsize = (24, 12))

sns.violinplot(x = 'Target', y = 'confidence', hue = 'fold', data = predictions);
predictions.loc[predictions['idhogar'] == '000a08204',:]
predictions.groupby('idhogar', as_index = False).mean().head()
# Average the predictions over folds

predictions = predictions.groupby('idhogar', as_index = False).mean()



# Find the class and associated probability

predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

predictions = predictions.drop(columns = ['fold'])
predictions.loc[predictions['idhogar'] == '000a08204',:]
# Plot the confidence by each target

plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'confidence', data = predictions);

plt.title('Confidence by Target');



plt.figure(figsize = (10, 6))

sns.violinplot(x = 'Target', y = 'confidence', data = predictions);

plt.title('Confidence by Target');
%%capture

submission, gbm_fi, valid_scores = model_gbm(train_set, train_labels, 

                                             test_set, test_ids, return_preds=False)



submission.to_csv('gbm_baseline.csv')
_ = plot_feature_importances(gbm_fi, threshold=0.95)
submission, gbm_fi_selected, valid_scores_selected = model_gbm(train_selected, train_labels, 

                                                               test_selected, test_ids)
model_results
model_results = model_results.append(pd.DataFrame({'model': ["GBM", "GBM_SEL"], 

                                                   'cv_mean': [valid_scores.mean(), valid_scores_selected.mean()],

                                                   'cv_std':  [valid_scores.std(), valid_scores_selected.std()]}),

                                                sort = True)

model_results.reset_index(drop=True)
model_results_ = model_results.set_index('model')

model_results_['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                 edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');

# model_results.reset_index(inplace = True)
submission, gbm_fi, valid_scores = model_gbm(train_set, train_labels, test_set, test_ids, 

                                             nfolds=10, return_preds=False)
submission.to_csv('gbm_10fold.csv', index = False)
submission, gbm_fi_selected, valid_scores_selected = model_gbm(train_selected, train_labels, test_selected, test_ids,

                                                               nfolds=10)
submission.to_csv('gmb_10fold_selected.csv', index = False)
model_results
model_results = model_results.append(pd.DataFrame({'model': ["GBM_10Fold", "GBM_10Fold_SEL"], 

                                                   'cv_mean': [valid_scores.mean(), valid_scores_selected.mean()],

                                                   'cv_std':  [valid_scores.std(), valid_scores_selected.std()]}),

                                    sort = True)
model_results_ = model_results.set_index('model')

model_results_['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                 edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');
model_results
print(f"There are {gbm_fi_selected[gbm_fi_selected['importance'] == 0].shape[0]} features with no importance.")
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

from hyperopt.pyll.stochastic import sample
import csv

import ast

from timeit import default_timer as timer
def objective(hyperparameters,nfolds=5):

    

    global ITERATION

    ITERATION += 1

    

    subsample = hyperparameters['boosting_type'].get('subsample',1.0)

    subsample_freq = hyperparameters['boosting_type'].get('subsample_freq',0)

    

    boosting_type = hyperparameters['boosting_type']['boosting_type']

    

    if boosting_type == 'dart':

        hyperparameters['drop_rate'] = hyperparameters['boosting_type']['drop_rate']

        

    hyperparameters['subsample'] = subsample

    hyperparameters['subsample_freq'] = subsample_freq

    hyperparameters['boosting_type'] = boosting_type

    

    if not hyperparameters['limit_max_depth']:

        hyperparameters['max_depth'] = -1

        

    for parameter_name in ['max_depth','num_leaves','subsample_for_bin','min_child_samples','subsample_freq']:

        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

        

    if 'n_estimators' in hyperparameters:

        del hyperparameters['n_estimators']

    

    strkfold = StratifiedKFold(n_splits=nfolds, shuffle=True)

    

    features = np.array(train_selected)

    # ????

    labels = np.array(train_labels).reshape((-1))

    

    valid_scores = []

    best_estimators = []

    run_times =[]

    

    model = lgb.LGBMClassifier(**hyperparameters,class_weight='balanced',n_jobs=-1,metric='None',n_estimators=10000)

    

    for i,(train_indices,valid_indices) in enumerate(strkfold.split(features,labels)):

        

        X_train,X_valid = features[train_indices],features[valid_indices]

        y_train,y_valid = labels[train_indices],labels[valid_indices]

        

        start = timer()

        

        model.fit(X_train,y_train,early_stopping_rounds=100,eval_metric=macro_f1_score,eval_set=[(X_train,y_train),(X_valid,y_valid)],eval_names=['train','valid'],verbose=400)

        

        end = timer()

        

        valid_scores.append(model.best_score_['valid']['macro_f1'])

        best_estimators.append(model.best_iteration_)

        

        run_times.append(end-start)

        

    score = np.mean(valid_scores)

    score_std = np.std(valid_scores)

    loss = 1 - score

    

    run_time = np.mean(run_times)

    run_time_std = np.std(run_times)

    

    estimators = int(np.mean(best_estimators))

    hyperparameters['n_estimators'] = estimators

    

    of_connection = open(OUT_FILE,'a')

    writer = csv.writer(of_connection)

    writer.writerow([loss,hyperparameters,ITERATION,run_time,score,score_std])

    of_connection.close()

    

    if ITERATION % PROGRESS == 0:

        display(f'Iteration: {ITERATION}, Current Score: {round(score, 4)}.')

    

    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,

            'time': run_time, 'time_std': run_time_std, 'status': STATUS_OK, 

            'score': score, 'score_std': score_std}
# Define the search space

space = {

    'boosting_type': hp.choice('boosting_type', 

                              [{'boosting_type': 'gbdt', 

                                'subsample': hp.uniform('gdbt_subsample', 0.5, 1),

                                'subsample_freq': hp.quniform('gbdt_subsample_freq', 1, 10, 1)}, 

                               {'boosting_type': 'dart', 

                                 'subsample': hp.uniform('dart_subsample', 0.5, 1),

                                 'subsample_freq': hp.quniform('dart_subsample_freq', 1, 10, 1),

                                 'drop_rate': hp.uniform('dart_drop_rate', 0.1, 0.5)},

                                {'boosting_type': 'goss',

                                 'subsample': 1.0,

                                 'subsample_freq': 0}]),

    'limit_max_depth': hp.choice('limit_max_depth', [True, False]),

    'max_depth': hp.quniform('max_depth', 1, 40, 1),

    'num_leaves': hp.quniform('num_leaves', 3, 50, 1),

    'learning_rate': hp.loguniform('learning_rate', 

                                   np.log(0.025), 

                                   np.log(0.25)),

    'subsample_for_bin': hp.quniform('subsample_for_bin', 2000, 100000, 2000),

    'min_child_samples': hp.quniform('min_child_samples', 5, 80, 5),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0)

}
sample(space)
algo = tpe.suggest
trials = Trials()



OUT_FILE = 'optimization.csv'

of_connection = open(OUT_FILE,'w')

writer = csv.writer(of_connection)



MAX_EVALS = 100

PROGRESS = 10

N_FOLDS = 5

ITERATION = 0



# Write column names

headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score', 'std']

writer.writerow(headers)

of_connection.close()
display("Running Optimization for {} Trials.".format(MAX_EVALS))



# Run optimization

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

            max_evals = MAX_EVALS)
import json



# Save the trial results

with open('trials.json', 'w') as f:

    f.write(json.dumps(str(trials)))
results = pd.read_csv(OUT_FILE).sort_values('loss', ascending = True).reset_index()

results.head()
plt.figure(figsize = (8, 6))

sns.regplot('iteration', 'score', data = results);

plt.title("Optimization Scores");

plt.xticks(list(range(1, results['iteration'].max() + 1, 3)));
best_hyp = ast.literal_eval(results.loc[0, 'hyperparameters'])

best_hyp
submission, gbm_fi, valid_scores = model_gbm(train_selected, train_labels, 

                                             test_selected, test_ids, 

                                             nfolds = 10, return_preds=False)



model_results = model_results.append(pd.DataFrame({'model': ["GBM_OPT_10Fold_SEL"], 

                                                   'cv_mean': [valid_scores.mean()],

                                                   'cv_std':  [valid_scores.std()]}),

                                    sort = True).sort_values('cv_mean', ascending = False)
submission, gbm_fi, valid_scores = model_gbm(train_set, train_labels, 

                                             test_set, test_ids, 

                                             nfolds = 10, return_preds=False)



model_results = model_results.append(pd.DataFrame({'model': ["GBM_OPT_10Fold"], 

                                                   'cv_mean': [valid_scores.mean()],

                                                   'cv_std':  [valid_scores.std()]}),

                                    sort = True).sort_values('cv_mean', ascending = False)
model_results.head()
submission, gbm_fi, valid_scores = model_gbm(train_selected, train_labels, 

                                             test_selected, test_ids, 

                                             nfolds = 10, return_preds=False)
submission.to_csv('gbm_opt_10fold_selected.csv', index = False)
_ = plot_feature_importances(gbm_fi)
preds = submission_base.merge(submission,on='Id',how='left')

preds = pd.DataFrame(preds.groupby('idhogar')['Target'].mean())



fig,axes = plt.subplots(1,2,sharey=True,figsize=[12,6])

heads['Target'].sort_index().plot.hist(normed=True,edgecolor=r'k',linewidth=2,ax=axes[0])

axes[0].set_xticks([1,2,3,4])

axes[0].set_xticklabels(poverty_mapping.values(),rotation=60)

axes[0].set_title('Train Label Distribution')





preds['Target'].sort_index().plot.hist(normed = True, 

                                       edgecolor = 'k',

                                       linewidth = 2,

                                       ax = axes[1])

axes[1].set_xticks([1, 2, 3, 4]);

axes[1].set_xticklabels(poverty_mapping.values(), rotation = 60)

plt.subplots_adjust()

plt.title('Predicted Label Distribution');
heads['Target'].value_counts()
preds['Target'].value_counts()