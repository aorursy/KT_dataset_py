

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.shape,test.shape
train.head()
train.dtypes.value_counts()
train.select_dtypes(include=['object']).head()
test['Target']=0

train['is_train']=1

test['is_train']=0
#del train2

train2=train.copy()
train2=train2.append(test,ignore_index=True)

train2['is_train'].value_counts()
train2.columns[train2.isnull().any()]
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
missing_values = missing_values_table(train2)
missing_values.head(20)
del train2['rez_esc']
del train2['v18q1']
del train2['v2a1']


train2['meaneduc'].fillna((train2['meaneduc'].mean()), inplace=True)
train2['SQBmeaned'].fillna((train2['SQBmeaned'].mean()), inplace=True)
dependency=pd.DataFrame()
dependency['dependency']=train2['dependency'].loc[train2['dependency']!=('yes')]
dependency['dependency']=dependency['dependency'].loc[dependency['dependency']!=('no')]

dependency['dependency'].astype('float64').mean()
train2.loc[train2['dependency']=='yes','dependency'] = 1.59
train2.loc[train2['dependency']=='no','dependency'] = 0
train2['dependency']=train2['dependency'].astype('float64')
dependency['edjefe']=train2['edjefe'].loc[train2['edjefe']!=('yes')]
dependency['edjefe']=dependency['edjefe'].loc[dependency['edjefe']!=('no')]
dependency['edjefe'].astype('float64').mean()

train2.loc[train2['edjefe']=='yes','edjefe'] = 8.54
train2.loc[train2['edjefe']=='no','edjefe'] = 0
train2['edjefe']=train2['edjefe'].astype('float64')
dependency['edjefa']=train2['edjefa'].loc[train2['edjefa']!=('yes')]
dependency['edjefa']=dependency['edjefa'].loc[dependency['edjefa']!=('no')]
dependency['edjefa'].astype('float64').mean()
train2.loc[train2['edjefa']=='yes','edjefa'] = 8.47
train2.loc[train2['edjefa']=='no','edjefa'] = 0
train2['edjefa']=train2['edjefa'].astype('float64')
train2.select_dtypes(include=['object']).head()
train2.dtypes.value_counts()
train=train2.loc[train2['is_train']==1]
test=train2.loc[train2['is_train']==0]

train.shape,test.shape
train['Target'].astype('int64').plot.hist()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
train['Target'].value_counts(normalize=True).plot(ax=ax, kind='bar')
train.describe()
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import  f1_score

def f1_macro(y_true, y_pred): return f1_score(y_true, y_pred, average='macro')
def print_score(m):
    res = [f1_macro(m.predict(X_train), y_train), f1_macro(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
from sklearn.model_selection import train_test_split
df=train.copy()
y=df['Target']
del df['Target']
del df['Id']
del df['idhogar']
X_train, X_valid, y_train, y_valid = train_test_split(
 df, y, test_size=0.33, random_state=42)
m =RandomForestClassifier(random_state=2,n_jobs=-1,criterion="gini" )

%time m.fit(X_train, y_train)
print_score(m)
feature_importances = pd.DataFrame(m.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
#Which are the top 10 factors influencing the vulnerability of a family
feature_importances.head(10)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)

grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'meaneduc',normed=True, bins=np.linspace(0, 40, 15));
grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'SQBmeaned',normed=True, bins=np.linspace(0, 40, 15));
grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'dependency',normed=True, bins=np.linspace(0, 40, 15));
train2=train.copy()

#First lets try to get a mobiles used per person which should be highly predictive of financial well being
train2['Tot_persons']=train2['overcrowding']*train2['bedrooms']
train2['mob_perperson']=train2['qmobilephone']/train2['Tot_persons'] #This has a higher correlation than individual variables it uses
#Can we merge both the Education and overcrowding together?

train['Edu_crwd_ratio']=train['meaneduc']/train['overcrowding']

data = train2[['Target','Edu_crwd_ratio','Tot_persons','mob_perperson', 'dependency', 'SQBmeaned', 'meaneduc','qmobilephone','overcrowding','SQBhogar_nin','edjefe','escolari','SQBovercrowding']]
data_corrs = data.corr()
data_corrs
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['dependency', 'SQBmeaned', 'meaneduc']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(train.loc[train['Target'] == 1, source], label = 'target == 1')
    # plot loans that were not repaid
    sns.kdeplot(train.loc[train['Target'] == 2, source], label = 'target == 2')
    
    sns.kdeplot(train.loc[train['Target'] == 3, source], label = 'target == 3')
    
    sns.kdeplot(train.loc[train['Target'] == 4, source], label = 'target == 4')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
test.head()
del test['Target']
test_df=test.copy()
del test_df['Id']
del test_df['idhogar']
Target=m.predict(test_df)
test.tail()
pd.options.mode.chained_assignment = None
test['Target'] = Target

test[['Id', 'Target']].to_csv('submission.csv', index= False)
