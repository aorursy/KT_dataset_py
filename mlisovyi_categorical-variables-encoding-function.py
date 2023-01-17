from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])
import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
cat_cols = train.select_dtypes(include=['object', 'category'])
cat_cols.head(10)
def plot_value_counts(series, title=None):
    '''
    Plot distribution of values counts in a pd.Series
    '''
    _ = plt.figure(figsize=(12,6))
    z = series.value_counts()
    sns.barplot(x=z, y=z.index)
    _ = plt.title(title)
    
plot_value_counts(train['edjefe'], 'Value counts of edjefe')
plot_value_counts(train['edjefa'], 'Value counts of edjefa')
# Family member counts
hogar_df = train[[f_ for f_ in train.columns if f_.startswith('hogar_')]]
# Family member type of this person
parentesco_df = train[[f_ for f_ in train.columns if f_.startswith('parentesco')]]

# Family status dataset
family_status = pd.concat([cat_cols, train[['female', 'male', 'parentesco1', 'meaneduc']], hogar_df], axis =1)
family_status.head(5)
# this is a hand-picked example to illustrate the point
family_status.query('idhogar == "2b58d945f"')
# this is a hand-picked example to illustrate the point
family_status.query('idhogar == "200099351"')
family_status.query('(edjefe=="no" & edjefa=="no")').head()
family_status.query('(edjefe=="yes") | (edjefa=="yes")').head(20)
plot_value_counts(train['dependency'], 'Value counts of dependency')
train[['dependency', 'SQBdependency']].head(20)

