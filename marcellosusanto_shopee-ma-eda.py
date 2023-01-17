# Setting package umum 
import pandas as pd
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import time
import tensorflow as tf
%matplotlib inline

from matplotlib.pylab import rcParams
# For every plotting cell use this
# grid = gridspec.GridSpec(n_row,n_col)
# ax = plt.subplot(grid[i])
# fig, axes = plt.subplots()
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:.5f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
### Install packages
!pip install -U ppscore
### Load dataset
df_train = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/train.csv')
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/test.csv')
df_user = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/users.csv')
# Sumber : https://www.kaggle.com/kabure/eda-feat-engineering-encode-conquer
def count_pcg_plot(df, var, target, ax, bar_color, line_color, text_size, y2_label, adjust_height=1000) :

    # Bikin countplotnya
    ax = sns.countplot(data=df, x=var, color=bar_color, order=df[var].sort_values().unique())
    ax.set_title('Information of '+var, fontsize=20, fontname='Monospace', fontweight="bold")

    # Buat jadi dua y-axis
    ax2 = ax.twinx()

    # Hitung persentase target tiap value di variabel
    tmp = pd.crosstab(df[var], df[target], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

    # Buat lineplotnya
    ax2 = sns.pointplot(x=var, y='Yes', data=tmp
                     ,order=list(tmp[var].values)
                     ,color=line_color, legend=False, scale=0.5)
    ax2.set_ylabel(y2_label, fontname='Monospace')

    # Kosmetiknya
    total = len(df)
    height_plus = 0.01*total
    sizes = []
    for p in ax.patches :
        height = p.get_height()
        sizes.append(height)
        ax.text(p.get_x()+p.get_width()/2., height + adjust_height,
              '{:1.2f}%'.format(height/total*100),
              ha="center", fontsize=text_size, fontname='Monospace')
    ax.set_ylim(0, max(sizes)*1.2) ;
# Function to make a donut chart
def make_donut_chart(sizes, labels, colors=None, explode=None) :
    '''
    Make a donut chart

    Args :
    - sizes (list) : Proporsi ukuran tiap class
    - labels (list) : Nama tiap class
    - colors (list) : Hexcode untk tiap class
    - explode (list) : Untuk membuat donut yang misah tiap class
    '''

    # Buat plot
    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)

    # Buat lingkaran dalam
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Kasih detail tambahan
    plt.axis('equal')  
    plt.tight_layout()
### Overview data train
df_train.head(11)
### Overview data test
df_test.head(11)
### Overview data user
df_user.head(11)
# Percentage of open flag
rcParams['figure.figsize'] = [7,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Plot prep
sizes = df_train['open_flag'].value_counts() / len(df_train) * 100
labels = ['Not Open','Open']
colors = ['#4285F4','#EA4335']
explode_donut = [0.05, 0.1]

# Plot
make_donut_chart(sizes, labels, colors, explode_donut)
plt.title('Percentage of open flag', fontsize=17, fontname='Monospace', fontweight="bold") ;
# Function to give quick general information about the dataset
def dataset_summary(df) :
    '''
    Give quick feneral information about the dataset such as dtpes, missing values, and unique values

    Args :
    - df (pd.DataFrame) : Dataset

    Return :
    - summary_df (pd.DataFrame) : Contain general information
    '''

    # Make summary dataframe
    summary_df = pd.DataFrame()

    # Input the characteristic of the dataset
    summary_df['Var'] = df.columns
    summary_df['Dtypes'] = df.dtypes.values
    summary_df['Total Missing'] = df.isnull().sum().values
    summary_df['Missing%'] = summary_df['Total Missing'] / len(df) * 100
    summary_df['Total Unique'] = df.nunique().values
    summary_df['Unique%'] = summary_df['Total Unique'] / len(df) * 100

    # Dataset dimension
    print('Dataset dimension :',df.shape)

    return summary_df
### Summary of data
dataset_summary(df_train)
### Are all the row with missing "attr_1" is the same row as missing "age"
dataset_summary(df_user[df_user['attr_1'].isna()])
### Check wether all row in train dataset can be merge with data user
df_comb_train = df_train.merge(df_user, on='user_id')

print('Total row on data train :',len(df_train))
print('After merge :',len(df_comb_train))
### Check wether all row in test dataset can be merge with data user
df_comb_test = df_test.merge(df_user, on='user_id')

print('Total row on data test :',len(df_test))
print('After merge :',len(df_comb_test))
### Check the distribution of missing attr_1 and not
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,1)

# Dataset prep
df_plot = df_comb_train.copy()
df_plot['attr1_is_nan'] = df_plot['attr_1'].isna()

# Plot prep
line_color = '#EA4335'
bar_color = '#4285F4'
text_size = 12
y2_label = 'open_flag_rate'
list_var = ['attr1_is_nan']

# Plot
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    count_pcg_plot(df_plot, var, 'open_flag', ax, bar_color, line_color, text_size, y2_label)
### Check the distribution of attr var
rcParams['figure.figsize'] = [10,15]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(3,1)

# Dataset prep
df_plot = df_comb_train.copy()

# Plot prep
line_color = '#EA4335'
bar_color = '#4285F4'
text_size = 12
y2_label = 'open_flag_rate'
list_var = ['attr_1','attr_2','attr_3']

# Plot
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    count_pcg_plot(df_plot, var, 'open_flag', ax, bar_color, line_color, text_size, y2_label)
    
plt.tight_layout()
### Remove attr_2
df_comb_train.drop(columns=['attr_2'], inplace=True)
df_comb_test.drop(columns=['attr_2'], inplace=True)
### Fill NaN values for attr_1
df_comb_train['attr_1'] = df_comb_train['attr_1'].fillna(1)
df_comb_test['attr_1'] = df_comb_test['attr_1'].fillna(1)
### Check the distribution of domain
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,1)

# Dataset prep
df_plot = df_comb_train.copy()

# Plot prep
line_color = '#EA4335'
bar_color = '#4285F4'
text_size = 12
y2_label = 'open_flag_rate'
list_var = ['domain']

# Plot
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    count_pcg_plot(df_plot, var, 'open_flag', ax, bar_color, line_color, text_size, y2_label)
    
plt.tight_layout()
### Make 'domain_type'
list_low_domain = ['@163.com','@gmail.com','@yahoo.com','@ymail.com']
list_med_domain = ['@outlook.com','@qq.com','@rocketmail.com']
list_high_domain = ['@hotmail.com','@icloud.com','@live.com','other']

def make_domain_type(dom) :
    if dom in list_low_domain :
        res = 'low_domain'
    elif dom in list_med_domain :
        res = 'med_domain'
    elif dom in list_high_domain :
        res = 'high_domain'
        
    return res

df_comb_train['domain_type'] = df_comb_train.apply(lambda x : make_domain_type(x['domain']), axis=1)
df_comb_test['domain_type'] = df_comb_test.apply(lambda x : make_domain_type(x['domain']), axis=1)
### Check age variable
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,1)

# Plot prep
df_plot = df_comb_train.copy()
list_var = ['age']
df_open = df_plot[df_plot['open_flag']==1]
df_not_open = df_plot[df_plot['open_flag']==0]

# Prop
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    sns.distplot(df_open[var], color='#EA4335', ax=ax, label='Open', hist=False)
    sns.distplot(df_not_open[var], color='#4285F4', ax=ax, kde_kws={'alpha':0.5}, label='Not Open', hist=False)
    plt.legend()

# Additional cosmetics
plt.plot([30,30],[0,0.06], '--', color='#6a737b')
plt.text(x=32,y=0.057, s='Age 30')
plt.title('Distribution of age', fontsize=20, fontname='Monospace', fontweight="bold")
plt.tight_layout() ;
### Use PPS Score to see which variable have predictive power to impute "age"
import ppscore as pps
pps.predictors(df_comb_train.dropna(subset=['age']), "age")
### Make "age_class"
def make_age_class(dataset) :
    df = dataset.copy()
    
    # For NaN values
    df['age_class'] = df['age'].isna()
    df['age_class'] = df['age_class'].map({True:'Unknown',False:'<>'})
    
    # Make class for '>=30' and '<30' age
    df.loc[df['age']>=30, 'age_class'] = '>=30'
    df.loc[df['age']<30, 'age_class'] = '<30'
    
    return df

df_comb_train = make_age_class(df_comb_train)
df_comb_test = make_age_class(df_comb_test)
### Correlation of open, login, and checkout day with target
rcParams['figure.figsize'] = [15,10]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Data Prep
list_col = pd.Series(df_comb_train.columns)
bool_open = list_col.str.contains('open')
bool_login = list_col.str.contains('login')
bool_checkout = list_col.str.contains('checkout')
col_to_plot = list(list_col[bool_login]) + list(list_col[bool_checkout]) + list(list_col[bool_open])

df_plot = df_comb_train[col_to_plot]

df_plot = df_plot[df_plot['last_open_day'].str.isnumeric()]
df_plot = df_plot[df_plot['last_login_day'].str.isnumeric()]
df_plot = df_plot[df_plot['last_checkout_day'].str.isnumeric()]

df_plot['last_open_day'] = df_plot['last_open_day'].astype('int')
df_plot['last_login_day'] = df_plot['last_login_day'].astype('int')
df_plot['last_checkout_day'] = df_plot['last_checkout_day'].astype('int')
corr = df_plot.corr()

# Plot Prep
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot
sns.heatmap(corr, square =True, vmin=-1, vmax=1, linewidths=0.1, annot=True, fmt='.2f', mask=mask, cmap='Pastel1_r') ;
plt.title('Correlation of open, login, and checkout') ; 
### PCA for last n days variable
from sklearn.decomposition import PCA
pca = PCA(n_components=1)

list_open_var = ['open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days']
list_login_var = ['login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days']
list_checkout_var = ['checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days']
dict_var = {'open':list_open_var, 'login':list_login_var, 'checkout':list_checkout_var}

# Do PCA
for name,var in dict_var.items() :
    
    # Fit PCA
    pca.fit(df_comb_train[var])
    print(name,':',pca.explained_variance_ratio_)
    
    # Make new var
    df_comb_train[name+'_count'] = pca.transform(df_comb_train[var])
    df_comb_test[name+'_count'] = pca.transform(df_comb_test[var])

### Make new variable using division operator
df_comb_train['open_per_login'] = df_comb_train['open_count'] / df_comb_train['login_count']
df_comb_train.loc[df_comb_train['login_count']==0, 'open_per_login'] = 0

df_comb_train['open_per_checkout'] = df_comb_train['open_count'] / df_comb_train['checkout_count']
df_comb_train.loc[df_comb_train['checkout_count']==0, 'open_per_checkout'] = 0

df_comb_test['open_per_login'] = df_comb_test['open_count'] / df_comb_test['login_count']
df_comb_test.loc[df_comb_test['login_count']==0, 'open_per_login'] = 0

df_comb_test['open_per_checkout'] = df_comb_test['open_count'] / df_comb_test['checkout_count']
df_comb_test.loc[df_comb_test['checkout_count']==0, 'open_per_checkout'] = 0
### Check new variable disribution
rcParams['figure.figsize'] = [10,8]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(3,2)

# Plot prep
df_plot = df_comb_train.copy()
var_to_plot = ['open_count','login_count','checkout_count','open_per_login','open_per_checkout']
df_open = df_plot[df_plot['open_flag']==1]
df_not_open = df_plot[df_plot['open_flag']==0]

# Prop
for i,var in enumerate(var_to_plot) :
    ax = plt.subplot(grid[i])
    sns.distplot(df_open[var], color='#EA4335', ax=ax, label='Open', hist=False)
    sns.distplot(df_not_open[var], color='#4285F4', ax=ax, kde_kws={'alpha':0.5, 'bw': 0.1}, label='Not Open', hist=False)
    plt.legend()

# Additional cosmetics
plt.tight_layout() ;
### Make variable that represent if the user have ever open, check, checkout
def make_var_check(var):
    if var.isnumeric():
        return 'Yes'
    else:
        return 'No'
    
df_comb_train['last_open_check'] = df_comb_train.apply(lambda x : make_var_check(x['last_open_day']), axis=1)
df_comb_train['last_login_check'] = df_comb_train.apply(lambda x : make_var_check(x['last_login_day']), axis=1)
df_comb_train['last_checkout_check'] = df_comb_train.apply(lambda x : make_var_check(x['last_checkout_day']), axis=1)

df_comb_test['last_open_check'] = df_comb_test.apply(lambda x : make_var_check(x['last_open_day']), axis=1)
df_comb_test['last_login_check'] = df_comb_test.apply(lambda x : make_var_check(x['last_login_day']), axis=1)
df_comb_test['last_checkout_check'] = df_comb_test.apply(lambda x : make_var_check(x['last_checkout_day']), axis=1)
### Check the distribution of country
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,1)

# Dataset prep
df_plot = df_comb_train.copy()

# Plot prep
line_color = '#EA4335'
bar_color = '#4285F4'
text_size = 12
y2_label = 'open_flag_rate'
list_var = ['country_code']

# Plot
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    count_pcg_plot(df_plot, var, 'open_flag', ax, bar_color, line_color, text_size, y2_label)
    
plt.tight_layout()
### Get variable from date
df_comb_train['grass_date'] = pd.to_datetime(df_comb_train['grass_date'])
df_comb_test['grass_date'] = pd.to_datetime(df_comb_test['grass_date'])

df_comb_train['day'] = df_comb_train['grass_date'].dt.day.astype('category')
df_comb_train['dayofweek'] = df_comb_train['grass_date'].dt.dayofweek.astype('category')
df_comb_train['month'] = df_comb_train['grass_date'].dt.month.astype('category')

df_comb_test['day'] = df_comb_test['grass_date'].dt.day.astype('category')
df_comb_test['dayofweek'] = df_comb_test['grass_date'].dt.dayofweek.astype('category')
df_comb_test['month'] = df_comb_test['grass_date'].dt.month.astype('category')
### Check the distribution of attr var
rcParams['figure.figsize'] = [15,15]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(3,1)

# Dataset prep
df_plot = df_comb_train.copy()

# Plot prep
line_color = '#EA4335'
bar_color = '#4285F4'
text_size = 12
y2_label = 'open_flag_rate'
list_var = ['day','dayofweek','month']

# Plot
for i,var in enumerate(list_var) :
    ax = plt.subplot(grid[i])
    count_pcg_plot(df_plot, var, 'open_flag', ax, bar_color, line_color, text_size, y2_label)
    
plt.tight_layout()
### Change day into numeric
df_comb_train['day'] = df_comb_train['day'].astype('int')
df_comb_test['day'] = df_comb_test['day'].astype('int')
### Make mean encoding subject_line_length
dict_encode = df_comb_train.groupby('subject_line_length').mean()['open_flag'].to_dict()

df_comb_train['subject_line_length_encoded'] = df_comb_train['subject_line_length'].map(dict_encode)

df_comb_test['subject_line_length_encoded'] = df_comb_test['subject_line_length'].map(dict_encode)
df_comb_test['subject_line_length_encoded'] = df_comb_test['subject_line_length_encoded'].fillna(0)
# Chi-square test dari variabel kategorikal terhadap variabel respon
def chi_square_test(dfa, var1, var2) :
    # Membuat contingency table
    df = pd.crosstab(dfa[var1], dfa[var2], margins=False)

    # Menghitung nilai statistiknya
    from scipy.stats import chi2_contingency, chi2
    stat, p, dof, expected = chi2_contingency(observed=df
                                         ,correction=True #Jika dof=1, maka digunakan Yates Correction (?)
                                         ,lambda_=None #Untuk mengganti Pearson Chi-Square menjadi Cressie-Read Divergence)
                                          )

    # Interpretasi nilai statistik
    prob = 0.95
    critical = chi2.ppf(prob, dof)

    if abs(stat) >= critical:
        print(var1,': Dependent')
    else:
        print(var1,': Independent')   
    
# Lakukan tes chi square untuk tiap variabel kategorikal
cat_var = ['country_code','last_open_check','last_login_check','last_checkout_check','attr_1',
           'attr_3','domain_type','age_class','dayofweek','month']
for var in cat_var :
    chi_square_test(df_comb_train, var, 'open_flag')
### Initialize h2o
import h2o
h2o.init()
### Define predictor and response
X = ['country_code','subject_line_length_encoded','attr_1','attr_3','domain_type','age_class',
     'open_count_last_10_days','open_count_last_30_days','open_count_last_60_days',
     'open_count','login_count','checkout_count','open_per_login','open_per_checkout',
     'last_open_check','last_login_check','last_checkout_check','day','dayofweek','month']

# X_ori = ['country_code','subject_line_length','last_open_day','last_login_day','last_checkout_day',
#      'open_count_last_10_days','open_count_last_30_days','open_count_last_60_days',
#      'login_count_last_10_days','login_count_last_30_days','login_count_last_60_days',
#      'checkout_count_last_10_days','checkout_count_last_30_days','checkout_count_last_60_days',
#      'attr_1','attr_3','age_class','domain','day','dayofweek','month']

Y = 'open_flag'

list_col = X + [Y]
### Split dataframe
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(df_comb_train, stratify=df_comb_train['open_flag'], test_size = 0.2, random_state=11)
### Make H2O Frame
h2o_train = h2o.H2OFrame(train_data[list_col])
h2o_val = h2o.H2OFrame(val_data[list_col])
h2o_test = h2o.H2OFrame(df_comb_test[X])
### Make categorical
X_cat = ['country_code','attr_1','attr_3','domain_type','age_class',
         'last_open_check','last_login_check','last_checkout_check',
         'dayofweek','month']

# X_cat = ['country_code','attr_1','attr_3','domain','age_class',
#          'dayofweek','month']

for var in X_cat :
    h2o_train[var] = h2o_train[var].asfactor()
    h2o_val[var] = h2o_val[var].asfactor()
    h2o_test[var] = h2o_test[var].asfactor()
    
h2o_train[Y] = h2o_train[Y].asfactor()
h2o_val[Y] = h2o_val[Y].asfactor()
### Make all H2O baseline model
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import time

def h2o_compare_models(df_train, df_test, X, Y) :
    
    start = time.time()
    
    # Initialize all model (Ganti family/distributionnya)
    glm = H2OGeneralizedLinearEstimator(family='binomial', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    gbm = H2OGradientBoostingEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    xgb = H2OXGBoostEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",
                              nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    rf = H2ORandomForestEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    ext = H2ORandomForestEstimator(distribution='bernoulli', histogram_type="Random",
                                  nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')
    
    # Train model
    glm.train(x=X, y=Y, training_frame=df_train)
    gbm.train(x=X, y=Y, training_frame=df_train)
    xgb.train(x=X, y=Y, training_frame=df_train)
    lgbm.train(x=X, y=Y, training_frame=df_train)
    rf.train(x=X, y=Y, training_frame=df_train)
    ext.train(x=X, y=Y, training_frame=df_train)
    
    # Calculate train metrics (Bisa diganti)
    from sklearn.metrics import matthews_corrcoef
    train_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.predict(h2o_train)['predict'].as_data_frame())
    train_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.predict(h2o_train)['predict'].as_data_frame())
    train_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.predict(h2o_train)['predict'].as_data_frame())
    train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.predict(h2o_train)['predict'].as_data_frame())
    train_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.predict(h2o_train)['predict'].as_data_frame())
    train_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.predict(h2o_train)['predict'].as_data_frame())

    # Calculate CV metrics for all model (Bisa diganti)
    met_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.cross_validation_holdout_predictions()['predict'].as_data_frame())
    met_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.cross_validation_holdout_predictions()['predict'].as_data_frame())
    
    # Calculate holdout metrics
    from sklearn.metrics import matthews_corrcoef
    hold_glm = matthews_corrcoef(h2o_val[Y].as_data_frame(), glm.predict(h2o_val)['predict'].as_data_frame())
    hold_gbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), gbm.predict(h2o_val)['predict'].as_data_frame())
    hold_xgb = matthews_corrcoef(h2o_val[Y].as_data_frame(), xgb.predict(h2o_val)['predict'].as_data_frame())
    hold_lgbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), lgbm.predict(h2o_val)['predict'].as_data_frame())
    hold_rf = matthews_corrcoef(h2o_val[Y].as_data_frame(), rf.predict(h2o_val)['predict'].as_data_frame())
    hold_ext = matthews_corrcoef(h2o_val[Y].as_data_frame(), ext.predict(h2o_val)['predict'].as_data_frame())
    
    # Make result dataframe
    result = pd.DataFrame({'Model':['GLM','GBM','XGB','LGBM','RF','ExtraTree'],
                          'Train Metrics':[train_glm,train_gbm,train_xgb,train_lgbm,train_rf,train_ext],
                          'CV Metrics':[met_glm,met_gbm,met_xgb,met_lgbm,met_rf,met_ext],
                          'Holdout Metrics':[hold_glm,hold_gbm,hold_xgb,hold_lgbm,hold_rf,hold_ext]})
    
    end = time.time()
    print('Time Used :',(end-start)/60)
    
    return result.sort_values('Holdout Metrics') 
### Compare models
res = h2o_compare_models(h2o_train, h2o_test, X, Y) 
res
### Search max depth
from h2o.estimators import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch

start = time.time()
lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",
                           nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo',
                           ntrees=100, learn_rate=0.05,
                           sample_rate = 0.8, col_sample_rate = 0.8, seed=11, score_tree_interval = 10,
                           stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 1e-4)

# LGBM Params
lgbm_params = {'max_depth' : [3,5,7,9,11,13,15]}

# Search criteria
search_criteria = {'strategy': "Cartesian"}

# Make grid model
lgbm_grid = H2OGridSearch(model=lgbm,
                          grid_id='best_lgbm_max_depths',
                          hyper_params=lgbm_params,
                          search_criteria=search_criteria)

# Train model
lgbm_grid.train(x=X, y=Y, training_frame=h2o_train, validation_frame=h2o_val)

# Get best GLM
lgbm_res = lgbm_grid.get_grid(sort_by='auc', decreasing=True)
print(lgbm_res)

end = time.time()
print('Time Used :',(end-start)/60)
### Tune Model - LGBM - RandomGridSearch
from h2o.estimators import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import log_loss
start = time.time()
lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",
                           nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo',
                           ntrees=100, seed=11, score_tree_interval = 10,
                           stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 1e-4)

# LGBM Params
lgbm_params = {'max_depth' : [7,9,11],
                'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
              'reg_lambda':list(np.arange(0.5,1.05,0.05)),
              'reg_alpha':list(np.arange(0.01,0.11,0.01)),
             'learn_rate':list(np.arange(0.01,0.11,0.01)),
             'booster':['dart','gbtree']}

# Search criteria
search_criteria = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 3600,  ## limit the runtime to 60 minutes
                   'max_models': 20,  ## build no more than 100 models
                   'seed' : 11,
                   'stopping_rounds' : 5,
                   'stopping_metric' : "auc",
                   'stopping_tolerance': 1e-3
                   }

# Make grid model
lgbm_grid = H2OGridSearch(model=lgbm,
                          grid_id='best_lgbm_cmon',
                          hyper_params=lgbm_params,
                          search_criteria=search_criteria)

# Train model
lgbm_grid.train(x=X, y=Y, training_frame=h2o_train, validation_frame=h2o_val)

# Get best GLM
lgbm_res = lgbm_grid.get_grid(sort_by='auc', decreasing=True)
best_lgbm = lgbm_res.models[0]
# Hitung metrics
from sklearn.metrics import matthews_corrcoef
train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_lgbm.predict(h2o_train)['predict'].as_data_frame())
met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())
hold_lgbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), best_lgbm.predict(h2o_val)['predict'].as_data_frame())

# Print result
print('Train metrics :',train_lgbm)
print('CV metrics :',met_lgbm)
print('Holdout metrics :',hold_lgbm)

end = time.time()
print('Time Used :',(end-start)/60)
### Make submission
pred = best_lgbm.predict(h2o_test)['predict'].as_data_frame()
sub = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')
sub['open_flag'] = pred

sub.to_csv('subs_lgbm_gridsearch.csv', index=False)
