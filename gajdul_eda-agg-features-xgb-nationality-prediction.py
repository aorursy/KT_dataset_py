import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

%matplotlib inline
# Check the mpl version (3.1.1 causes issues with seaborn)
matplotlib.__version__
# command for readable pandas formatting
pd.options.display.float_format = "{:.2f}".format
# Load the data
df = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
# Inspect the data
df.head(1)
# Inspect the metadata.
df.info(verbose=False)
# Inspect the data shape
print('Number of rows:',df.shape[0])
# Classify the columns to categorical and numerical
num_cols = df._get_numeric_data().columns
cat_cols = [col for col in df.columns if col not in num_cols]
print('Number of columns:',len(df.columns),
      f' (numerical:{len(num_cols)},',
      f' categorical:{len(cat_cols)})')
# First step of cleaning- IPC.
# Limit the analysis to IPC =1 to get rid of duplicated submissions.
"""
As per Kaggle dataset description:
The number of records from the user's IP address in the dataset. 
For max cleanliness, only use records where this value is 1. 
High values can be because of shared networks (e.g. entire universities) or multiple submissions
"""
df = df.loc[df['IPC']==1]
# Get rid of invalid results 
# As the answers are in scale 1 to 5, we want to delete invalid inputs 
df = df.loc[(df[df.columns.tolist()[:49]] >= 1).all(axis=1)]
# positive questions adding to the trait.
pos_questions = [ 
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7
]

# negative (negating) questions subtracting from the trait.
neg_questions = [ 
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5
    'EST2','EST4',                       # 2
    'AGR1','AGR3','AGR5','AGR7',         # 4
    'CSN2','CSN4','CSN6','CSN8',         # 4
    'OPN2','OPN4','OPN6',                # 3
]

# Replace the question answer with -2 to 2 scale depending if the question is positive or negative.
df[pos_questions] = df[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
df[neg_questions] = df[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
# Check for missing data.
df.isna().mean().sum()
df = df.dropna()
df.isna().mean().sum()
# columns with time spent answering questions
qtime_cols = list(df.columns)[50:100]
# Check if selected correct columns
qtime_cols[0], qtime_cols[-1]
# Calculate the total time for each survey
df['total_time']=df[qtime_cols].sum(axis=1)
df['total_time'].describe()
# Can't see anything due to large outliers
ax = sns.distplot(df['total_time'])
# See how much data will be lost if we get rid of the outliers
total_respondents = len(df)
fast_respondents = len(df[df['total_time']<10000])
slow_respondents = len(df[df['total_time']>1000000])

print("Total respondents:",total_respondents)
print("Slowest respondents:",slow_respondents/total_respondents)
print("Fastest respondents:",fast_respondents/total_respondents)
df = df[df['total_time'].between(10000,1000000)]
from matplotlib import style
style.use("seaborn-darkgrid")
df[['total_time']].plot(kind='hist',bins=20)
plt.title('Test completion times')
plt.show()
# List the redundant cols such as longitude and latitudee
drop_cols=list(df.columns[50:107])+['lat_appx_lots_of_err','long_appx_lots_of_err']
# Drop the redundant cols
df=df.drop((drop_cols), axis=1)
df
# List the number of unique countries, count them
countries = df['country'].unique()
len(countries)
# A list of all EU countries, count them
EU = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
len(EU)
# Check if all EU countries are in the data
intersection = set(EU).intersection(set(countries))
len(intersection)
# Limit the analysis to EU countries
df = df.loc[df['country'].isin(EU)]
# Count responses by country
df['country'].value_counts()[:5]
# This gives us percentage of responses from each country
df['country'].value_counts(normalize=True) * 100
df.head()
# Create an aggregated feature for each of the five personality dimensions.
# They will average the 10 answers across the dimension.

# Extraversion 
EXT = list(df.columns[:10])
# Emotional Stability
EST = list(df.columns[10:20])
# Agreeableness
AGR = list(df.columns[20:30])
# Conscientiousness
CSN = list(df.columns[30:40])
# Openness
OPN = list(df.columns[40:50])

dimensions = [EXT,EST,AGR,CSN,OPN]
dimension_averages=["extraversion","emotional_stability",
       "agreeableness","conscientiousness","openness"]
for d in range(len(dimensions)):
    df[dimension_averages[d]] = df[dimensions[d]].mean(axis=1)
df.head(1)
# Analyse the aggregated features
df[dimension_averages].describe()
# Use a boxlot to visualise the 5 variables
# This method will give us a good overview of the distribution across the variables
sns.set_style("darkgrid")

#reset default parameters
sns.set()
plt.figure(figsize=(12, 6))
sns.set(font_scale=1.5)
sns.boxplot(data=df[dimension_averages]);
plt.title("Average characteristics of European citizens",fontsize=22)
plt.savefig('avg_char.png')
plt.show()
#reset default parameters
sns.set()
plt.figure(figsize=(12, 6))

# Visualise the correlation
corr=df[dimension_averages].corr()
mask = np.triu(corr)
sns.set(font_scale=1.2)
sns.heatmap(df[dimension_averages].corr(),
            vmin=0,
            vmax=1,
            annot = True,
            square=True, 
            mask=mask,
            cbar=True,
            cmap='Blues')
plt.title('Correlation between personality traits',fontsize=22)
plt.savefig('correlations.png')
plt.show()
# Subset df to only those with country GB, PL
gb = df.loc[df['country']=="GB"]
pl = df.loc[df['country']=="PL"]
# Limit the analysis to two countries and averages across 5 dimensions
gb = gb[gb.columns[-6:]]
pl = pl[pl.columns[-6:]]
def transpose_table(df, col_list):
    """
    INPUT 
        df - a dataframe holding the col_list columns
        col_list- columns that we want to transpose into rows
        
    OUTPUT
        new_df- a transposed dataframe.
    """
    new_df = defaultdict(int)
    for i in col_list:
        new_df[i]=df[i].mean()
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.rename(columns={'index': 'personality', 0: 'average'}, inplace=True)
    new_df.set_index('personality', inplace=True)
    return new_df 
dimension_averages
gb_avg = transpose_table(gb,dimension_averages)
pl_avg = transpose_table(pl,dimension_averages)
comp_df = pd.merge(gb_avg, pl_avg, left_index=True, right_index=True)
comp_df.columns = ['gb_avg', 'pl_avg']
comp_df['value_difference'] = comp_df['gb_avg'] - comp_df['pl_avg']
comp_df.style.bar(subset=['value_difference'], align='mid', color=['#d65f5f', '#5fba7d'])
df.head()
# Add binary column to indicate if Great Britain 
df['is_gb'] = df['country'].apply(lambda x: 1 if x =='GB' else 0)
# Copy the dataframe
df_ml = df.copy()

to_drop =["country","total_time"]
          #+["extraversion","emotional_stability","agreeableness","conscientiousness","openness"]
    
# Delete old column indicating country
df_ml = df_ml.drop(columns=to_drop)

# Shuffle the data to ensure that split is fair
df_ml = df_ml.sample(n=len(df_ml),random_state=42)
corr_data = pd.DataFrame(df_ml.corr()['is_gb'][:])
corr_data = corr_data.reset_index()
corr_data = corr_data.sort_values(by=['is_gb'])
corr_data[:3]
corr_data[-4:-1]
top_correlation = corr_data.sort_values('is_gb', ascending=False).head(10)['index'].to_list()
least_correlation = corr_data.sort_values('is_gb', ascending=False).tail(5)['index'].to_list()
# Count the outcome variables to identify the baseline
positives = len(df.loc[df_ml['is_gb']==1])
negatives = len(df.loc[df_ml['is_gb']==0])
1-(positives/(positives+negatives))
# Select the dependent variable
Y = df_ml['is_gb']
X = df_ml.drop('is_gb',axis=1)
# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
import xgboost as xgb

# Instantiate the model
xgb_model = xgb.XGBClassifier(learning_rate=0.05, 
              max_depth=3,
              gamma=0.08435594187707007,
              colsample_bytree=0.5336629698328548,
              n_estimators=1000, 
              objective='binary:logistic', 
              random_state=42)

# fit model to training data
xgb_model.fit(X_train, y_train)
# make predictions for test data
y_pred = xgb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Find optimal threshold
thresholds=(np.linspace(0.45,0.50,20))
for t in thresholds:
    predictions=xgb_model.predict_proba(X_test)[:,1]>t
    print("AUC for threshold",t,":",
         roc_auc_score(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print("XGB Classifier accuracy: %.2f%%" % (accuracy * 100.0))
69/61
# Check the most important features
importance = xgb_model.get_booster().get_score(importance_type= 'gain')
sorted(importance.items(), key=lambda x:x[1],reverse=True)[:3]
gb_df = df.loc[df['is_gb']==1]
eu_df = df.loc[df['is_gb']==0]
comp_metrics = ['AGR3','CSN8','EST9']

gb_df = transpose_table(gb_df,comp_metrics)
eu_df = transpose_table(eu_df,comp_metrics)
comp_df = pd.merge(gb_df, eu_df, left_index=True, right_index=True)
comp_df.columns = ['gb_avg','eu_avg']
comp_df['value_difference'] = comp_df['gb_avg'] - comp_df['eu_avg']
comp_df.style.bar(subset=['value_difference'], align='mid', color=['#d65f5f', '#5fba7d'])