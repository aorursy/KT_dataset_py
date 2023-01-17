import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head(3)
data.shape
data.info()
data.describe().T
data.columns = data.columns.str.replace(' ','_').str.replace(':','')
data.columns
relevant_cols = ['Job_Title', 'Salary_Estimate', 'Job_Description',
       'Rating', 'Company_Name', 'Location', 'Headquarters', 'Size', 'Founded',
       'Type_of_ownership', 'Industry', 'Sector', 'Revenue', 'Competitors']
df = data.loc[:,relevant_cols]
filt_sal_error = (df['Salary_Estimate'].str.contains('-1'))
df.loc[filt_sal_error]
df = df.loc[~filt_sal_error]
df['Salary_Range'] = df.loc[:,'Salary_Estimate'].str.split().str[0].str.replace('K', '000').str.replace('$','')
df[['Minimum_Salary', 'Maximum_Salary']] = df.loc[:,'Salary_Range'].str.split('-', expand=True)
df.loc[:,'Minimum_Salary'] = df.loc[:,'Minimum_Salary'].astype('float64')
df.loc[:,'Maximum_Salary'] = df.loc[:,'Maximum_Salary'].astype('float64')
df['AVG_Salary'] = round((df.loc[:,'Minimum_Salary']+df.loc[:,'Maximum_Salary'])/2, 2)
f = plt.figure(figsize=(18,4))
f.add_subplot(1,3,1)
sns.distplot(df['Minimum_Salary'], kde=True, bins=10, color='y')
f.add_subplot(1,3,2)
sns.distplot(df['Maximum_Salary'], kde=True, bins=10, color='c')
f.add_subplot(1,3,3)
sns.distplot(df['AVG_Salary'], kde=True, bins=10, color='g')
plt.show()
df['Job_Title'].value_counts().head(20).to_frame()
# Analyst Type
analyst_types = ['Quality', 'Business', 'Governance', 'Healthcare', 'Financial', 'Research', 'Marketing', 'Reporting', 'SQL', 'Manager']

def get_job_type(job_title):
    for type in analyst_types:
        if type.lower() in job_title.lower():
            if type.lower() == 'manager':
                return 'Manager'
            else:
                return type + ' Data Analyst'
    else:
        return 'General Data Analyst'
df['Analyst_Type'] = df['Job_Title'].apply(lambda x: get_job_type(x))
df['Analyst_Type'].value_counts()
pd.pivot_table(df, index='Analyst_Type', values='AVG_Salary', aggfunc=np.mean).sort_values(by='AVG_Salary', ascending=False)
job_exp_lst = ['manager', 'senior', 'sr.', 'sr', 'lead', 'junior', 'jr', 'jr.']
low_exp_lst = ['junior', 'jr', 'jr.']
def get_job_exp(job_title):
    for exp in job_exp_lst:
        if exp in job_title.lower() and exp not in low_exp_lst:
            return 'senior'
        elif exp in job_title.lower() and exp in low_exp_lst:
            return 'junior'
    else:
        return 'intermediate'
df['Job_EXP'] = df['Job_Title'].apply(get_job_exp)
df['Job_EXP'].value_counts(dropna=False)
plt.figure(figsize=(16,4))
sns.boxplot(x='Job_EXP', y='AVG_Salary', data=df, order=['junior', 'intermediate', 'senior'])
plt.show()
df_high_rank = df[df['Job_EXP']=='senior']
df_low_rank = df[df['Job_EXP']=='junior']
df_no_rank = df[df['Job_EXP'] == 'intermediate']
plt.figure(figsize=(16,4))
sns.distplot(df_high_rank['AVG_Salary'], kde=True, hist=False,color='orange', label='Senior')
sns.distplot(df_low_rank['AVG_Salary'], kde=True, hist=False,color='green', label='Junior')
sns.distplot(df_no_rank['AVG_Salary'], kde=True, hist=False,color='navy', label='Intermediate')
plt.legend()
plt.show()
skills = ['SAS', 'Hadoop', 'Python', 'R program','AWS', 'Azure','SQL', 'Excel','Machine Learning', 'Tableau', 'Power BI', 'Qlik']
def skills_to_cols(dataframe, skills_list):
    for skill in skills_list:
        dataframe[skill+'_extracted'] = dataframe['Job_Description'].apply(lambda x: 1 if skill.lower() in x.lower() else 0)
skills_to_cols(df, skills)
cols_to_melt = [item+'_extracted' for item in skills]
df_melted = pd.melt(df, value_vars=cols_to_melt, var_name='Skill',value_name='TrueFalse')
filt = (df_melted['TrueFalse'] == 1)
df_melted = df_melted.loc[filt]
f = plt.figure(figsize=(20,4))
f = sns.countplot(df_melted['Skill'], order = df_melted['Skill'].value_counts().index)
f.set_xticklabels(f.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
df['Competitors'].value_counts().head()
df['Competitors_count'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x!='-1' else 0)
filt_rating_err = (df['Rating'] == -1)
print(f'Rating entries containing -1: {len(df.loc[filt_rating_err])}')
df.loc[filt_rating_err, 'Rating'] = np.nan
plt.figure(figsize=(10,4))
sns.distplot(df['Rating'], kde=True, bins=10)
plt.show()
df['Company_Name'].sample(10)
df['Company_Name_splitted'] = df.loc[:,'Company_Name'].str.split("\n").str[0]
rows_with_companies=len(df[df['Company_Name_splitted'].notna()])
num_unique_companies= df['Company_Name_splitted'].nunique()
num_non_unique_companies = (rows_with_companies)-num_unique_companies
print(f"Number of rows containing a company name in the dataset: {rows_with_companies}")
print(f"Number of unique companies: {num_unique_companies}")
print(f"Companies that occure more than once: {num_non_unique_companies}")
df_comp_dup = df.loc[df.duplicated(['Company_Name_splitted'])]
print(f"{df_comp_dup['Company_Name_splitted'].nunique()} companies \
occure more than once in this dataset and result in {num_non_unique_companies} duplicated entries.")
plt.figure(figsize=(16,6))
sns.barplot(x=df_comp_dup['Company_Name_splitted'].value_counts().head(10), y=df_comp_dup['Company_Name_splitted'].value_counts().head(10).index)
plt.xlabel('Number of job advertisements')
plt.ylabel('Company Name')
plt.show()
df['Location'].str.split(",", expand=True, n=1).iloc[:,1].value_counts()
df['State'] = df.loc[:,'Location'].str.split(",").str[-1]
df['City'] = df.loc[:,'Location'].str.split(",").str[0]
plt.figure(figsize=(16,4))
sns.countplot(df['State'], order= df['State'].value_counts().index)
plt.show()
plt.figure(figsize=(16,6))
sns.barplot(x=df['City'].value_counts().head(10), y=df['City'].value_counts().head(10).index)
plt.xlabel('Number of job advertisements')
plt.ylabel('City')
plt.show()
df['Size'].value_counts(dropna=False).sort_index()
size_missing = (df['Size']=='-1') | (df['Size']=='Unknown')
df.loc[size_missing, 'Size'] = np.nan
df.loc[:,'Size'] = df.loc[:,'Size'].str.replace(' employees', '')
df['Size'].value_counts(dropna=False).sort_index()
filt_year_missing = (df['Founded'] ==-1)
filt_year_missing.value_counts()
df.loc[filt_year_missing, 'Founded'] = np.nan
plt.figure(figsize=(16,4))
sns.boxplot(x='Founded', data=df)
plt.show()
year_filt_company = (df['Founded'] < 1900) & (df['Company_Name_splitted'].str.lower().str.contains('university')==False)
print(f"Founded below 1900 and not contain term <University> in their name: {len(df.loc[year_filt_company])} entries.")
df.loc[year_filt_company, ['Company_Name_splitted', 'Founded']].head(5)
year_filt_university = (df['Founded'] < 1900) & (df['Company_Name_splitted'].str.lower().str.contains('university'))
df.loc[year_filt_university].sort_values(by='Company_Name_splitted').head()
filt_year_min = (df['Founded'] < 1700)
df.loc[filt_year_min]
df.loc[filt_year_min, 'Founded'] = np.nan
df['Company_age'] = 2020-df['Founded']
df['Type_of_ownership'].value_counts(dropna=False)
type_filt_missing = (df['Type_of_ownership']=='-1') | (df['Type_of_ownership']=='Unknown')
df.loc[type_filt_missing,'Type_of_ownership'] = np.nan
sorted_index = df.groupby('Type_of_ownership')['AVG_Salary'].mean().sort_values(ascending=False).index
g = plt.figure(figsize=(16,4))
g = sns.boxplot(x='Type_of_ownership', y='AVG_Salary', data=df,  order=sorted_index)
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
filt_self_employed = (df['Type_of_ownership']=='Self-employed')
f"Number of entries for Self-employed: {len(df.loc[filt_self_employed])}"
filt_industry_missing = (df['Industry']=='-1')
df.loc[filt_industry_missing, 'Industry'] = np.nan
print(f"Number of unique industries: {df['Industry'].nunique()}")
f=plt.figure(figsize=(16,4))
f=sns.barplot(x=df['Industry'].value_counts(normalize=True).iloc[0:10].index, y=df['Industry'].value_counts(normalize=True).iloc[0:10]*100)
plt.xlabel('Industry Type')
plt.ylabel('Percentage')
f.set_xticklabels(f.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
percent_top_ten_ind = (df['Industry'].value_counts(normalize=True).iloc[0:10].sum()) * 100
print(f"The 10 most frequently represented industries represent {round(percent_top_ten_ind, 2)}% of all entries.")
df['Industry'].value_counts().loc[lambda x: x<3]
filt_sector_missing = (df['Sector']=='-1')
print(f"Missing values for Sector: {len(df.loc[filt_sector_missing])}")
df.loc[filt_sector_missing, 'Sector'] = np.nan
sorted_index=df.groupby('Sector')['AVG_Salary'].mean().sort_values(ascending=False).index
p = plt.figure(figsize=(18,4))
p = sns.boxplot(x='Sector', y='AVG_Salary', data=df, order=sorted_index)
p.set_xticklabels(p.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
f=plt.figure(figsize=(18,4))
f=sns.barplot(x=df['Sector'].value_counts(normalize=True).index, y=df['Sector'].value_counts(normalize=True)*100)
plt.xlabel('Sector')
plt.ylabel('Percentage')
f.set_xticklabels(f.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
df['Revenue'].value_counts()
filt_revenue_missing = (df['Revenue']=='Unknown / Non-Applicable')
df.loc[:,'Revenue'] = df.loc[:,'Revenue'].str.replace(' \(USD\)', '')
df.loc[filt_revenue_missing, 'Revenue'] = np.nan
df.loc[:,cols_to_melt] = df.loc[:,cols_to_melt].astype(str)
included_features = ['Analyst_Type', 'Rating', 'Job_EXP','Size', 'Competitors_count',\
                     'Company_age', 'Type_of_ownership', 'Industry','Sector', 'Revenue', 'State', 'City','AVG_Salary']\
+cols_to_melt
included_features
df_rel = df.loc[:,included_features]
len(df_rel.columns)
df_rel.dropna(axis=0, thresh=len(df_rel.columns)-3, inplace=True)
initial_len = len(df)
reduced_len = len(df_rel)
print(f'By dropping rows with more than 3 NaN values we lost: {initial_len-reduced_len} rows')
df_rel_nan_cols = df_rel.isna().sum()
df_rel_nan_cols[df_rel_nan_cols>0]
numerical_features = [col_name for col_name in df_rel.columns if df_rel[col_name].dtype in ['int64', 'float64'] and 'Salary' not in col_name]
OHE_features = [col_name for col_name in df_rel.columns if df_rel[col_name].dtype=='object' and df_rel[col_name].nunique() <=10]
HC_features = [col_name for col_name in df_rel.columns if df_rel[col_name].dtype=='object' and df_rel[col_name].nunique() >10]
print(f'Numerical Features: {numerical_features}')
print(f'Low Cardinality Features: {OHE_features}')
print(f'High Cardinality Features: {HC_features}')
X = df_rel.drop('AVG_Salary', axis=1)
y = df_rel['AVG_Salary'].values
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import category_encoders as ce
num_transformer = Pipeline(steps=[
    ('num_impute', SimpleImputer(strategy='median')),
    ('normalize', Normalizer())
])
OHE_transformer = Pipeline(steps=[
    ('OHE_impute_cols', SimpleImputer(strategy='constant')),
    ('OHE', OneHotEncoder(handle_unknown='ignore'))
])
HC_transformer = Pipeline(steps=[
    ('HC', ce.CatBoostEncoder())
])
preprocessor= ColumnTransformer(transformers=[
    ('OHE_transform', OHE_transformer, OHE_features),
    ('numerical_transform', num_transformer, numerical_features),
    ('HC_transform', HC_transformer, HC_features)
])
OHE_features
# high cardinality
HC_features
numerical_features
from sklearn.ensemble import RandomForestRegressor
model_regr = RandomForestRegressor(random_state = 42)
pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model_regr)
])
pipe.fit(X_train, y_train)
mae_avg_scores = np.mean(-1*cross_val_score(pipe, X, y, cv=10, scoring='neg_mean_absolute_error'))
print(mae_avg_scores)
from sklearn.model_selection import RandomizedSearchCV
rand_grid = {'model__bootstrap': [True, False],
 'model__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'model__min_samples_leaf': [1, 2, 4],
 'model__min_samples_split': [2, 5, 10],
 'model__n_estimators': [200, 400, 600, 800, 1000]}
rs = RandomizedSearchCV(pipe, param_distributions = rand_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rs.fit(X_train, y_train)
rs.best_params_
y_pred_rs = rs.best_estimator_.predict(X_valid)
mae_rs = round(mean_absolute_error(y_valid, y_pred_rs), 2)
print(f'Mean Absolute Error - RandomSearch: \n -> {mae_rs}')
from sklearn.model_selection import GridSearchCV
param_grid = {'model__n_estimators': [100, 150, 200],
 'model__min_samples_split': [4, 5],
 'model__min_samples_leaf':[4, 5],
 'model__max_features': ['auto'],
 'model__max_depth': [4, 5, 6],
 'model__bootstrap': [True]}
gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv = 3, n_jobs = -1, verbose = 2)
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
y_pred_gs = gridsearch.best_estimator_.predict(X_valid)
mae_gs = round(mean_absolute_error(y_valid, y_pred_gs), 2)
print(f'Mean Absolute Error - Gridsearch: \n -> {mae_gs}')