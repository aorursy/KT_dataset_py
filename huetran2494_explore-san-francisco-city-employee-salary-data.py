# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_path = '/kaggle/input/SFSalaries.csv'
df = pd.read_csv(data_path)
print(df.shape)
df.head(5)
df.shape
print('Total records:', df.shape[0])
print('Total columns:',df.shape[1])
df.info()
df = df.drop(columns=['Id'])
df['JobTitle'] = df['JobTitle'].map(lambda x: x.lower())
for column in ['BasePay', 'OvertimePay', 'Benefits', 'OtherPay']:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(value=np.float64(0))
def exploring_stats(df_input):
    total_rows = df_input.shape[0]
    total_columns = df_input.shape[1]
    # check data type
    name = []
    sub_type = []
    for n, t in df_input.dtypes.iteritems():
        name.append(n)
        sub_type.append(t)

    # check distinct
    # cname is column name
    check_ndist = []
    for cname in df_input.columns:
        ndist = df_input[cname].nunique()
        pct_dist = ndist * 100.0 / total_rows
        check_ndist.append("{} ({:0.2f}%)".format(ndist, pct_dist))
    # check missing
    check_miss = []
    for cname in df_input.columns:
        nmiss = df_input[cname].isnull().sum()
        pct_miss = nmiss * 100.0 / total_rows
        check_miss.append("{} ({:0.2f}%)".format(nmiss, pct_miss))
    # check zeros
    check_zeros = []
    for cname in df_input.columns:
        try:
            nzeros = (df_input[cname] == 0).sum()
            pct_zeros = nzeros * 100.0 / total_rows
            check_zeros.append("{} ({:0.2f}%)".format(nzeros, pct_zeros))
        except:
            check_zeros.append("{} ({:0.2f}%)".format(0, 0))
            continue
    # check negative
    check_negative = []
    for cname in df_input.columns:
        try:
            nneg = (df_input[cname].astype("float") < 0).sum()
            pct_neg = nneg * 100.0 / total_rows
            check_negative.append("{} ({:0.2f}%)".format(nneg, pct_neg))
        except:
            check_negative.append("{} ({:0.2f}%)".format(0, 0))
            continue
    data = {"column_name": name, "data_type": sub_type, "n_distinct": check_ndist, "n_miss": check_miss, "n_zeros": check_zeros,
            "n_negative": check_negative, }
    # check stats
    df_stats = df_input.describe().transpose()
    check_stats = []
    for stat in df_stats.columns:
        data[stat] = []
        for cname in df_input.columns:
            try:
                data[stat].append(df_stats.loc[cname, stat])
            except:
                data[stat].append(0.0)
    # col_ordered = ["name", "sub_type", "n_distinct", "n_miss", "n_negative", "n_zeros",
    #                "25%", "50%", "75%", "count", "max", "mean", "min", "std"]  # + list(pdf_sample.columns)
    df_data = pd.DataFrame(data)
    # df_data = pd.concat([df_data, df_sample], axis=1)
    # df_data = df_data[col_ordered]
    return df_data
exploring_stats(df)
condition = (df['BasePay']<0) | (df['OvertimePay']<0) | (df['TotalPay']<0) |(df['TotalPayBenefits']<0)
df = df[~condition]
df.info()
df = df[~df.JobTitle.isin(['not provided'])]
def job_type(row):
  for key, value in job_type_dict.items():
    for val in value:
      if val in row.lower():
        return key
  return 'Other'

def job_level(row):
  for key, value in level_dict.items():
    for val in value:
      if val in row.lower():
        return key
  return 'Other'
job_type_dict = dict({
    'Management' : ['department head', 'manager', 'deputy director', 'human resources', 'employee relations', 'project director','superintendent'],
    'Fire': ['fire'],
    'Police': ['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant', 'safetycomm',
               'public defender', 'incident', 'forensic', 'criminalist'],
    'Transit': ['mta', 'transit', 'truck', 'transportation', 'railway'],
    'Medical': ['anesth', 'medical', 'nurs', 'health', 'physician', 'diagnostic imaging tech', 'psychologist',
             'orthopedic', 'health', 'pharm', 'care', 'dentist', 'therapist'],
    'Airport': ['airport'],
    'Animal': ['animal'],
    'Architecture': ['architect', 'construction'],
    'Court': ['court', 'legal'],
    'Automotive': ['mechanic', 'automotive', 'transmission line'],
    'Engineering': ['engineer', 'engr', 'eng', 'program', 'information technology', 'power generation', 'radiologic technologist',],
    'General Laborer': ['general laborer', 'painter', 'inspector', 'machinist', 'administrative',
                        'carpenter', 'electrician', 'plumber', 'maintenance', 'dispatcher',
                        'custodian', 'garden', 'guard',  'cleaner', 'sewer repair',
                        'gardener', 'social worker', 'public works', 'parking', 'technician', 'worker'],
    'Service': ['food serv', 'public service', 'service'],
    'Admin': ['admin', 'aide', 'assistant', 'secretary', 'elections clerk', 'executive contract employee',
              'attendant', 'librar', 'public svc aide-public works'],
    'Data': ['analyst', 'data'],
    'Accounting': ['accountant', 'account', 'account clerk','treasurer'],
    'Government': ['mayoral', 'mayor', 'assessor'], 
    'Recreation': ['recreation'], 
    'Lawyer': ['attorney', 'lawyer'],
    'Education' : ['instructor', 'counselor', 'employment & training'],
    'Shipping': ['port','porter', 'train controller'],
    'Sales' : ['senior clerk', 'junior clerk', 'clerk'],
    'NA':['not provide']
})
level_dict = dict({
    'Top Level Management' : ['chief' , 'senior management', 'senior deputy', 'director', 'mayor', 'superintendent'],
    'Middle-Level Management': ['manager',  'deputy', 'department head' , 'commander'],
    'First-Level Management' : ['leader', 'supervisor', 'sergeant', 'captain', 'administrator', 'head park',
                                'lieutenant', 'sheriff', 'management assistant'], 
    'Expert' : ['prof', 'specialist', 'counselor', 'attorney', 'pharmacist', 'forensic', 'dentist','criminalist' , 'therapist', 'psychologist', 
                'board secretary', 'employment & training'],
    'Experienced Level': ['senior', 'special', 'registered', 'anesthetist', 'paramedic', 'investigator', 'marine engineer', 'engineer',
                          'pr administrative analyst', 'assessor', 'treasurer', 'clerk', 'management assistant', 'architect',
                          'commissioner',
                          'stationary eng', 'public defender', 'executive contract employee', 'court executive officer', 'pilot of fire boats',
                          'instructor','patient care assistant', 'principal', 'account clerk', 'assistant medical examiner'], 
    'Entry Level': ['trainee', 'junior', 'practitioner', 'associate', 'physician assistant', 'library page', 'medical evaluations assistant',
                    'asst engr', 
                    'nursing assistant', 'licensed vocational', 'nurse midwife', 'administrative'],
    'Technician' : ['technician', 'operator', 'firefighter', 'inspector', 'diagnostic imaging tech','electrical transit system mech', 'clerk typist',
                    'library assistant', 'train controller', 'safetycomm',
                    'police officer', 'firefighter', 'mechanic', 'librarian', 'machinist', 'dispatcher', 'elections clerk', 'tech'],
    'General' : ['worker', 'guard', 'custodian',  'gardener', 'plumber', 'porter', 'public svc aide-public works',
                 'parking', 'driver', 'painter', 'cleaner', 'electrician', 'carpenter', 'general laborer'],
    'NA':['not provide']
})
df['JobType'] = df['JobTitle'].map(job_type)
df['Level'] = df['JobTitle'].map(job_level)
# plt.figure(figsize=(15, 8))
# ax = sns.boxplot(x="Level", y="BasePay", hue="Status", data=df)   
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.show()
is_null = df.isnull().sum()
is_null = is_null[is_null>0]
is_null.sort_values(inplace=True, ascending=False)

#missing data
total = is_null
percent = is_null/len(df) * 100

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# plot missing data percent again
print(missing_data.index)
plt.figure(figsize=(8, 8))
sns.set(style='whitegrid')
g = sns.barplot(x=missing_data.index, y='Percent', data=missing_data)
plt.xticks(rotation = 90)
plt.title("missing values percentage.")
plt.xticks(rotation=45)

#plot value on top of bar
for p in range(len(missing_data)):
  value = missing_data.iloc[p, 1]
  g.text(p, value, f'{value:1.2f}%', color='black', ha="center")

plt.show()
df_drop_columns = df.drop(columns=['Notes', 'Agency'])
df_drop_columns=df_drop_columns[~df_drop_columns['JobTitle'].isin(['not provided'])]
df_drop_columns['Benefits'] = df_drop_columns.groupby(['JobType', 'Level'])['Benefits'].transform(lambda x: x.fillna(x.median()))
df_drop_columns = df_drop_columns.dropna(subset=['Benefits'])
df_drop_columns['BasePay'] = df_drop_columns.groupby(['JobType', 'Level'])['BasePay'].transform(lambda x: x.fillna(x.median()))
df_drop_columns['OvertimePay'] = df_drop_columns['OvertimePay'].fillna(0)
df_drop_columns['Status'] = df_drop_columns['Status'].fillna('NA')
df_drop_columns['OtherPay'] = df_drop_columns['OtherPay'].astype(float)
df_drop_columns['Benefits'] = df_drop_columns['Benefits'].astype(float)
df_no_missing = df_drop_columns[(df_drop_columns['OtherPay']>=0) & (df_drop_columns['Benefits']>=0)]
# exploring_stats(df_no_missing)
# for column in ['OtherPay', 'Benefits']:
#     df_no_mising.info()[column] = pd.to_numeric(df_no_mising.info()[column], errors='coerce')
#     df_no_mising.info()[column].fillna(value=np.float64(0))
# df_drop_columns['BasePay'] = df_drop_columns['BasePay'].apply(lambda x: np.nan if x <= 0.00 else x)
exploring_stats(df_no_missing)
# df_drop_columns['BasePay'] = df_drop_columns.groupby(['JobTitle'])['BasePay'].transform(lambda x: x.fillna(x.median()))
df_no_missing['TotalPay']  = df_no_missing['BasePay'] + df_no_missing['OvertimePay'] + df_no_missing['OtherPay'] 
df_no_missing['TotalPayBenefits'] = df_no_missing['TotalPay'] + df_no_missing['Benefits'] 
exploring_stats(df_no_missing)
# df_no_missing[df_no_missing.BasePay == 0]
plt.figure(figsize=(15,8))
df_no_missing.boxplot(column=['BasePay', 'OvertimePay', 'TotalPay', 'TotalPayBenefits', 'OtherPay', 'Benefits'])
plt.show()
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] >= fence_low) & (df_in[col_name] <= fence_high)]
#     df_out_outliers = df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
    return df_out
# def outlier(df_in, col_name):
#     q1 = df_in[col_name].quantile(0.25)
#     q3 = df_in[col_name].quantile(0.75)
#     iqr = q3-q1 #Interquartile range
#     fence_low  = q1-1.5*iqr
#     fence_high = q3+1.5*iqr
#     df_out = df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
#     return df_out
df_out = df_no_missing
for col_name in df_no_missing._get_numeric_data().columns:
  df_out = remove_outlier(df_out, col_name)
  
df_out.shape
plt.figure(figsize=(15,8))
bp_dict = df_out.boxplot(column=['BasePay', 'OvertimePay', 'TotalPay', 'TotalPayBenefits', 'OtherPay', 'Benefits'], return_type='both',
    patch_artist = True)

plt.show()
# plt.figure(figsize=(15,8))
# df_out.boxplot(column=['BasePay', 'OvertimePay', 'TotalPay', 'TotalPayBenefits', 'OtherPay', 'Benefits'])
# plt.show()
plt.figure(figsize=(5,6))
sns.boxplot(x='Status', y='BasePay', data=df_out, showfliers=False)
plt.show()
mean_pt = df_out[df_out.Status.isin(['PT'])]['BasePay']
mean_ft =df_out[df_out.Status.isin(['FT'])]['BasePay']
f_val, p_val = stats.f_oneway(mean_pt, 
                              mean_ft)  
print('PT std is ', mean_pt.std())
print('FT std is ', mean_ft.std())
print( "ANOVA results: F=", f_val, ", P =", p_val )
# Approach using p-value
if p_val > 0.05:
    print('Failed to reject the null hypothesis')
else:
    print('Reject the null hypothesis (Null hypothesis: the means of all groups are identical )')
pay_columns = ['BasePay', 'OvertimePay', 'OtherPay', 'TotalPay', 'Benefits', 'TotalPayBenefits']
pays_arrangement = list(zip(*(iter(pay_columns),) * 3))
pays_arrangement
#     2x3 array of axes
fig, axes = plt.subplots(2,3)

# set the figure height
fig.set_figheight(6)
fig.set_figwidth(18)

for i in range(len(pays_arrangement)):
    for j in range(len(pays_arrangement[i])):
        # pass in axes to pandas hist
        df_out[pays_arrangement[i][j]].hist(ax=axes[i,j], color = 'g')
        axes[i,j].set_title('Histogram of '+ pays_arrangement[i][j])
#         axes[i,j].set_xlim(0, df_no_missing[pays_arrangement[i][j]].max())
        
# add a row of emptiness between the two rows
plt.subplots_adjust(hspace=1)
# add a row of emptiness between the cols
plt.subplots_adjust(wspace=1)
plt.show()
t, pvalue = stats.ttest_rel(df_out['TotalPay'], df_out['BasePay'])
print("ttest is ", t, ", and pvalue is ", pvalue)
# Approach using p-value
if pvalue > 0.05:
    print('Failed to reject the null hypothesis')
else:
    print('Reject the null hypothesis')
import seaborn as sns
plt.figure(figsize=(10,7))
sns.distplot(df_out['TotalPay'], rug=True, hist=False, label='TotalPay')
sns.distplot(df_out['BasePay'], rug=True, hist=False, label='BasePay')
t, pvalue = stats.shapiro(df_no_missing['TotalPay'])
print(pvalue)
# Approach using p-value
if pvalue > 0.05:
    print('Failed to reject the null hypothesis')
else:
    print('Reject the null hypothesis (null hypothesis is normally distributed)')
import seaborn as sns
train_df = df_out
plt.figure(figsize=(16,8))
sns.countplot('JobType', data = train_df, hue = 'Year')
plt.xticks(rotation = 30)
plt.xlabel('Job Type')
plt.ylabel('Number of Labors')
plt.title('Number of labors for each job type of each year')
plt.tight_layout()
import seaborn as sns
plt.figure(figsize=(15,7))
sns.countplot('Level', data = train_df, hue = 'Year')
plt.xticks(rotation = 30)
plt.xlabel('Level')
plt.ylabel('Number of Labors')
plt.title('Labours Distribution of each years by level')
plt.tight_layout()
df_data1 = train_df[~train_df.Status.isin(['NA'])][['JobType', 'Status', 'TotalPay']]
df_data = df_data1.pivot_table(values='TotalPay', index='JobType', columns='Status', aggfunc='count').fillna(0)
df_data
status_percents = df_data.div(df_data.sum(1), axis=0)
plt.figure(figsize=(20,8))
# and plot the bar graph with a stacked argument.  
ax = status_percents.plot(kind='barh', stacked=True, rot=0, figsize=(5,8), color ='gy')
plt.ylabel('Job Type')
plt.xlabel('Percentage')
plt.xlim(0,1)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,6))
sns.barplot(data= train_df.groupby('JobType')['TotalPay'].agg('mean').reset_index(), x ='JobType', y = 'TotalPay')
plt.xticks(rotation=90)
plt.title('Average TotalPay in each Job type')
plt.tight_layout()
grouped_jobtype_anavo=train_df[['JobType', 'TotalPay']].groupby(['JobType'])
f_val, p_val = stats.f_oneway(grouped_jobtype_anavo.get_group('Lawyer')['TotalPay'], 
                              grouped_jobtype_anavo.get_group('Data')['TotalPay'], 
                              grouped_jobtype_anavo.get_group('Architecture')['TotalPay'])  
print('Lawyer std is ', grouped_jobtype_anavo.get_group('Lawyer')['TotalPay'].std())
print('Data std is ', grouped_jobtype_anavo.get_group('Data')['TotalPay'].std())
print('Architecture std is ', grouped_jobtype_anavo.get_group('Architecture')['TotalPay'].std())
print( "ANOVA results: F=", f_val, ", P =", p_val )
# Approach using p-value
if p_val > 0.05:
    print('Failed to reject the null hypothesis')
else:
    print('Reject the null hypothesis (the means of all groups are identical )')
plt.figure(figsize=(15,6))
sns.barplot(data= train_df.groupby('Level')['TotalPay'].agg('mean').reset_index(), x = 'TotalPay',  y ='Level', order=['General','Other','Technician','Entry Level','Experienced Level','Expert','First-Level Management','Middle-Level Management','Top Level Management'])
plt.xticks(rotation=0)
plt.title('Average TotalPay in each Level')
plt.tight_layout()
df_data2 = train_df[~train_df.Status.isin(['NA'])][['Level', 'Status', 'TotalPay']]
df_dataa = df_data2.pivot_table(values='TotalPay', index='Level', columns='Status', aggfunc='count').fillna(0)
status_percents = df_dataa.div(df_dataa.sum(1), axis=0)
plt.figure(figsize=(20,8))
# and plot the bar graph with a stacked argument.  
ax = status_percents.plot(kind='barh', stacked=True, rot=0, figsize=(5,5), color ='gy')
plt.ylabel('Level')
plt.xlabel('Percentage')
plt.xlim(0,1)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,6))
ax = sns.boxplot(y="Level", x="BasePay", data=train_df, order=['General','Other','Technician','Entry Level','Experienced Level','Expert','First-Level Management','Middle-Level Management','Top Level Management'])
plt.show()
plt.figure(figsize=(15,8))
ax = sns.boxplot(y="Level", x="TotalPay", data=train_df, order=['General','Other','Technician','Entry Level','Experienced Level','Expert','First-Level Management','Middle-Level Management','Top Level Management'])
plt.show()
top_16_occupations = train_df.JobType.value_counts().sort_values(ascending=False).head(16).index
salaries_averages_by_occupation = (train_df[train_df.JobType.isin(top_16_occupations)].
                                   groupby('JobType')[['BasePay', 'Benefits', 'OvertimePay', 'OtherPay']].
                                   aggregate('mean'))

ax = salaries_averages_by_occupation.plot(kind='bar', figsize=(20,8))
ax.set_title('Average salaries by job type (top 16 job type popular)')
ax.set_xlabel('Mean Compensation ')
salary_percents = salaries_averages_by_occupation.div(salaries_averages_by_occupation.sum(1), axis=0)
plt.figure(figsize=(20,8))
# and plot the bar graph with a stacked argument.  
ax = salary_percents.plot(kind='bar', stacked=True, rot=90, figsize=(15,8))
plt.xlabel('Job Type')
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()
level = train_df.Level.value_counts().sort_values(ascending=False).head(13).index
salaries_averages_by_level = (train_df[train_df.Level.isin(level)]
                                   .groupby('Level')[['BasePay', 'Benefits', 'OvertimePay', 'OtherPay']]
                                   .aggregate('mean')
)

ax = salaries_averages_by_level.plot(kind='bar', figsize=(10,8))
ax.set_title('Average salaries by level')
ax.set_ylabel('Value')
ax.set_xlabel('Level')
salary_percents_by_level = salaries_averages_by_level.div(salaries_averages_by_level.sum(1), axis=0)
plt.figure(figsize=(25,15))
# and plot the bar graph with a stacked argument.  
ax = salary_percents_by_level.plot(kind='bar', stacked=True, rot=90, figsize=(10,8), color='gyrb')
plt.xlabel('Level')
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()
corr_matrix = df_out._get_numeric_data().sample(frac=0.1).corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_matrix, vmax=1, annot=True, square=True);
plt.show()
# plt.figure(figsize=(6,6))
# plt.scatter(df_no_missing['TotalPayBenefits'], df_no_missing['TotalPay'])
# plt.xlabel("TotalPayBenefits")
# plt.ylabel('TotalPay')
# plt.title("Relationship between TotalPayBenefits and TotalPay")
# plt.show()
# # Create correlation matrix
# corr_matrix = corr_matrix.abs()
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# # Find index of feature columns with correlation greater than 0.95
# to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
# train_df = df_out.drop(columns=to_drop_corr)
# train_df.columns
train_df.columns
train_df['LastNameLength'] = (train_df.EmployeeName.str.split().apply(lambda a: len(a[-1])))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
train_df.LastNameLength.hist(bins=20, ax=ax, alpha=0.3)
plt.title('Histograme of last name length')

t, pvalue = stats.shapiro(train_df['LastNameLength'])
print(pvalue)
# Approach using p-value
if pvalue > 0.05:
    print('Failed to reject the null hypothesis')
else:
    print('Reject the null hypothesis')
plt.figure(figsize=(6, 6))
plt.scatter(train_df['LastNameLength'], train_df['TotalPay'], color='g')
plt.xlim(0,)
plt.ylim(0,)
plt.show()
pearson_coef, p_value = stats.pearsonr(train_df['LastNameLength'], train_df['TotalPay'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a p-value of p =", p_value)  
plt.figure(figsize=(8, 8))
sns.scatterplot(train_df['TotalPay'], train_df['Benefits'], color ='g')
plt.xlabel('TotalPay')
plt.ylabel('Benefits')
plt.title('Relationship between TotalPay and Benefits')
plt.show()
# train_df['bnf_bp_ration'] = train_df['Benefits']/train_df['BasePay']
# train_df[train_df.JobTitle =='senior deputy sheriff']
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = train_df.drop(columns='Year')._get_numeric_data()
x = data['TotalPay']
y= data['Benefits']

# create new plot and data
plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'o--')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for optimal k')
plt.show()
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# normData = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
centroids = np.round(kmeans.cluster_centers_,2)
clustering = kmeans.predict(data)
print(centroids)
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# normData = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(normData)
# centroids = np.round(kmeans.cluster_centers_,2)
# clustering = kmeans.predict(normData)
# print(centroids)
output = []
for x in clustering:
    if x not in output:
        output.append(x)
print(output)
# normData.head()
train_df['cluster'] = pd.Series(clustering, index=data.index)
train_df.head()
train_df.columns
print(centroids)
pd.crosstab(train_df.JobType, train_df.cluster)
level_df = pd.crosstab(train_df.Level, train_df.cluster)
level_df
from sklearn.cluster import KMeans
X=data.values[1:,:]
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
y_km
X
#X=data
# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.xlabel('TotalPay')
plt.ylabel('Benefits')
plt.grid()
plt.show()
# # Feature Importance
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.ensemble import RandomForestRegressor

# # fit an Extra Trees model to the data
# model = RandomForestRegressor()
# model.fit(train_df.drop(columns=['TotalPay']), train_df.TotalPay)
# feature_importances = pd.DataFrame(model.feature_importances_,
#                                    index = train_df.drop(columns=['TotalPay']).columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)
# ft = list(feature_importances[:20].index)
# print(ft)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df[['LastNameLength']], train_df.TotalPay, test_size=0.2, random_state=42)

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('R-squared = ', r2_score(y_test, y_pred))
print('MSE       = ', mean_squared_error(y_test, y_pred))