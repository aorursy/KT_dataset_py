# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import scipy.stats as stats



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# df = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv')

df = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv/Salaries.csv')

df.head()
df.columns
df.dtypes
df.info()
df.describe(include='all')
df.head()
from fuzzywuzzy import process

str2Match = "police"

strOptions = "CAPTAIN III (POLICE DEPARTMENT)".split()

Ratios = process.extract(str2Match,strOptions)

# print(Ratios)

# You can also select the string with the highest matching percentage

highest = process.extractOne(str2Match,strOptions)

print(highest[1])
from fuzzywuzzy import process

def fuzzy_job_field(row):

    strOptions = row.lower().split()

    for field, field_key in all_career.items():

        for key in field_key:

            highest= process.extractOne(key,strOptions)

            if highest[1] > 90:

                return field

    return "Other"



def fuzzy_job_level(row):

    strOptions = row.lower().split()

    for field, field_key in all_career.items():

        for key in field_key:

            highest= process.extractOne(key,strOptions)

            if highest[1] > 90:

                return field

    return "Staff"





#Map để Extract

# df['Career'] = df['JobTitle'].map(fuzzy_job_field)

# df['Level'] = df['JobTitle'].map(fuzzy_job_level)

df.head()
all_career = dict({

    'Fire': ['fire'],

    'Police': ['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant'],

    'Transit': ['mta', 'transit', 'truck'],

    'Medical': ['anesth', 'medical', 'nurs', 'health', 'physician',

             'orthopedic', 'health', 'pharm', 'care'],

    'Airport': ['airport'],

    'Animal': ['animal'],

    'Architectural': ['architect'],

    'Court': ['court', 'legal'],

    'Mayor': ['mayor'],

    'Library': ['librar'],

    'Parking': ['parking'],

    'Public Works': ['public works'],

    'Attorney': ['attorney'],

    'Mechanic': ['mechanic', 'automotive'],

    'Custodian': ['custodian'],

    'Engineering': ['engineer', 'engr', 'eng', 'program'],

    'Accounting': ['account'],

    'Gardening': ['gardener'],

    'General Laborer': ['general laborer', 'painter', 'inspector',

                     'carpenter', 'electrician', 'plumber', 'maintenance',

                        'custodian', 'garden', 'guard', 'clerk', 'porter'],

    'Food Service': ['food serv'],

    'Clerk': ['clerk'],

    'Porter': ['porter'],

    'Aide': ['aide', 'assistant', 'secretary', 'attendant'],

    'Data': ['analyst', 'data'],

    'Airport': ['airport'],

    'Architect': ['architect'],

    'Accountant': ['Accountant'],

    'Mayoral': ['mayoral'],

    'Recreation': ['recreation'], 

    'Admin': ['Admin', 'account'], 

    'Lawyer': ['attorney', 'lawyer'],

    'Public Service': ['public service', 'Social Worker'],

    'Food Service': ['food serv'],

    'Not provided':['not provide']

})



all_level = dict({

    'Manager': ['manager', 'chief'],

    'Senior': ['senior'],

    'Junior': ['Junior'],

    'Trainee': ['trainee'],

    'Not provided':['not provide']

})



def find_job_field(row):

    for field, field_key in all_career.items():

        for key in field_key:

            if key in row.lower():

                return field

    return "Other"



def find_job_level(row):

    for field, field_key in all_level.items():

        for key in field_key:

            if key in row.lower():

                return field

    return "Staff"



def fuzzy_job_field(row):

    strOptions = row.lower().split()

    for field, field_key in all_career.items():

        for key in field_key:

            highest= process.extractOne(key,strOptions)

            if highest[1] > 90:

                return field

    return "Other"



def fuzzy_job_level(row):

    strOptions = row.lower().split()

    for field, field_key in all_career.items():

        for key in field_key:

            highest= process.extractOne(key,strOptions)

            if highest[1] > 90:

                return field

    return "Staff"



#Map để Extract

df['Career'] = df['JobTitle'].map(find_job_field)

df['Level'] = df['JobTitle'].map(find_job_level)

# df['Career'] = df['JobTitle'].map(fuzzy_job_field)

# df['Level'] = df['JobTitle'].map(fuzzy_job_level)



# df[df['JobTitle'].str.lower().str.contains('food serv')].JobTitle



df.head()
all_pay_columns = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits',

                   'TotalPay', 'TotalPayBenefits']



pay_columns = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']
#Loại bỏ những cột mà được khai là "Not provide" (không có giá trị gì)

print('Number of Not Provided ', df[(df=='Not provided').any(axis=1)].shape[0])

df.drop(df[(df=='Not provided').any(axis=1)].index, inplace=True)
# convert the pay columns to numeric

for col in all_pay_columns:

    df[col] = pd.to_numeric(df[col], errors='coerce')
df.describe()
print('Total sample with Negative value is ', df[(df[pay_columns] < 0).any(axis=1)].shape[0])

df[(df[all_pay_columns] < 0).any(axis=1)].head()

print('Data shape before remove Negative sample ', df.shape)

df.drop(df[(df[all_pay_columns] < 0).any(axis=1)].index, inplace=True)

print('Data shape after remove Negative sample ', df.shape)

    

df.describe()
df.isnull().sum()
is_null = df.isnull().sum()

is_null = is_null[is_null>0]

is_null.sort_values(inplace=True, ascending=False)



#missing data

total = is_null

percent = is_null/len(df) * 100



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



# plot missing data percent again



print(missing_data.index)

plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

g = sns.barplot(x=missing_data.index, y='Percent', data=missing_data)



plt.xticks(rotation = 90)

plt.title("Actual Percentage of missing values.")

plt.xticks(rotation=45)



#plot value on top of bar

for p in range(len(missing_data)):

  value = missing_data.iloc[p, 1]

  g.text(p, value, f'{value:1.2f}%', color='black', ha="center")



plt.show()
df["BasePay"].fillna(df.groupby("Career")["BasePay"].transform("median"), inplace=True)

df["Benefits"].fillna(df.groupby("Career")["Benefits"].transform("median"), inplace=True)
df['Career'].isnull().sum()
is_null = df.isnull().sum()

is_null = is_null[is_null>0]

is_null.sort_values(inplace=True, ascending=False)



#missing data

total = is_null

percent = is_null/len(df) * 100



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



# plot missing data percent again



print(missing_data.index)

plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

g = sns.barplot(x=missing_data.index, y='Percent', data=missing_data)



plt.xticks(rotation = 90)

plt.title("Actual Percentage of missing values.")

plt.xticks(rotation=45)



#plot value on top of bar

for p in range(len(missing_data)):

  value = missing_data.iloc[p, 1]

  g.text(p, value, f'{value:1.2f}%', color='black', ha="center")



plt.show()
all_pay_columns
# fig, axes = plt.subplots(6, 1, figsize=(8,24))



# for i in range(6):

#     df[all_pay_columns[i]].hist(bins=100, ax = axes[i])

#     axes[i].set_title(all_pay_columns[i])



# plt.show()
sns.pairplot(df[pay_columns])

# sns.pairplot(df[pay_columns], kind='reg')
df_sal = df[all_pay_columns]

# df_sal = df[pay_columns]





# Scale the data using the natural logarithm

df_log_sal = np.log1p(df_sal)



# Produce a scatter matrix for each pair of newly-transformed features

# sns.pairplot(df[pay_columns], kind='reg')

# sns.pairplot(df_log_sal, diag_kind = 'kde', kind='reg')



fig, axes = plt.subplots(6, 2, figsize=(12,24))

for i in range(6):

    df[all_pay_columns[i]].hist(bins=100, ax = axes[i, 0])

    df_log_sal[all_pay_columns[i]].hist(bins=100, ax = axes[i, 1])

#     axes[i, 0].set_title(all_pay_columns[i])

#     axes[i, 1].set_title(all_pay_columns[i])

    axes[i, 0].set_xlabel(f'Original {all_pay_columns[i]}')

    axes[i, 1].set_xlabel(f'Log {all_pay_columns[i]}')



plt.show()
# For each feature find the data points with extreme high or low values

import collections 



outliers_index_all = []

df_find_outliers = df_log_sal

for feature in df_find_outliers.keys():

    

    # TODO: CalAculate Q1 (25th percentile of the data) for the given feature

    Q1 = df_find_outliers[feature].quantile(0.25)

    

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = df_find_outliers[feature].quantile(0.75)

    

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = (Q3 - Q1) * 1.5

    

    # Display the outliers

    feature_outliers = df_find_outliers[~((df_find_outliers[feature] >= Q1 - step) & (df_find_outliers[feature] <= Q3 + step))]

    print(f"{feature_outliers.shape[0]} Data points considered outliers for the feature '{feature}':")

#     display(feature_outliers)

   

# OPTIONAL: Select the indices for data points you wish to remove

    outliers_index_all  += feature_outliers.index.tolist()



print("\nTotal outliers are" , len(set(outliers_index_all)))

# print("\nFollowing index were found as outliers in more than one features")

# print([(item,count) for item, count in collections.Counter(outliers_index_all).items() if count > 1])



outliers_index = list(set(outliers_index_all))

# Remove the outliers, if any were specified

df_log_sal_no_outlier = df_find_outliers.drop(outliers_index) #.reset_index(drop = True)
# For each feature find the data points with extreme high or low values

import collections 



outliers_index_all = []

df_find_outliers = df_sal

for feature in df_find_outliers.keys():

    

    # TODO: CalAculate Q1 (25th percentile of the data) for the given feature

    Q1 = df_find_outliers[feature].quantile(0.25)

    

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = df_find_outliers[feature].quantile(0.75)

    

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = (Q3 - Q1) * 1.5

    

    # Display the outliers

    feature_outliers = df_find_outliers[~((df_find_outliers[feature] >= Q1 - step) & (df_find_outliers[feature] <= Q3 + step))]

    print(f"{feature_outliers.shape[0]} Data points considered outliers for the feature '{feature}':")

#     display(feature_outliers)

   

# OPTIONAL: Select the indices for data points you wish to remove

    outliers_index_all  += feature_outliers.index.tolist()



print("\nTotal outliers are" , len(set(outliers_index_all)))

# print("\nFollowing index were found as outliers in more than one features")

# print([(item,count) for item, count in collections.Counter(outliers_index_all).items() if count > 1])



outliers_index = list(set(outliers_index_all))

# Remove the outliers, if any were specified

df_sal_no_outlier = df_find_outliers.drop(outliers_index) #.reset_index(drop = True)
fig, axes = plt.subplots(6, 2, figsize=(12,24))

for i in range(6):

    df_sal[all_pay_columns[i]].hist(bins=100, ax = axes[i, 0])

    df_sal_no_outlier[all_pay_columns[i]].hist(bins=100, ax = axes[i, 1])

#     axes[i, 0].set_title(all_pay_columns[i])

#     axes[i, 1].set_title(all_pay_columns[i])

    axes[i, 0].set_xlabel(f'Original {all_pay_columns[i]}')

    axes[i, 1].set_xlabel(f'Removed outlier {all_pay_columns[i]}')



plt.show()
fig, axes = plt.subplots(6, 2, figsize=(12,24))

for i in range(6):

    df_log_sal[all_pay_columns[i]].hist(bins=100, ax = axes[i, 0])

    df_log_sal_no_outlier[all_pay_columns[i]].hist(bins=100, ax = axes[i, 1])

#     axes[i, 0].set_title(all_pay_columns[i])

#     axes[i, 1].set_title(all_pay_columns[i])

    axes[i, 0].set_xlabel(f'Log of {all_pay_columns[i]}')

    axes[i, 1].set_xlabel(f'Removed outlier Log of {all_pay_columns[i]}')



plt.show()
from sklearn.decomposition import PCA
df_clean = df.loc[df_log_sal_no_outlier.index]

df_model = df_clean[pay_columns]
# Apply PCA by fitting the good data with only two dimensions

df_pca = df_model

pca = PCA(n_components=2).fit(df_pca)



# Transform the good data using the PCA fit above

reduced_data = pca.transform(df_pca)



# Create a DataFrame for the reduced data

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'], index=df_pca.index)



# Generate PCA results plot

pca_table = pd.DataFrame(np.round(pca.components_, 4), columns = list(df_pca.keys()))

print('% Variance ', pca.explained_variance_ratio_.cumsum())

pca_table.plot.bar(figsize=(15,8))
def biplot(good_data, reduced_data, pca, feature_scale_ratio = 1):

    '''

    Produce a biplot that shows a scatterplot of the reduced

    data and the projections of the original features.

    

    good_data: original data, before transformation.

               Needs to be a pandas dataframe with valid column names

    reduced_data: the reduced data (the first two dimensions are plotted)

    pca: pca object that contains the components_ attribute



    return: a matplotlib AxesSubplot object (for any additional customization)

    

    This procedure is inspired by the script:

    https://github.com/teddyroland/python-biplot

    '''



    fig, ax = plt.subplots(figsize = (14,8))

    # scatterplot of the reduced data    

    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 

        facecolors='b', edgecolors='b', s=70, alpha=0.5)

    

    feature_vectors = pca.components_.T * feature_scale_ratio



    # we use scaling factors to make the arrows easier to see

    arrow_size, text_pos = 7.0, 8.0,



    # projections of the original features

    for i, v in enumerate(feature_vectors):

        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 

                  head_width=0.2, head_length=0.2, linewidth=2, color='red')

        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 

                 ha='center', va='center', fontsize=18)



    ax.set_xlabel("Dimension 1", fontsize=14)

    ax.set_ylabel("Dimension 2", fontsize=14)

    ax.set_title("PC plane with original feature projections.", fontsize=16);

    return ax
biplot(df_pca, reduced_data, pca, 10000)
plt.figure(figsize=(16,5))

sns.countplot('Career', data = df, order = df['Career'].value_counts().index)

plt.xticks(rotation = 45)

plt.tight_layout()
plt.figure(figsize=(16,5))

sns.countplot('Career', data = df, order = df['Career'].value_counts().index, hue = 'Year')

plt.xticks(rotation = 45)

plt.tight_layout()
plt.figure(figsize=(12,5))

sns.barplot(data= df.groupby('Career')['TotalPayBenefits'].agg('median').reset_index(), x ='Career', y = 'TotalPayBenefits')

plt.xticks(rotation=85)

plt.title('Mean Pay')
df_input = df.loc[df_log_sal_no_outlier.index]

top_ten_occupations = df_input['Career'].value_counts().sort_values(ascending=False)[:10].index

top_ten_occupations
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('mean')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Mean Pay of Top 10 Polular Career ')
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('median')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Median Pay of Top 10 Polular Career ')
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('min')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Min Pay of Top 10 Polular Career ')
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('max')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Max Pay of Top 10 Polular Career ')
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('var')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Var Pay of Top 10 Polular Career ')
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('mean')).sort_values('BasePay', 0)



ax.set_xlabel('Mean Pay of Top 10 Polular Career ')

# the above graph can be transformed into a proportions stacked bar graph



# use the dataframe method div to proportionalize the values by axis=0(row)

salary_percents = salaries_averages_by_occupation.div(salaries_averages_by_occupation.sum(1), 

                                                      axis=0)



# and plot the bar graph with a stacked argument.  

ax = salary_percents.plot(kind='bar', stacked=True, rot=90)
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('median')).sort_values('BasePay', 0)



ax.set_xlabel('Mean Pay of Top 10 Polular Career ')

# the above graph can be transformed into a proportions stacked bar graph



# use the dataframe method div to proportionalize the values by axis=0(row)

salary_percents = salaries_averages_by_occupation.div(salaries_averages_by_occupation.sum(1), 

                                                      axis=0)



# and plot the bar graph with a stacked argument.  

ax = salary_percents.plot(kind='bar', stacked=True, rot=90)
salaries_averages_by_occupation = (df_input[df_input.Career.isin(top_ten_occupations)]

                                   .groupby('Career')[pay_columns]

                                   .aggregate('mean')).sort_values('BasePay', 0)



ax = salaries_averages_by_occupation.plot(kind='barh', figsize=(8,8))



ax.set_xlabel('Mean Pay of Top 10 Polular Career ')
sal_feauture = 'BasePay'

group_data = df[df['Career'] == 'Data'][sal_feauture]

group_2 = df[df['Career'] == 'Engineering'][sal_feauture]



print("Data group vs Engineering group")

stats.ttest_ind(group_data, group_2)
sal_feauture = 'TotalPay'

group_data = df[df['Career'] == 'Data'][sal_feauture]

group_medical = df[df['Career'] == 'Medical'][sal_feauture]



print("Data group vs Medical group")



stats.ttest_ind(group_data, group_medical)
sal_feauture = 'TotalPay'

group_1 = df[df['Career'] == 'Police'][sal_feauture]

group_2 = df[df['Career'] == 'Fire'][sal_feauture]



#Null

print("Police group vs Fire group")

stats.ttest_ind(group_1, group_2)
sal_feauture = 'BasePay'

group_1 = df[df['Career'] == 'Data'][sal_feauture]

group_2 = df[df['Career'] == 'Aide'][sal_feauture]



print("Data group vs Aide group")

stats.ttest_ind(group_1, group_2)
df.columns
career_group = []

temp = top_ten_occupations

# temp = df['Career'].unique()

sal_feauture = 'TotalPay'

for i in temp:

    df_temp = df[df['Career']== i]

#     df_temp['Overtime/Total'] = df_temp['OvertimePay'] / df_temp['TotalPay']

    result = df_temp['TotalPay'].dropna()

    career_group.append(result)
stats.levene(*career_group)
stats.f_oneway(*career_group)
!pip install kneed
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from kneed import KneeLocator



# k means determine k

# distortions = []

# inertia = []

# K = range(1,10)

# X = df_model

# for k in K:

#     kmeanModel = KMeans(n_clusters=k).fit(X)

#     kmeanModel.fit(X)

#     #append error

#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

#     inertia.append(kmeanModel.inertia_)



# kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')

# kn2 = KneeLocator(list(K), inertia, S=1.0, curve='convex', direction='decreasing')



# # Plot the elbow

# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))

# fig.suptitle('The Elbow Method showing the optimal k')



# ax1.plot(K, distortions, 'bx-')

# ax1.set_xlabel('k')

# ax1.set_ylabel('Distortion')

# ax1.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')



# ax2.plot(K, distortions, 'bx-')

# ax2.set_xlabel('k')

# ax2.set_ylabel('Distortion')

# ax2.vlines(kn2.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')



# print("Number of clusters by distortions", kn.knee)

# print("Number of clusters by inertia", kn2.knee)
url = '/kaggle/input/sf-kmean/kmean1_9.png'

Image(url)
url = '/kaggle/input/sf-kmean/kmean10_15.png'

Image(url)
kmeans_centers
df_input = df_model



kmeans = KMeans(n_clusters=4)

kmeans.fit(df_input)



kmeans_centers = pd.DataFrame(kmeans.cluster_centers_, columns=df_input.columns)

kmeans_centers_reduced = pd.DataFrame(pca.transform(kmeans_centers), columns = ['Dimension 1', 'Dimension 2'])

kmeans_centers = pd.concat([kmeans_centers, kmeans_centers_reduced], axis=1)



labels = pd.DataFrame(kmeans.labels_, columns=['Label'], index = df_input.index)

career_labels = df.loc[df_input.index]['Career']



sal_centers = kmeans_centers

clustered_data = pd.concat([df_input, reduced_data, career_labels, labels], axis=1)



display(sal_centers.head())

clustered_data.head()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))



ax1.set_title('K Means')

sns.scatterplot(x='Dimension 1', y='Dimension 2',hue = 'Label',data=clustered_data, ax =ax1)

ax1.scatter(x='Dimension 1', y='Dimension 2', marker="X", c='r', s = 100, data=kmeans_centers)



ax2.set_title("Original")

sns.scatterplot(x='Dimension 1', y='Dimension 2',hue = 'Career',data=clustered_data, ax =ax2)
sal_centers
display(sal_centers.head())

#BasePay very High, OT Average

#BasePay High, OT High

#BasePay Average, OT Average

#BasePay Low, OT Low
fig, axes = plt.subplots(2, 2, figsize=(12,8))

for i in range(2):

    df[pay_columns[i]].hist(bins=100, ax = axes[i, 0])

    df_input[pay_columns[i]].hist(bins=100, ax = axes[i, 1])

#     axes[i, 0].set_title(all_pay_columns[i])

#     axes[i, 1].set_title(all_pay_columns[i])

    axes[i, 0].set_xlabel(f'Original {all_pay_columns[i]}')

    axes[i, 1].set_xlabel(f'Removed outlier Log of {all_pay_columns[i]}')



axes[1,0].set_xlim([0, 40000])

axes[1,1].set_xlim([0, 40000])



plt.show()
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score



df_input = df_model

GM_clusterer = GaussianMixture(4, random_state=0).fit(df_input)

# GM_clusterer.fit(df_input)



GM_centers = pd.DataFrame(GM_clusterer.means_, columns=df_input.columns)

GM_centers_reduced = pd.DataFrame(pca.transform(GM_centers), columns = ['Dimension 1', 'Dimension 2'])

GM_centers = pd.concat([GM_centers, GM_centers_reduced], axis=1)



GM_labels = pd.DataFrame(GM_clusterer.predict(df_input), columns=['GM_labels'], index = df_input.index)



clustered_data = pd.concat([clustered_data, GM_labels], axis=1)



display(GM_centers.head())

clustered_data.head()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))



ax1.set_title('GM')

sns.scatterplot(x='Dimension 1', y='Dimension 2',hue = 'GM_labels',data=clustered_data, ax =ax1)

ax1.scatter(x='Dimension 1', y='Dimension 2', marker="X", c='r', s = 100, data=GM_centers)



ax2.set_title("Original")

sns.scatterplot(x='Dimension 1', y='Dimension 2',hue = 'Career',data=clustered_data, ax =ax2)
# from sklearn.mixture import GaussianMixture

# from sklearn.metrics import silhouette_score



# def clusterGMM(input_df, k):

#     global clusterer, preds, centers, sample_preds

#     # Apply your GMM algorithm to the reduced data 

#     clusterer = GaussianMixture(k, random_state=0).fit(input_df)



#     # Predict the cluster for each data point

#     preds = clusterer.predict(input_df)



#     # Find the cluster centers

#     centers = clusterer.means_



#     # Predict the cluster for each transformed sample data point

# #     sample_preds = clusterer.predict(pca_samples)



#     # Calculate the mean silhouette coefficient for the number of clusters chosen

#     score = silhouette_score(input_df, preds)

#     return score



# results = pd.DataFrame(columns=['Silhouette Score'])

# results.columns.name = 'Number of Clusters'



# for k in range(2,16):

#     score = clusterGMM(reduced_data, k)

#     results = results.append(pd.DataFrame([score], columns=['Silhouette Score'], index=[k]))

# results
