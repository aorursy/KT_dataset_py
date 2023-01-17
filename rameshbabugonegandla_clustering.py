from IPython.display import Image
import os
#!ls ../input/
Image("../input/clusterimages/CoverPage.PNG")
# Loading Libraries
import pandas as pd # for data analysis
import numpy as np # for scientific calculation
import seaborn as sns # for statistical plotting
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
#Reading market segment data set.
import os
for dirname, _, filenames in os.walk('/kaggle/input/marketsegment-sns/marketsegment.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

marketsegment_eda=pd.read_csv('/kaggle/input/marketsegment-sns/marketsegment.csv')
marketsegment_eda.describe()
marketsegment_eda.shape
marketsegment_eda.info()
marketsegment_eda.head()
marketsegment_eda.isnull().sum()
# Identify Duplicate Records
duplicate_records = marketsegment_eda[marketsegment_eda.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(len(duplicate_records))
print(marketsegment_eda.shape)
print(duplicate_records.head(2))
# dropping duplicate values 
marketsegment_eda.drop_duplicates(keep=False,inplace=True) 
#Validate duplicate records after dropping duplicate record rows.
duplicate_records = marketsegment_eda[marketsegment_eda.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(len(duplicate_records))
print(marketsegment_eda.shape)
print(marketsegment_eda.head(2))
# https://www.kaggle.com/pavansanagapati/comprehensive-feature-engineering-tutorial
# Missing Data Percentage
total = marketsegment_eda.isnull().sum().sort_values(ascending=False)
percent = (marketsegment_eda.isnull().sum()/marketsegment_eda.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of Missing Values', fontsize=15)
plt.title('Percentage of Missing Data by Feature', fontsize=15)
missing_data.head()
# https://thispointer.com/pandas-drop-rows-from-a-dataframe-with-missing-values-or-nan-in-columns/
# Drop rows with any NaN in the selected columns only
print(marketsegment_eda.shape)
marketsegment_eda = marketsegment_eda.dropna(how='any', subset=['gender'])
print(marketsegment_eda.shape)
print("Contents of the Modified Dataframe : ")
print(marketsegment_eda.head(2))
total = marketsegment_eda.isnull().sum().sort_values(ascending=False)
percent = (marketsegment_eda.isnull().sum()/marketsegment_eda.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(3))
# Verify Age count based on gradyear.
marketsegment_eda.groupby(by=['gradyear', 'gender'])['age'].mean()
# Below apply_age function is used to assign missing age value calculate w.r.t to mean age value based on gradyear and Sex.
# To maintain accuracy of missing Age value. In real time it is very difficult to calculate missing Age value in case of
# DateOfBirth value column details were missing in the given dataset. To maintain consistency written below function.
def apply_age(gradyear,sex):
    if(gradyear==2006):
        age=19
    elif (gradyear==2007):
        age=18
    elif (gradyear==2008 and sex=='F'):
        age=17
    elif (gradyear==2008 and sex=='M'):
        age=18
    elif (gradyear==2007):
        age=17
    else:
        age=18 # mean age considered from describe()
    return age

#print(apply_age(1,'male'))
# Filling missing values of age column.
age_nulldata=marketsegment_eda[marketsegment_eda['age'].isnull()]
age_nulldata['age'] = age_nulldata.apply(lambda row : apply_age(row['gradyear'],row['sex']), axis = 1) 
marketsegment_eda['age'].fillna(value=age_nulldata['age'],inplace=True)
# total number of missing Age value count is 263.
marketsegment_eda['age'].isnull().sum()
# https://www.kaggle.com/viratkothari/eda-worldwide-meat-consumption-analysis
# Learnt it from @Virat Kothari article
# Analysis of Non-numerical columns
marketsegment_eda.describe(include=['O'])
# Gender Types
print(marketsegment_eda['gender'].unique())
print("Type of Gender: %s" % (marketsegment_eda['gender'].nunique()))
print(marketsegment_eda['gender'].value_counts())
# Generated HeatMap.
# Observation: From the below heatmap it is difficult to do correlation analysis between the variables.
# Let's see next observation after EDA and feature engineering.
corr = marketsegment_eda.corr()
ax = sns.heatmap( corr,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');



marketsegment_eda.head(2)
#https://www.geeksforgeeks.org/combining-multiple-columns-in-pandas-groupby-with-dictionary/
#Sports
groupby_dict_sports = {"basketball":"Sports","football":"Sports","soccer":"Sports","softball":"Sports",
 "volleyball":"Sports","swimming":"Sports","cheerleading":"Sports","baseball":"Sports","tennis":"Sports","sports":"Sports"}
marketsegment_eda['Sports'] = marketsegment_eda.groupby(groupby_dict_sports, axis = 1).sum() 
#Religion
groupby_dict_religion = {"god":"Religion","church":"Religion","jesus":"Religion","bible":"Religion","hollister":"Religion"}
marketsegment_eda['Religion'] = marketsegment_eda.groupby(groupby_dict_religion, axis = 1).sum() 
#Music
groupby_dict_music = {"dance":"Music","band":"Music","music":"Music","rock":"Music"}
marketsegment_eda['Music'] = marketsegment_eda.groupby(groupby_dict_music, axis = 1).sum() 
#Others
groupby_dict_others = {"cute":"Others","sexy":"Others","hot":"Others","kissed":"Others","marching":"Others","hair":"Others",
                       "dress":"Others","blonde":"Others","mall":"Others","shopping":"Others","clothes":"Others",
                       "abercrombie":"Others","die":"Others","death":"Others","drunk":"Others","drugs":"Others"}
marketsegment_eda['Others'] = marketsegment_eda.groupby(groupby_dict_others, axis = 1).sum() 
print(marketsegment_eda.groupby('gender')['Sports'].sum())
print(marketsegment_eda.groupby('gender')['Religion'].sum())
print(marketsegment_eda.groupby('gender')['Music'].sum())
print(marketsegment_eda.groupby('gender')['Others'].sum())


marketsegment_eda_final = marketsegment_eda.copy()
marketsegment_eda_final = marketsegment_eda_final.drop(['basketball','football','soccer','softball','volleyball','swimming','cheerleading','baseball','tennis','sports','cute','sex','sexy','hot','kissed','dance','band','marching','music','rock','god','church','jesus','bible','hair','dress','blonde','mall','shopping','clothes','hollister','abercrombie','die','death','drunk','drugs'],axis = 1) 

print(marketsegment_eda_final.head(2))
print(marketsegment_eda_final.shape)
# displaying the datatypes 
display(marketsegment_eda_final.dtypes) 
  
# converting 'age' from float to int 
marketsegment_eda_final['age'] = marketsegment_eda_final['age'].astype(int) 
  
# displaying the datatypes 
display(marketsegment_eda_final.dtypes) 

# Correlation Heat Map
figsize=[10,8]
plt.figure(figsize=figsize)
sns.heatmap(marketsegment_eda_final.corr(),annot=True)
plt.show()
# Analyzing data points using pairplot
sns.pairplot(marketsegment_eda_final, hue='gender')
# Label Encoding for gender
# Returns dictionary having key as category and values as number
def find_category_mappings(marketsegment_eda_final, variable):
    return {k: i for i, k in enumerate(marketsegment_eda_final[variable].unique())}

# Returns the column after mapping with dictionary
def integer_encode(marketsegment_eda_final,variable, ordinal_mapping):
    marketsegment_eda_final[variable] = marketsegment_eda_final[variable].map(ordinal_mapping)

for variable in ['gender']:
    mappings = find_category_mappings(marketsegment_eda_final,variable)
    integer_encode(marketsegment_eda_final, variable, mappings)
    
marketsegment_eda_final.head()
knn_data = marketsegment_eda_final.drop(['gradyear', 'friends','age'], axis =1)
knn_data.describe()
# Removing (statistical) outliers for Sports
Q1 = knn_data.Sports.quantile(0.05)
Q3 = knn_data.Sports.quantile(0.95)
IQR = Q3 - Q1
knn_data = knn_data[(knn_data.Sports >= Q1 - 1.5*IQR) & (knn_data.Sports <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Religion
Q1 = knn_data.Religion.quantile(0.05)
Q3 = knn_data.Religion.quantile(0.95)
IQR = Q3 - Q1
knn_data = knn_data[(knn_data.Religion >= Q1 - 1.5*IQR) & (knn_data.Religion <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Music
Q1 = knn_data.Music.quantile(0.05)
Q3 = knn_data.Music.quantile(0.95)
IQR = Q3 - Q1
knn_data = knn_data[(knn_data.Music >= Q1 - 1.5*IQR) & (knn_data.Music <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Others
Q1 = knn_data.Others.quantile(0.05)
Q3 = knn_data.Others.quantile(0.95)
IQR = Q3 - Q1
knn_data = knn_data[(knn_data.Others >= Q1 - 1.5*IQR) & (knn_data.Others <= Q3 + 1.5*IQR)]
print(knn_data.describe())
print(knn_data.shape)
      
print(knn_data.head(2))
print(knn_data.shape)
from sklearn.cluster import KMeans

ssw=[]
cluster_range=range(1,10)
for i in cluster_range:
    model=KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300, random_state=0)
    model.fit(knn_data)
    ssw.append(model.inertia_)
ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)
plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="cyan")
plt.xlabel("Number of clusters")
plt.ylabel("sum squared within")
plt.title("Elbow method to find optimal number of clusters")
plt.show()
# We'll continue our analysis with n_clusters=5
kmeans=KMeans(n_clusters=5, init="k-means++", n_init=10, random_state = 42)
# Fit the model
k_model=kmeans.fit(knn_data)
## It returns the cluster vectors i.e. showing observations belonging which clusters 
clusters=k_model.labels_
clusters
knn_data['clusters'] = clusters
knn_data['clusters'].value_counts()
knn_data.head(2)
knn_data = knn_data.drop(['gender'], axis =1)
### Visualizing the cluster based on each pair of columns
sns.pairplot(knn_data, hue="clusters", diag_kind="kde")
# Complete linkage
from scipy.cluster.hierarchy import dendrogram, linkage
mergings = linkage(knn_data, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()
from IPython.display import Image
import os
#!ls ../input/
Image("../input/clusterimages/DecisionandConclusion.PNG")