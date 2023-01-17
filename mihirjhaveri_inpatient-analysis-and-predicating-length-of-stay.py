import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandasql import *
import requests
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import plot_confusion_matrix
from collections import Counter 
from imblearn.under_sampling import RandomUnderSampler
import os
print(os.listdir("../input"))
import warnings            
warnings.filterwarnings("ignore") 
hos_in_pt_dis_df = pd.read_csv('../input/2015-deidentified-ny-inpatient-discharge-sparcs/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv')
hos_in_pt_dis_df.head()
display(hos_in_pt_dis_df)
hos_in_pt_dis_df.dtypes
hos_in_pt_dis_df.describe()
hos_in_pt_dis_df.isna().sum()
#Lets drop Other Provider License Number since over a almost 2/3 of these rows have NaN values
#Lets also drop the other columns with a significant amount on NaNs since it will be difficult to extract value from them
hos_df = hos_in_pt_dis_df.drop(['Other Provider License Number','Payment Typology 2','Payment Typology 3','Operating Provider License Number'], axis=1)


hos_df.isna().sum()
#next, lets drop rows with NaNs for columns such as the APR Severity of Illness Description and APR Risk of Mortality since these have a 
#relatively small number of NaN values and thus will be insignificant to drop.
hos_df = hos_df.dropna(subset=['APR Severity of Illness Description', 'APR Risk of Mortality','Zip Code - 3 digits','Facility Id','Attending Provider License Number'])
hos_df.isna().sum()
#Here, the apply operation is used as an alternative to an if statement for highest computational efficiency
#Convert number objects
hos_df['Length of Stay'] = hos_df['Length of Stay'].apply(lambda x: str(x).split(' ')[0])
hos_df['Length of Stay'] = pd.to_numeric(hos_df['Length of Stay'])
hos_df['Total Costs'] = hos_df['Total Costs'].apply(lambda x: str(x).replace('$',''))
hos_df['Total Costs'] = pd.to_numeric(hos_df['Total Costs'])
hos_df['Total Charges'] = hos_df['Total Charges'].apply(lambda x: str(x).replace('$',''))
hos_df['Total Charges'] = pd.to_numeric(hos_df['Total Charges'])
#upon inspection, I also found that some entries in the zip code column had the string OOS instead of a number. 67,000 rows had this
#which seems to large to simply drop these rows. Looking into the information about the dataset, these zipcodes refer to out of state. 
#This could be useful because these people might be rich so there might be differences in length of stay
#Thus, I will keep these rows and signify them with a 999, which now indicates out of state
hos_df['Zip Code - 3 digits'] = hos_df['Zip Code - 3 digits'].apply(lambda x: str(x).replace('OOS','999'))  
hos_df['Zip Code - 3 digits'] = pd.to_numeric(hos_df['Zip Code - 3 digits'])
display(hos_df)

#Now lets visualize some initial stats on the results of the data cleaning above
hos_df.describe()
#Make a heatmap
hos_df.corr()
f, ax = plt.subplots(figsize=(11, 9))
corr = hos_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
sns.set(font_scale=1.2)
sns.set_style("white")
f, ax = plt.subplots(figsize=(11, 9))
sns.distplot(hos_df['Length of Stay'], norm_hist=False);
plt.ylabel('Probability Density')
plt.title('Univariate Distribution Plot of Length of Stay')
plt.show()
#Lets see if we can confirm that there is data only input for the youngest age group

birth_weight_df = hos_df[['Type of Admission', 'Birth Weight']].groupby('Type of Admission').mean()
display(birth_weight_df)
weight_age_df = hos_df[['Age Group', 'Birth Weight']].groupby('Age Group').mean()
display(weight_age_df)
#Lets look at the relationship between birth weight and length of stay only for the newborns as this feature makes the most sense
#for this group

birth_youngest_stay = hos_df[hos_df['Type of Admission'].str.contains('Newborn')]
birth_youngest_stay['Birth Weight'] = birth_youngest_stay['Birth Weight'].apply(lambda x: float(x/454)) #convert from grams to pounds
f, ax = plt.subplots(figsize=(11, 9))
sns.scatterplot(x="Birth Weight", y="Length of Stay",
                data=birth_youngest_stay)
longest_newborn_df = birth_youngest_stay[birth_youngest_stay['Length of Stay']==120]
display(longest_newborn_df)
f, ax = plt.subplots(figsize=(25, 15))
sns.countplot(x='CCS Procedure Description', data = longest_newborn_df)
plt.xticks(rotation=90)
plt.title('Procedure Descriptions for Newborns Who Stay 120 Days or Longer')
plt.show()
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Gender", y="Length of Stay",
            hue="Race",
            data=hos_df)

f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Gender", y="Length of Stay",
            hue="Race",
            data=hos_df)
ax.set(ylim=(0, 30))
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Age Group", y="Length of Stay",
            data=hos_df)
ax.set(ylim=(0, 30))
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Age Group", y="Length of Stay", order=['0 to 17','18 to 29','30 to 49','50 to 69','70 or Older'],
            palette="Set1", data=hos_df)
ax.set(ylim=(0, 30))
plt.title('Length of Stay vs. Age Group')
plt.show()

f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Race", y="Length of Stay", data=hos_df, palette='Set1')
plt.title('Length of Stay vs. Patient Race')
ax.set(ylim=(0, 30))
plt.show()
f, ax = plt.subplots(figsize=(35, 20))
sns.barplot(x="CCS Diagnosis Description", y="Length of Stay", data=hos_df[['CCS Diagnosis Description','Length of Stay']].groupby('CCS Diagnosis Description', as_index=False).mean())
ax.set(ylim=(0, 20))
plt.xticks(rotation=90)
plt.show()
diag_stay_df = hos_df[['CCS Diagnosis Description','Length of Stay']].groupby('CCS Diagnosis Description', as_index=False).mean()
diag_stay_df = diag_stay_df.sort_values(by='Length of Stay', ascending=False, ignore_index=True)
display(diag_stay_df)
sns.set(font_scale=1.2)
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 9))
sns.barplot(y="CCS Diagnosis Description", x="Length of Stay", data=diag_stay_df[0:5], palette='Set1')
#plt.xticks(rotation=45)
plt.title('Top 5 Diagnoses with Longest Average Length of Stay')
plt.show()
f, ax = plt.subplots(figsize=(30, 15))
sns.barplot(x="CCS Diagnosis Description", y="Total Costs", data=hos_df[['CCS Diagnosis Description','Total Costs']].groupby('CCS Diagnosis Description', as_index=False).mean())
plt.xticks(rotation=90)
plt.show()
diag_costs_df = hos_df[['CCS Diagnosis Description','Total Costs']].groupby('CCS Diagnosis Description', as_index=False).mean()
diag_costs_df = diag_costs_df.sort_values(by='Total Costs', ascending=False, ignore_index=True)
display(diag_costs_df)
#Rank the total costs and then do an inner join
diag_costs_df = diag_costs_df.reset_index()
joined_df = diag_stay_df.merge(right=diag_costs_df, how='inner', on='CCS Diagnosis Description')
display(joined_df)
sns.set(font_scale=1.25)
sns.set_style("white")
f, ax = plt.subplots(figsize=(15, 9))
sns.boxplot(x="Payment Typology 1", y="Length of Stay", data=hos_df, palette='Set1')
#plt.title('Type of Patient Payments vs. Length of Stay')
ax.set(ylim=(0, 20))
plt.xticks(rotation=80)
plt.title('Length of Stay vs. Primary Payment Typology')
plt.show()
f, ax = plt.subplots(figsize=(11, 9))
sns.countplot(x='Age Group', data = hos_df[hos_df['Payment Typology 1']=='Medicare'], order=['0 to 17','18 to 29','30 to 49','50 to 69','70 or Older'], palette='Set1')
plt.title('Number of Medicare Patients in Each Age Group')
plt.show()
f, ax = plt.subplots(figsize=(25, 9))
sns.boxplot(x="Zip Code - 3 digits", y="Length of Stay", data=hos_df)
ax.set(ylim=(0, 20))
#We can scrap the data from the web, but I have downloaded the file and uploaded the same.

#Scraped income data by zipcode from web (data from 2006-2010)

#dls = "https://www.psc.isr.umich.edu/dis/census/Features/tract2zip/MeanZIP-3.xlsx"
#resp = requests.get(dls)

#output = open('zip_incomes.xlsx', 'wb')
#output.write(resp.content)
#output.close()
zip_income_df = pd.read_excel('../input/mean-zip/MeanZIP-3.xlsx')
display(zip_income_df)
zip_income_df['Zip'] = zip_income_df['Zip'].apply(lambda x: int(x))
income_df = zip_income_df[zip_income_df['Zip'] > 9999]
income_df['Zip'] = income_df['Zip'].apply(lambda x: math.floor(x/100)) #cut down zip code to just first three digits
display(income_df)
income_df.dtypes
query = '''SELECT ZIP, AVG(MEDIAN) as median FROM income_df GROUP BY ZIP'''
avg_income = sqldf(query, locals())
display(avg_income)
#Now, we can inner join this onto our dataframe to get the average income for each patient zipcode

avg_income['Zip'] = avg_income['Zip'].astype('object')

query = '''SELECT d.*, a.median as AvgIncome FROM hos_df d inner join avg_income a on d.'Zip Code - 3 digits' = a.Zip'''
hos_sql_df = sqldf(query, locals())
display(hos_sql_df)
f, ax = plt.subplots(figsize=(11, 9))
sns.countplot(x="Payment Typology 1",
            data=hos_sql_df[hos_sql_df['Zip Code - 3 digits']==999])
plt.xticks(rotation=90)
#for this calculation, I will exlclude the data with zipcode equal to 999
df_no999 = hos_sql_df[hos_sql_df['Zip Code - 3 digits'] != 999]
insurance_df = df_no999[df_no999['Payment Typology 1'].isin(['Medicare','Private Health Insurance'])]
mean_zip999 = insurance_df['AvgIncome'].mean()
#round zipcodes to make replace easier
hos_sql_df['AvgIncome'] = hos_sql_df['AvgIncome'].round(2)
hos_sql_df[hos_sql_df['Zip Code - 3 digits']==999]['AvgIncome']
hos_sql_df = hos_sql_df.replace(47010.32, round(mean_zip999,2))
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x="Zip Code - 3 digits", y="AvgIncome",
            data=hos_sql_df, palette='Set1')
plt.ylabel('Median Income')
plt.title('Income Distribution Across 3 Digit Patient Zipcodes')
plt.xticks(rotation=90)
plt.show()
#From left to right we have increasing income, organized this way since the x-axis labels are hard to read
f, ax = plt.subplots(figsize=(25, 9))
sns.boxplot(x="AvgIncome", y="Length of Stay",
            data=hos_sql_df.sort_values(by='AvgIncome', ascending=True))
plt.xticks(rotation=90)
ax.set(ylim=(0, 20))
sns.set(font_scale=1.25)
sns.set_style("white")
f, ax = plt.subplots(figsize=(15, 9))
sns.barplot(x="Payment Typology 1", y="AvgIncome",
            data=hos_sql_df, palette='Set1')
plt.xticks(rotation=80)
plt.title('Median Income vs. Primary Payment Typology')
plt.ylabel('Median Income')
plt.show()

f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Health Service Area", y="Length of Stay",
            data=hos_sql_df)
plt.xticks(rotation=90)
ax.set(ylim=(0, 20))
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="APR Severity of Illness Description", y="Length of Stay",
            data=hos_sql_df, palette='Reds')
ax.set(ylim=(0, 40))
plt.title('Length of Stay vs. Severity of Illness')
plt.show()
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Ethnicity", y="Length of Stay",
            data=hos_sql_df)
ax.set(ylim=(0, 20))
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Type of Admission", y="Length of Stay",
            data=hos_sql_df)
ax.set(ylim=(0, 20))
f, ax = plt.subplots(figsize=(11, 9))
sns.boxplot(x="Type of Admission", y="Length of Stay",
            data=hos_sql_df, palette='Set1')
plt.title('Length of Stay vs. Type of Admission')
ax.set(ylim=(0, 20))
plt.show()
f, ax = plt.subplots(figsize=(11, 9))
sns.regplot(x="Length of Stay", y="Total Costs",
            data=hos_sql_df[0:200000])
f, ax = plt.subplots(figsize=(10, 8))
sns.countplot(x='APR Severity of Illness Description', data = hos_sql_df[hos_sql_df['Length of Stay']==120])
plt.title('Severity of Illness vs. Length of Stay for Patients With 120+ Lengths of Stay')
plt.show()
diagnosis_desc_df = hos_sql_df[['CCS Diagnosis Description']][0:10000]
diagnosis_desc_df['CCS Diagnosis Description'] = diagnosis_desc_df['CCS Diagnosis Description'].astype('str', errors = 'ignore')
diagnosis_desc_df['CCS Diagnosis Description'] = diagnosis_desc_df['CCS Diagnosis Description'].apply(lambda x: x.lower())

list_of_titles = []
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
diagnosis_desc_df['Tokens'] = diagnosis_desc_df['CCS Diagnosis Description'].apply(lambda x: tokenizer.tokenize(x))
list_of_tokens = diagnosis_desc_df['Tokens'].tolist()
diagnosis_words = []
for sublist in list_of_tokens:
    for item in sublist:
        diagnosis_words.append(item)

from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

#Let's remove stop words as well, such as "a", "and", and "the"
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
for word in list(diagnosis_words):
    if word in stop_words:
        diagnosis_words.remove(word)

from collections import Counter
Counter1 = Counter(diagnosis_words)
most_occur = Counter1.most_common(30) 
diagnosis_counter = []
for item in most_occur:
  diagnosis_counter.append(item[0])

#Create word cloud plot
cloud_words = ' '
for words in diagnosis_counter: 
    cloud_words = cloud_words + words + ' '

diagnosis_word_plot = WordCloud(width = 800, height = 800).generate(cloud_words)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(diagnosis_word_plot)
plt.show()
#Minor
hos_sql_df['CCS Diagnosis Description'] = hos_sql_df['CCS Diagnosis Description'].astype('str', errors = 'ignore')
hos_sql_df['CCS Diagnosis Description'] = hos_sql_df['CCS Diagnosis Description'].apply(lambda x: x.lower())
minor_df = hos_sql_df[['CCS Diagnosis Description']][hos_sql_df['APR Severity of Illness Description']=='Minor']
minor_df = minor_df[['CCS Diagnosis Description']][0:40000]

#lets create a function for this
def generate_cloud(type_df):
  list_of_titles = []
  from nltk.tokenize import RegexpTokenizer
  tokenizer = RegexpTokenizer(r'\w+')
  type_df['Tokens'] = type_df['CCS Diagnosis Description'].apply(lambda x: tokenizer.tokenize(x))
  list_of_tokens = type_df['Tokens'].tolist()
  type_words = []
  for sublist in list_of_tokens:
      for item in sublist:
          type_words.append(item)

  for word in list(type_words):
    if word in stop_words:
        type_words.remove(word)

  Counter1 = Counter(type_words)
  most_occur = Counter1.most_common(30) 
  diagnosis_counter = []
  for item in most_occur:
    diagnosis_counter.append(item[0])
  
  from wordcloud import WordCloud
  #Create word cloud plot
  cloud_words = ' '
  for words in diagnosis_counter: 
      cloud_words = cloud_words + words + ' '

  type_word_plot = WordCloud(width = 800, height = 800).generate(cloud_words)
  plt.figure(figsize = (8, 8), facecolor = None) 
  plt.imshow(type_word_plot)
  plt.show()

generate_cloud(minor_df)

moderate_df = hos_sql_df[['CCS Diagnosis Description']][hos_sql_df['APR Severity of Illness Description']=='Moderate']
moderate_df = moderate_df[['CCS Diagnosis Description']][0:40000]

generate_cloud(moderate_df)
major_df = hos_sql_df[['CCS Diagnosis Description']][hos_sql_df['APR Severity of Illness Description']=='Major']
major_df = major_df[['CCS Diagnosis Description']][0:40000]

generate_cloud(major_df)

extreme_df = hos_sql_df[['CCS Diagnosis Description']][hos_sql_df['APR Severity of Illness Description']=='Extreme']
extreme_df = extreme_df[['CCS Diagnosis Description']][0:40000]

generate_cloud(extreme_df)
#Finally, lets drop the columns the rest of the columns we won't need for the modeling portion


fig, ax =plt.subplots(1,2, figsize=(14,5))
sns.countplot(hos_sql_df['APR Severity of Illness Description'], ax=ax[0])
sns.countplot(hos_sql_df['APR Severity of Illness Code'], ax=ax[1])
fig.show()
hos_sql_df_1 = hos_sql_df.drop(['APR Severity of Illness Description'], axis=1)  #after confirming the illness code column encodes the same information

num_county = hos_sql_df_1['Hospital County'].unique().tolist()
num_zip = hos_sql_df_1['Zip Code - 3 digits'].unique().tolist()
print("Number of Hospital County's:",len(num_county))
print("Number of Zipcodes:",len(num_zip))
#Below, we can see hospital county and zipcode do not encode same info.
num_facilities = hos_sql_df_1['Facility Id'].unique().tolist()
num_facname = hos_sql_df_1['Facility Name'].unique().tolist()
print("Number of Facility Ids:",len(num_facilities))
print("Number of Facility Names:",len(num_facname))
#We can see that these most likely encode the same info even though they are on off, so I will drop the names column
num_diag_code = hos_sql_df_1['CCS Diagnosis Code'].unique().tolist()
num_diag_desc = hos_sql_df_1['CCS Diagnosis Description'].unique().tolist()
print("Number of Diagnosis Codes:",len(num_diag_code))
print("Number of Diagnosis Descriptions:",len(num_diag_desc))
#Diagnosis Codes and Descriptions encode the same info so we will drop the descriptions.
print("Types of Procedure Descriptions:",hos_sql_df_1['CCS Procedure Description'].unique().tolist())
#I want to see a list of the possible descriptions since many of them say NO PROC. Below is the output
#Lets verify the procedure code encodes the same info and then drop this column
num_proc_code = hos_sql_df_1['CCS Procedure Code'].unique().tolist()
num_proc_desc = hos_sql_df_1['CCS Procedure Description'].unique().tolist()
print("Number of Procedure Codes:",len(num_proc_code))
print("Number of Procedure Descriptions:",len(num_proc_desc))
#They do contain the same information, so lets drop the descriptions
num_drg_code = hos_sql_df_1['APR DRG Code'].unique().tolist()
num_drg_desc = hos_sql_df_1['APR DRG Description'].unique().tolist()
print("Number of DRG Codes:",len(num_drg_code))
print("Number of DRG Descriptions:",len(num_drg_desc))
#Same number of unique values, so drop descriptions
num_mdc_code = hos_sql_df_1['APR MDC Code'].unique().tolist()
num_mdc_desc = hos_sql_df_1['APR MDC Description'].unique().tolist()
print("Number of MDC Codes:",len(num_mdc_code))
print("Number of MDC Descriptions:",len(num_mdc_desc))
#Same number of unique values, so drop descriptions
print("Number of Attending Provider License Numbers:",len(hos_sql_df_1['Attending Provider License Number'].unique().tolist()))
#This person is responsible for the overall care of the inpatient. Thus, they might play a large role in how long that person stays
#so we will keep this column since there are 27,085 different attending providers

#Run these once you have compiled all of them!
hos_sql_df_1 = hos_sql_df_1.drop(['Zip Code - 3 digits'], axis=1) #use average income as a feature instead
hos_sql_df_1 = hos_sql_df_1.drop(['Facility Name'], axis=1)
hos_sql_df_1 = hos_sql_df_1.drop(['CCS Diagnosis Description'], axis=1)
hos_sql_df_1 = hos_sql_df_1.drop(['CCS Procedure Description'], axis=1)
hos_sql_df_1 = hos_sql_df_1.drop(['APR DRG Description'], axis=1)
hos_sql_df_1 = hos_sql_df_1.drop(['APR MDC Description'], axis=1)
hos_sql_df_1 = hos_sql_df_1.drop(['Discharge Year'], axis=1)    #since these are all 2015 since the dataset is from 2015 inpatient records
hos_sql_df_1 = hos_sql_df_1.drop(['Operating Certificate Number'], axis=1)  #drop this column since it should not be a predictor for inpatient length of stay
hos_sql_df_1 = hos_sql_df_1.drop(['Ethnicity'], axis=1) #contains less information than and is contained within the Race column, so let's drop
hos_sql_df_1 = hos_sql_df_1.drop(['Hospital County'], axis=1) #lets drop hospital county column for computational efficiency
hos_sql_df_1 = hos_sql_df_1.drop(['Birth Weight'], axis=1) #Now we will drop birth weight, which we looked at up above
hos_sql_df_1 = hos_sql_df_1.drop(['Attending Provider License Number'], axis=1) #to enable generalization of model to any attending providers
hos_sql_df_1 = hos_sql_df_1.drop(['Patient Disposition'], axis=1) #data leakage feature
hos_sql_df_1.dtypes
mort_string_index = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
age_string_index = {'0 to 17': 1, '18 to 29': 2, '30 to 49': 3, '50 to 69': 4, '70 or Older': 5}

hos_sql_df_1['Age Group'] = hos_sql_df_1['Age Group'].apply(lambda x: age_string_index[x])
hos_sql_df_1['APR Risk of Mortality'] = hos_sql_df_1['APR Risk of Mortality'].apply(lambda x: mort_string_index[x])
display(hos_sql_df_1)
encoded_df = pd.get_dummies(hos_sql_df_1)
display(encoded_df)
encoded_df['Facility Id'] = encoded_df['Facility Id'].astype('category')
encoded_df['CCS Diagnosis Code'] = encoded_df['CCS Diagnosis Code'].astype('category')
encoded_df['CCS Procedure Code'] = encoded_df['CCS Procedure Code'].astype('category')
encoded_df['APR DRG Code'] = encoded_df['APR DRG Code'].astype('category')
encoded_df['APR MDC Code'] = encoded_df['APR MDC Code'].astype('category')
encoded_df['APR Severity of Illness Code'] = encoded_df['APR Severity of Illness Code'].astype('category')
#encoded_df['Attending Provider License Number'] = encoded_df['Attending Provider License Number'].astype('category')
f, ax = plt.subplots(figsize=(11, 9))
corr = encoded_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
X = encoded_df.drop(['Length of Stay','Total Charges','Total Costs'], axis=1) #remove data leakage features
y = encoded_df[['Length of Stay']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_pred = reg.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
y_round_pred = np.ndarray.round(y_pred)
y_round_train_pred = np.ndarray.round(y_train_pred)
test_acc = accuracy_score(y_test, y_round_pred)
train_acc = accuracy_score(y_train, y_round_train_pred)
print(mse_test)
print(mse_train)
print('Test accuracy:', test_acc)
print('Train accuracy:', train_acc)
x_train = StandardScaler().fit_transform(X_train)
x_test = StandardScaler().fit_transform(X_test)

pca1 = PCA()
pca1.fit(x_train)
explained_variance_ratio = pca1.explained_variance_ratio_
pc_vs_variance = np.cumsum(pca1.explained_variance_ratio_)
plt.plot(pc_vs_variance)
plt.xlabel('Number of Components')
plt.ylabel('% Explained Variance')
plt.title('PCA Explained Variance vs. Number of Components')
plt.show()
variance_95 = list(filter(lambda i: i > 0.95, pc_vs_variance))[0]
component_95 = pc_vs_variance.tolist().index(variance_95)
print(component_95)

#perform the dimensionality reduction
pca2 = PCA(n_components=component_95)
x_train = pca2.fit_transform(x_train)
x_test = pca2.transform(x_test)
pc_df = pd.DataFrame(pca2.components_,columns=X_train.columns)
top_pc = pc_df[0:1]
display(top_pc)
new_top_pc = top_pc.abs()
features = []
for i in range(0,10):
  features.append(new_top_pc.idxmax(axis=1).tolist())
  new_top_pc = new_top_pc.drop(columns=features[i])
print(features)
top_pc = top_pc.abs()
list_magnitude = top_pc.loc[0, :].values.tolist()
labels = top_pc.columns.tolist()
feature_importance_df = pd.DataFrame({'Feature': labels, 'Relative Importance': list_magnitude})
#feature_importance_df = feature_importance_df.nlargest(10, 'Relative Importance')

fig, ax =plt.subplots(figsize=(18,15))
sns.barplot(x='Feature', y='Relative Importance', data=feature_importance_df.reset_index())
plt.xticks(rotation=90)
plt.show()
#Decision Tree - on non-pca data
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#from sklearn import metrics
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 10, max_leaf_nodes=150)
dtree.fit(x_train,y_train)

#from sklearn import metrics
train_predictions = dtree.predict(x_train)
test_predictions = dtree.predict(x_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#from sklearn.tree import plot_tree
f, ax = plt.subplots(figsize=(50, 30))
plot_tree(dtree)
plt.show()
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score

clf=DecisionTreeRegressor(max_depth=10, max_leaf_nodes=150)
clf.fit(x_train,y_train)
train_pred = clf.predict(x_train)
test_pred = clf.predict(x_test)

mse_test = mean_squared_error(y_test, test_pred)
mse_train = mean_squared_error(y_train, train_pred)
y_round_pred = np.ndarray.round(test_pred)
y_round_train_pred = np.ndarray.round(train_pred)
test_acc = accuracy_score(y_test, y_round_pred)
train_acc = accuracy_score(y_train, y_round_train_pred)
print(mse_test)
print(mse_train)
print('Test accuracy:', test_acc)
print('Train accuracy:', train_acc)
bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120]
encoded_df['stay_bin']=pd.cut(x = encoded_df['Length of Stay'],
                        bins = bins)
encoded_df['stay_label']=pd.cut(x = encoded_df['Length of Stay'],
                        bins = bins,
                        labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
encoded_df['stay_bin'] = encoded_df['stay_bin'].apply(lambda x: str(x).replace(',',' -'))
encoded_df['stay_bin'] = encoded_df['stay_bin'].apply(lambda x: str(x).replace('120','120+')) #make this bin more descriptive
display(encoded_df)
f, ax = plt.subplots(figsize=(20, 15))
sns.countplot(x='stay_bin', data = encoded_df)
plt.xticks(rotation=90)
plt.title('Class Distribution')
plt.show()
#from sklearn.model_selection import train_test_split

#create train and test sets
new_X = encoded_df.drop(['Length of Stay','Total Charges','Total Costs','stay_bin','stay_label'], axis=1)
new_y = encoded_df[['stay_label']]
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.3)

#perform pca
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
x_train = StandardScaler().fit_transform(X_train)
x_test = StandardScaler().fit_transform(X_test)

pca = PCA(n_components=29) #50 components, as found above
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#Create confusion matrix
#from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(dtree, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix with Normalization')
plt.show()
#We can see that we have major class imbalance issues. I deal with that here
#from collections import Counter 
counts = y_train['stay_label'].value_counts().tolist()
print(counts)
#df_class_0_under = df_class_0.sample(count_class_1)
#df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)


#from imblearn.under_sampling import RandomUnderSampler

resample_dict = {0:112307, 1:112307, 2:112307, 3:112307} #resample the first four classes to have the same number of instances as the (10-15] bucket
rus = RandomUnderSampler(random_state=0, sampling_strategy=resample_dict)
x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
unique_elements, counts_elements = np.unique(y_resampled, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
#Plot the new class distribution in the train set

#y_resampled_df = pd.DataFrame(data=y_resampled, columns=['label'])
y_resampled_df = pd.DataFrame(data=y_resampled, columns=['stay_label'])

f, ax = plt.subplots(figsize=(20, 15))
sns.countplot(x='stay_label', data = y_resampled_df)
plt.title('Class Distribution')
plt.show()
#Now lets run the decision tree and confusion matrix again
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 10, max_leaf_nodes=300)
dtree.fit(x_resampled,y_resampled)


#from sklearn import metrics
train_predictions = dtree.predict(x_train)
test_predictions = dtree.predict(x_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#Create confusion matrix
#from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(dtree, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix with Normalization')
plt.show()

#lets downsample all the way to one of our smallest bins (the last bin that contains lenght of stays 100-120+)
#from collections import Counter 
counts = y_train['stay_label'].value_counts().tolist()
print(counts)

#from imblearn.under_sampling import RandomUnderSampler

resample_dict = {0:1000, 1:1000, 2:1000, 3:1000, 4:1000, 5:1000, 6:1000, 7:1000, 8:1000, 9:1000, 10:1000, 11:1000, 12:1000, 13:1000,
                 14:1000, 15:1000, 16:1000, 17:1000, 18:1000, 19:1000, 20:1000, 23:1000 } #lets take everything down to the 14th bucket size = 6785
rus = RandomUnderSampler(random_state=0, sampling_strategy=resample_dict)
x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
unique_elements, counts_elements = np.unique(y_resampled, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
#Now lets run the decision tree and confusion matrix again
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 30, max_leaf_nodes=500)
dtree.fit(x_resampled,y_resampled)


#from sklearn import metrics
train_predictions = dtree.predict(x_train)
test_predictions = dtree.predict(x_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(dtree, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix with Normalization')
plt.show()
#perform decision tree classification no undersampling with non-pca data
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 10, max_leaf_nodes=300)
dtree.fit(X_train,y_train)

#from sklearn import metrics
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#perform decision tree classification with balanced class weight parameter
#from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 15, max_leaf_nodes=300, class_weight='balanced')
dtree.fit(x_train,y_train)

#from sklearn import metrics
train_predictions = dtree.predict(x_train)
test_predictions = dtree.predict(x_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
#confusion matrix for unbalanced classes with class_weights balanced
#from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(dtree, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix for Balanced Class Weights no Undersampling with Normalization')
plt.show()
#bins = [0,3,6,9,13,20,50,120]
#labels = [3,6,9,13,20,50,120]

bins = [0,5,10,20,30,50,120]
labels = [5,10,20,30,50,120]
encoded_df['stay_bin']=pd.cut(x = encoded_df['Length of Stay'], #encoded df is the raw dataframe following one-hot encoding
                        bins = bins)
encoded_df['stay_label']=pd.cut(x = encoded_df['Length of Stay'],
                        bins = bins,
                        labels = labels) #lets also rename our bins to be more descriptive since now they are much larger
encoded_df['stay_bin'] = encoded_df['stay_bin'].apply(lambda x: str(x).replace(',',' -'))
encoded_df['stay_bin'] = encoded_df['stay_bin'].apply(lambda x: str(x).replace('120','120+')) #make this bin more descriptive
display(encoded_df)
f, ax = plt.subplots(figsize=(15, 11))
sns.countplot(x='stay_bin', data = encoded_df, palette='Reds')
plt.xticks(rotation=90)
plt.title('Class Distribution')
plt.xlabel('Length of Stay Bins')
plt.ylabel('Patient Count (millions)')
plt.show()
from sklearn.model_selection import train_test_split

#create train and test sets
new_X = encoded_df.drop(['Length of Stay','Total Charges','Total Costs','stay_bin','stay_label'], axis=1)
new_y = encoded_df[['stay_label']]
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.3)

#perform pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
x_train = StandardScaler().fit_transform(X_train)
x_test = StandardScaler().fit_transform(X_test)

pca = PCA(n_components=29) #29 components, as found above
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#logisitic regression without class balance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
log_reg = LogisticRegression(multi_class='ovr').fit(x_train, y_train)
y_train_pred = log_reg.predict(x_train)
y_pred = log_reg.predict(x_test)

test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, y_train_pred)

print('Test accuracy:', test_acc)
print('Train accuracy:', train_acc)
#Create confusion matrix for no class balancing
from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(11, 9))
plot_confusion_matrix(log_reg, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix Without Class Balancing')
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#logistic regression - works better with standardized but no PCA performed on data
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#import numpy as np
log_reg = LogisticRegression(class_weight='balanced', multi_class='ovr').fit(x_train, y_train)
y_train_pred = log_reg.predict(x_train)
y_pred = log_reg.predict(x_test)

test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, y_train_pred)

print('Test accuracy:', test_acc)
print('Train accuracy:', train_acc)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#Another example of importance of class balancing
#Create confusion matrix
from sklearn.metrics import plot_confusion_matrix
f, ax = plt.subplots(figsize=(11, 9))
plot_confusion_matrix(log_reg, x_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title('Confusion Matrix With Class Balancing')
plt.show()
#overfit tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(class_weight='balanced')
dtree.fit(X_train,y_train)

from sklearn import metrics
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
from sklearn.model_selection import validation_curve
param_range = [15,20,30,35,100]
train_scores, test_scores = validation_curve(dtree, x_train, y_train, param_name='max_depth', 
                                             param_range=param_range, cv=3, scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplots(figsize=(11,9))
plt.title("Validation Curve with Decision Tree Classification")
plt.xlabel('Max Depth')
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
from sklearn.model_selection import validation_curve
dtree=DecisionTreeClassifier(class_weight='balanced')
dtree.fit(X_train,y_train)
param_range = [15,20,30,35,100]
train_scores, test_scores = validation_curve(dtree, X_train, y_train, param_name='max_depth', 
                                             param_range=param_range, cv=3, scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.subplots(figsize=(11,9))
plt.title("Validation Curve with Decision Tree Classification")
plt.xlabel('Max Depth')
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
#try a randomized search on decision tree with 3-fold cross validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

dtree = DecisionTreeClassifier(class_weight='balanced')
search_vals = dict(max_depth=[35,50,75,100], max_leaf_nodes=[800,1000,1500,2000])
dtree_search = RandomizedSearchCV(dtree, search_vals, cv=3)
search = dtree_search.fit(X_train,y_train)
search.best_params_
#decision tree optimal parameters
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth= 50, max_leaf_nodes=1000, class_weight='balanced')
dtree.fit(X_train,y_train)

from sklearn import metrics
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test, test_predictions))
#randomized search on random forest with 3-fold CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(class_weight='balanced')
search_vals = dict(max_depth=[15,25,50], max_leaf_nodes=[600,800,1400], n_estimators=[100,300,500])
dtree_search = RandomizedSearchCV(rf, search_vals, cv=3)
search = dtree_search.fit(X_train,y_train)
search.best_params_
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced')
rf.fit(X_train,y_train)

train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)
print("Train Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Test Accuracy:",metrics.accuracy_score(y_test, test_predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test, test_predictions))
feat_importances = rf.feature_importances_
feat_names = X_train.columns.tolist()

plt.subplots(figsize=(18,11))
plt.xticks(rotation=90)
plt.bar(x=feat_names, height=feat_importances)
plt.title('Importance of Input Features on Length of Stay Predictor in Random Forest Model')
plt.ylabel('Feature Importance')
plt.show()
#Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtree = DecisionTreeClassifier(random_state = 1, class_weight = "balanced", max_depth = 15)
boost = AdaBoostClassifier(dtree, n_estimators=75, random_state=0)
boost.fit(X_train, y_train)

train_predictions = boost.predict(X_train)
test_predictions = boost.predict(X_test)
print("Train Accuracy:", accuracy_score(y_train, train_predictions))
print("Test Accuracy:", accuracy_score(y_test, test_predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test, test_predictions))
hos_df.info()
#we found out how many Type of Admission
print("Type of Admission in Dataset:\n")
print(hos_df['Type of Admission'].unique())
#we found out how many Age group
print("\n\nAge Group in Dataset:\n")
print(hos_df['Age Group'].unique())
#we found out how many ARP Risk of Mortality
print("\n\nARP Risk of Mortality:\n")
print(hos_df['APR Risk of Mortality'].unique())
#we found out how many hospital country in our data
print("\n\nHospital Country in Dataset:\n")
print("There are {} different values\n".format(len(hos_df['Hospital County'].unique())))
print(hos_df['Hospital County'].unique())
#we found out how many ARP MDC Description
print("\n\nARP MDC Description(disease diagnosis) in Dataset:\n")
print("There are {} different values\n".format(len(hos_df['APR MDC Description'].unique())))
print(hos_df['APR MDC Description'].unique())
#We group features by data numbers
#show it if missing value(dropna=False)
hos_df['Type of Admission'].value_counts(dropna=False)
#number of patients by age groups
#show it if missing value(dropna=False)
hos_df['Age Group'].value_counts(dropna=False)
#show it if missing value(dropna=False)
print("Patients with or without abortion:\n")
print(hos_df['Abortion Edit Indicator'].value_counts(dropna=False))
#filtering
hos_df_newborn=hos_df['Type of Admission']=='Newborn'
print("Total Newborns:",hos_df_newborn.count())
hos_df[hos_df_newborn].head()
#grouping of mortality risk values
#show it if missing value(dropna=False)
hos_df['APR Severity of Illness Description'].value_counts(dropna=False)
hos_df_new = hos_df.head()
hos_df_melted = pd.melt(frame = hos_df_new, id_vars = 'APR MDC Description', value_vars = ['Age Group','Type of Admission'])
hos_df_melted
#firstly lets create 2 data frame
hos_df_data1=hos_df['APR MDC Description'].tail()
hos_df_data2=hos_df['Age Group'].tail()

conc_hos_df_col=pd.concat([hos_df_data1,hos_df_data2],axis=1)
conc_hos_df_col
#data frames from dictionary
Hospital=list(hos_df['Hospital County'].head())
Facility=list(hos_df['Facility Name'].head())
Year=list(hos_df['Discharge Year'].head())
Costs=list(hos_df['Total Costs'].head())

list_label=["hospital_country","facility_name","discharge_year","total_costs"]
list_col=[Hospital,Facility,Year,Costs]
zipped=list(zip(list_label,list_col))
hos_df_dict=dict(zipped)

hos_df_diff=pd.DataFrame(hos_df_dict)
hos_df_diff



hos_df_data1=hos_df.loc[:,["Total Costs","Total Charges","Birth Weight","Length of Stay"]]
hos_df_data1.plot()
plt.show()
hos_df_data1.plot(subplots=True)
plt.show()
hos_df_data1.plot(kind="hist",y="Total Costs",bins=50,range=(0,250))
plt.show()
#with non cumulative an cumulative
fig,axes=plt.subplots(nrows=2,ncols=1)

hos_df_data1.plot(kind="hist",y="Total Costs",bins=50,range=(0,250),ax=axes[0])
hos_df_data1.plot(kind="hist",y="Total Costs",bins=50,range=(0,250),ax=axes[1],cumulative=True)

plt.savefig("Graph.png")
plt.show()
print(hos_df['Discharge Year'])
hos_df['Discharge Year'] =pd.to_datetime(hos_df['Discharge Year'])
#lets make discharge_year as index
hos_df_dis=hos_df.set_index("Discharge Year")
hos_df_dis
print(hos_df.loc[85,['APR DRG Description']])
#selecting only some columns
hos_df[["APR DRG Description","Age Group","Length of Stay"]].head(20)
print(hos_df.loc[1:10,"Race":"Length of Stay"])

hos_df.loc[1:10,"Gender":]
print("Total hospitalization times for patients admitted to the hospital as Urgent:",
      hos_df['Length of Stay'][hos_df['Type of Admission']=='Urgent'].sum())

#The first value of unique races of patients coming to the hospital
hos_df.groupby("Race").first()
print("Total hospitalization times for patients admitted to the hospital as Emergency:",
      hos_df['Length of Stay'][hos_df['Type of Admission']=='Emergency'].sum())

#The first value of unique races of patients coming to the hospital
hos_df.groupby("Race").first()