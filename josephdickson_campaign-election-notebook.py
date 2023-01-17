# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/electionfinance/CandidateSummaryAction1.csv")
data.head()
data.shape

#first, visualize missing values

import missingno as msn

msn.matrix(data)
#process data values



data['cov_sta_dat'] = pd.to_datetime(data['cov_sta_dat'])

data['cov_end_dat'] = pd.to_datetime(data['cov_end_dat'])



data['campaign_duration'] = (data['cov_end_dat'] - data['cov_sta_dat']).dt.days
#create a function to check all columns with missing data greater than 90% and drop them



def process_missing_data(data, threshold, inplace_value):

    #create a list to hold columns with missing value above threshold

    drop_cols = []

    

    #create a variable to store all columns in the dataframe

    all_cols = data.columns

    

    #calculate all columns with missing values percentage greater than the threshold

    missing_percentage = (data[all_cols].isna().sum()/len(data))*100

    

    #create a dataframe to store all candidate columns and their percentage

    missing_df = pd.DataFrame({"cols":all_cols, "percentage":missing_percentage})

    

    #check for threshold condition

    missing_filtered = missing_df[missing_df['percentage'] >= threshold] 

    drop_cols.append(missing_filtered["cols"].tolist())

    

    #drop candidate columns

    drop_cols = drop_cols[0]

    data.drop(columns=drop_cols, inplace= inplace_value)

    

    return data.shape

#call function on data

process_missing_data(data=data, threshold=90, inplace_value=True)
data['can_off'].value_counts(normalize=True, sort=True) * 100
#first, convert the net_con column to a float data type and modify data inplace

def converter(data, data_col):

    value = data[data_col].str.replace('$','').str.replace(',','').str.replace('(','-').str.replace(')','').astype('float32')

    data[data_col] = value

    return data.head()



#call the function on the net_con feature

converter(data=data, data_col="net_ope_exp")
data['winner'] = data['winner'].fillna('N')
H_df = data.loc[data['can_off'] == "H"] 

S_df = data.loc[data['can_off'] == "S"]

P_df = data.loc[data['can_off'] == "P"]
#check the shape of the data



print(f'The shape of the House of assembly data is {H_df.shape}')

print(f'The shape of the senate data is {S_df.shape}')

print(f'The shape of the presidential data is {P_df.shape}')
Amt_per_sta_ds = H_df.groupby(['can_off_sta', 'can_off_dis'])['net_ope_exp'].sum().to_frame(name = "total_dis_sum").reset_index()
Amt_per_sta_ds.head()
#visualize the state with high spending



plt.figure(figsize=(20,10))



ax = sns.barplot(x="can_off_sta", y="total_dis_sum", data=Amt_per_sta_ds)
mt_comp = H_df.loc[H_df['can_off_sta'] == 'MT']

mt_comp
ax = sns.barplot(x='can_nam', y='net_ope_exp', hue = 'winner',data=mt_comp)
mt_comp
ax = sns.barplot(x='can_nam', y='campaign_duration',hue='winner', data=mt_comp)
competitors = H_df.groupby(['can_off_sta', 'can_off_dis'])['can_id'].count().to_frame(name = "num_of_comp").reset_index()

#eliminate data points where num_of_comp <= 1

#this means that these positions are unopposed

competitors = competitors[competitors['num_of_comp'] > 1]
competitors.head()
plt.figure(figsize=(20,10))



ax = sns.barplot(x="can_off_dis", y="num_of_comp", data=competitors)
al_comp = H_df.loc[H_df['can_off_sta'] == 'AL']

al_comp.shape
plt.figure(figsize=(20,10))

ax = sns.barplot(x='can_id', y='net_ope_exp', hue = 'winner',data=al_comp)
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="white", color_codes=True)

sns.jointplot(x=H_df["net_ope_exp"], y=H_df["votes"], kind='kde', color="skyblue")
S_df.head()
prf1 = S_df['votes'].isna().count()

prf2 =  len(S_df['votes'])



print(prf1)

print(prf2)
S_df.drop(columns='votes', inplace=True)
S_df.head()
#first, we check amount spent per state 

Amt_per_sta_ds = S_df.groupby(['can_off_sta', 'can_off_dis'])['net_ope_exp'].sum().to_frame(name = "total_dis_sum").reset_index()
Amt_per_sta_ds.head()
#visualize to see the highest spending state



plt.Figure(figsize=(20,10))

ax = sns.barplot(x='can_off_sta', y='total_dis_sum', data=Amt_per_sta_ds)
#create the three dataframes

fl_comp = S_df.loc[S_df['can_off_sta'] == 'FL']

pa_comp = S_df.loc[S_df['can_off_sta'] == 'PA']

nv_comp = S_df.loc[S_df['can_off_sta'] == 'NV']
#check winners in FL



ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='winner', data=fl_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)
fl_comp.head()
ax = sns.barplot(x='can_nam', y='campaign_duration', hue='winner', data=fl_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)
ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='can_par_aff', data=fl_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)
fl_comp['can_par_aff'].value_counts(normalize=True, sort=True).plot()
#check winners in PA



ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='winner', data=pa_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)


plt.style.use('seaborn-white')

plt.subplot(121)

ax = sns.barplot(x='can_nam', y='campaign_duration', hue='winner', data=pa_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")





plt.subplot(122)

pa_comp['can_par_aff'].value_counts(normalize=True, sort=True).plot()

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("dominant party")







ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='can_par_aff', data=pa_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")

#check winners in PA



ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='winner', data=nv_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)
plt.style.use('seaborn-white')

plt.subplot(121)

ax = sns.barplot(x='can_nam', y='campaign_duration', hue='winner', data=nv_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")





plt.subplot(122)

nv_comp['can_par_aff'].value_counts(normalize=True, sort=True).plot()

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("dominant party")

ax = sns.barplot(x='can_nam', y='net_ope_exp', hue='can_par_aff', data=nv_comp)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")
P_df.head()
P_df.shape
#create dataframe grouped by total amount spent

Amt_per_sta_ds = P_df.groupby(['can_nam', 'winner', 'can_par_aff', 'campaign_duration'])['net_ope_exp'].sum().to_frame(name = "total_dis_sum")
#sort result

Amt_per_sta_ds = Amt_per_sta_ds.sort_values(by = ['total_dis_sum'], ascending=False).reset_index()
#create visualization to reach an assumption on which section of the data points could be candidates to win the election

Amt_per_sta_ds['total_dis_sum'].plot()

# select first 20 data points as candidates

Amt_per_sta_ds = Amt_per_sta_ds.iloc[:20, :]
Amt_per_sta_ds
#check winner

ax = sns.barplot(x='can_nam', y = 'total_dis_sum', hue='winner', data=Amt_per_sta_ds)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.style.use('seaborn-white')

plt.subplot(121)

ax = sns.barplot(x='can_nam', y='campaign_duration', hue='winner', data=Amt_per_sta_ds)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")





plt.subplot(122)

Amt_per_sta_ds['can_par_aff'].value_counts(normalize=True, sort=True).plot()

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("dominant party")
ax = sns.barplot(x='can_nam', y='total_dis_sum', hue='can_par_aff', data=Amt_per_sta_ds)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',  

)

plt.title("campaign duration")
#create regression data

Regression_data = data[['can_off', 'can_off_sta', 'can_off_dis', 'can_inc_cha_ope_sea', 'net_ope_exp', 'can_par_aff','campaign_duration','votes']]





#create classification data

Classification_data = data[['can_off', 'can_off_sta', 'can_off_dis', 'can_inc_cha_ope_sea', 'net_ope_exp', 'can_par_aff','campaign_duration','winner']]

Regression_data.isna().sum()/len(Regression_data)
Classification_data.isna().sum()
from sklearn.impute import SimpleImputer



imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



Classification_data['can_off_dis'] = imp_mode.fit_transform(Classification_data[['can_off_dis']]).copy()

Classification_data['can_inc_cha_ope_sea'] = imp_mode.fit_transform(Classification_data[['can_inc_cha_ope_sea']]).copy()

Classification_data['net_ope_exp'] = Classification_data['net_ope_exp'].fillna(-99999999999999999999999).copy()

Classification_data['can_par_aff'] = imp_mode.fit_transform(Classification_data[['can_par_aff']]).copy()
Classification_data.isna().sum()
Classification_data = Classification_data[Classification_data.can_par_aff != 'PPT']
#make respective dataframes

H_model_data_cla = Classification_data.loc[Classification_data['can_off'] == 'H']

P_model_data_cla = Classification_data.loc[Classification_data['can_off'] == 'P']

S_model_data_cla = Classification_data.loc[Classification_data['can_off'] == 'S']
from sklearn.model_selection import train_test_split



X = H_model_data_cla.iloc[:,:-1]

y = H_model_data_cla.iloc[:,-1]
# determine categorical and numerical features







numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns

categorical_ix = X.select_dtypes(include=['object', 'bool']).columns





# define the data preparation for the columns

t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]

col_transform = ColumnTransformer(transformers=t)
# define base the model



model = SVC(kernel='rbf',gamma='scale',C=100)

# define the data preparation and modeling pipeline

pipeline = Pipeline(steps=[('prep',col_transform), ('m', model) ])
X.isna().sum()
#divide data into train and test split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))

print(f'accuracy of the base model on house of rep election is {accuracy_score(y_val, y_pred) * 100}%')