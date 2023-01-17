# Import initial modules
import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings

import matplotlib.pyplot as plt
%matplotlib inline

warnings.filterwarnings('ignore')
import os
file_path = '../input/'

# # # Running the notebook locally:
# # Change Directory
# os.chdir('/Users/paula/Dropbox/15. Kaggle/DonorsChoose/Data')
# print(os.getcwd()) 
# file_path = ''

# Load the data into multiple dataframes
train_full = pd.read_csv(file_path + 'train.csv')
test = pd.read_csv(file_path + 'test.csv')
resources = pd.read_csv(file_path + 'resources.csv')
submission = pd.read_csv(file_path + 'sample_submission.csv')
# Reduce the dataframe for fast compute time (10% of the entire df)
sample_size = round(len(train_full)*.10)
train = train_full.sample(sample_size, random_state = 1)
def clean_input_data(df, combined_col_name, text_column_list, 
                     character_remove_list, character_replace_list, replace_numbers):
    
    # (1) Combine all text data into one column, separated by spaces
    
    # Remove NaN's in columns
    df.replace(np.nan, '', regex=True, inplace = True)
    
    # Join all text columns into one
    df[combined_col_name] = ''
    for t in range(0, len(text_column_list)):
        df[combined_col_name] += df[text_column_list[t]] + ' '
    # Replace all numbers with a space 
    if replace_numbers == 1:
        df[combined_col_name] = df[combined_col_name].replace('\d+', ' ', regex=True)

    
    # (2) Remove special characters from the text data
    for i in range(0,len(character_remove_list)):
        df[combined_col_name] = df[combined_col_name].apply(
            lambda x: x.replace(character_remove_list[i], character_replace_list[i])
        )
resources.head()
# Reset index to the id
res = resources.set_index('id')
res['description'] = res['description'].astype(str)

# Join all the resource descriptions into one string, with a space
%time rs = res.groupby('id')['description'].apply(lambda x: ' '.join(x))

# Convert rs into a DataFrame
df_rs = (pd.DataFrame(rs)).reset_index()
df_rs.head()
df_rsp = pd.DataFrame(res.groupby('id')['price'].sum()).reset_index()
df_rsp.head()
# Join the aggregated resources definitions to the main df
df = pd.merge(train, df_rs, on='id',  how='left')

# Join the aggregated resources definitions to the main df
df = pd.merge(df, df_rsp, on='id',  how='left')
combined_col_name = 'full_essay'
text_column_list = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
character_remove_list = ['\\r', '\\n', '\\']
character_replace_list = [' ', ' ', ' ']
replace_numbers = 1
# Apply the function to create the cleaned DF and additional columns
clean_input_data(df, combined_col_name, text_column_list, 
                 character_remove_list, character_replace_list, replace_numbers)

# Check the first entry in the dataframe
df.head(1)
round(df['project_is_approved'].sum()*100/len(df['project_is_approved']),2)
ix_list = df.index[df['project_is_approved']==0].tolist()
# Set the index of the proposal to explore
ix_num = 2341

df.loc[[ix_list[ix_num]]]
df[df['project_is_approved']==0]['full_essay'][ix_list[ix_num]]
df[df['project_is_approved']==0]['project_resource_summary'][ix_list[ix_num]]
df[df['project_is_approved']==0]['description'][ix_list[ix_num]]
def describe_by_approval(col_name):
    R = pd.DataFrame(df[df['project_is_approved'] == 0][col_name].describe())
    R[col_name] = round(R[col_name],1)
    R.rename(columns ={col_name: ('Rejected - ' + col_name) }, inplace = True) 

    A = pd.DataFrame(df[df['project_is_approved'] == 1][col_name].describe())
    A[col_name] = round(A[col_name],1)
    A.rename(columns ={col_name: ('Accepted - ' + col_name) }, inplace = True) 
    RA = pd.concat([R, A], axis =1, join = "inner")
    display(RA)
describe_by_approval('price')
# Violin Chart Inputs
y1 = 'price'
y2 = 'price'
data1 = df
data2 = df[df['price'] < 1000]
T1 = 'All Proposal Prices'
T2 = 'Proposals less than $1,000'

# Set Dimension and Color
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
color_set = ['sandybrown','cornflowerblue']

# Graph 1
sns.violinplot(x='project_is_approved', y=y1, data=data1, cut = 0, palette=color_set, ax=axes[0])
# Graph 2
sns.violinplot(x='project_is_approved', y=y2, data=data2, cut = 0, palette=color_set, ax=axes[1])

axes[0].set_title(T1)
axes[1].set_title(T2)

plt.show()
text = df.full_essay

# Count the token only once from each document (proxy for vocab variety)
from sklearn.feature_extraction.text import CountVectorizer
# Appearance of unique tokens
vect = CountVectorizer(binary=True)
dtm_token_unique = vect.fit_transform(text)

# Count all appearances of tokens
vect = CountVectorizer()
dtm_token_all = vect.fit_transform(text)
# Count the total number of tokens
tu = pd.DataFrame(dtm_token_unique.sum(axis=1))
tu.columns = ['count_unique_tokens']
df = pd.concat([df, tu], axis = 1, join = "inner")


ta = pd.DataFrame(dtm_token_all.sum(axis=1))
ta.columns = ['count_all_tokens']
df = pd.concat([df, ta], axis = 1, join = "inner")
describe_by_approval('count_unique_tokens')
describe_by_approval('count_all_tokens')
# Violin Chart Inputs
y1 = 'count_unique_tokens'
y2 = 'count_all_tokens'
data1 = df
data2 = df
T1 = 'Unique Tokens Per Proposal'
T2 = 'Total Tokens per Proposal'

# Set Dimension and Color
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
color_set = ['sandybrown','cornflowerblue']

# Graph 1
sns.violinplot(x='project_is_approved', y=y1, data=data1, cut = 0, palette=color_set, ax=axes[0])
# Graph 2
sns.violinplot(x='project_is_approved', y=y2, data=data2, cut = 0, palette=color_set, ax=axes[1])

axes[0].set_title(T1)
axes[1].set_title(T2)

plt.show()
# Clean and combine the essay columns
text_column_list = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']

clean_input_data(train_full, combined_col_name, text_column_list, 
                 character_remove_list, character_replace_list, replace_numbers)
# Accepted Proposals 10% Sample Size
accept_ss = round(len(train_full[train_full['project_is_approved'] == 1])*.10)

# What proportion of the rejected df would I need to equal the same amount from the accepted?
# Create equal sized dataframes
ac_df = train_full[train_full['project_is_approved'] == 1].sample(accept_ss, random_state = 1)
re_df = train_full[train_full['project_is_approved'] == 0].sample(accept_ss, random_state = 1)

train2 = pd.concat([ac_df, re_df])
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

def vectorize_and_get_model_params(data, text_column, output_column):
    vect = CountVectorizer(binary=True, stop_words = 'english', analyzer="word")
    X_dtm = vect.fit_transform(data[text_column])
    print('features:', X_dtm.shape[1])    
    model.fit(X_dtm, data[output_column])
    token_list = vect.get_feature_names()
# curious about the variety of vocab?
vectorize_and_get_model_params(ac_df, 'full_essay', 'project_is_approved')
vectorize_and_get_model_params(re_df, 'full_essay', 'project_is_approved')
data = train2
text_column = 'full_essay'
output_column = 'project_is_approved'

vect = CountVectorizer(binary=True, stop_words = 'english', analyzer="word")
X_dtm = vect.fit_transform(data[text_column])
print('features:', X_dtm.shape[1])    
model.fit(X_dtm, data[output_column])
token_list = vect.get_feature_names()
# number of times each token appears across all feature flag = 0 (in this case Rejected Project)
classA_token_count = model.feature_count_[0, :]
# number of times each token appears across all ClassB  (Accepted Project)
classB_token_count = model.feature_count_[1, :]
# create a DataFrame of tokens with their separate counts
tokens = pd.DataFrame({'token': token_list, 
                       'classA: Rejected':classA_token_count, 
                       'classB: Accepted':classB_token_count}).set_index('token')
# Extract the index column into its own column
tokens = tokens.reset_index()
classA_name = 'classA_norm'
classA_count = 'classA: Rejected'

classB_name = 'classB_norm'
classB_count = 'classB: Accepted'
# Convert the Rejected and Accepted Class counts into normalized frequencies
tokens[classA_name] = tokens[classA_count]  / model.class_count_[0]
tokens[classB_name] = tokens[classB_count]  / model.class_count_[1]

# Add 1 to Rejected and Accepted Class counts to avoid dividing by 0
A_plus = (tokens[classA_count] + 1)  / model.class_count_[0]
B_plus = (tokens[classB_count] + 1 )  / model.class_count_[1]

# Calculate the ratio of Accepted-to-Rejected for each token
tokens['classB_ratio'] = tokens[classB_name] / A_plus
tokens['classB_ratio'][(tokens[classA_name])>0] = tokens[classB_name] / tokens[classA_name]

tokens['classA_ratio_extra'] = tokens[classA_name] / B_plus
tokens['classA_ratio_extra'][(tokens[classB_name])>0] = tokens[classA_name] / tokens[classB_name]
## Get proportions (true, without adding 1)
tokens['classA_proportion'] = 0
total_projects_classA = model.class_count_[0]
tokens['classA_proportion'][(tokens[classA_count])>0] = (tokens[classA_count]) / total_projects_classA

tokens['classB_proportion'] = 0
total_projects_classB = model.class_count_[1]
tokens['classB_proportion'][(tokens[classB_count])>0] = (tokens[classB_count]) / total_projects_classB

# Subtract the proportions
tokens['proportion_diff'] = tokens['classB_proportion'] - tokens['classA_proportion']
sns.set(rc={'figure.figsize':(14,8)})

sns.kdeplot(tokens['classB_proportion'], color='cornflowerblue')
sns.kdeplot(tokens['classA_proportion'], color='sandybrown')

plt.show()
# What tokens appear more than 25% of both the corpuses and are a min length of 2?
type_best = tokens[((tokens['classB_proportion'] >= 0.25) | (tokens['classA_proportion'] >= 0.25)  ) 
                   & (tokens['token'].str.len() > 2)
      ].sort_values('classB_ratio', ascending=False)
from pylab import suptitle, yticks

main_var = 'classB_ratio'
secondary_var = 'classA_ratio_extra'
type_best = type_best.sort_values(main_var, ascending = True)

val1 = type_best[main_var].tolist()
x_max = max(type_best[main_var].max(), type_best[secondary_var].max())
val2 = type_best[secondary_var].tolist()

bars = type_best['token'].tolist()
pos = np.arange(len(val1))


fig, axes = plt.subplots(ncols=2, sharey=True)
fig.set_size_inches(7, 11)
axes[0].barh(pos,val1, align='center', color='cornflowerblue', label = 'Accepted')
axes[1].barh(pos,val2, align='center', color='sandybrown', label = 'Rejected')

yticks(pos, bars, fontsize = 14)

axes[0].yaxis.set_tick_params(labelsize=14)
axes[0].xaxis.set_tick_params(labelsize=14)
axes[1].xaxis.set_tick_params(labelsize=14)


axes[0].set_xlim(0, x_max+.1*x_max)
axes[0].invert_xaxis()
axes[1].set_xlim(0, x_max+.1*x_max)


title_string = "Ratio of Tokens between Corpus Types"
suptitle(title_string, fontsize=18, fontweight='bold')

axes[0].legend(bbox_to_anchor=(0.1, 1.0, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
axes[1].legend(bbox_to_anchor=(0.1, 1.0, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
type_best = tokens[(tokens['classB_proportion'] > 0.005) & (tokens['token'].str.len() > 2)
      ].sort_values('classB_ratio', ascending=False).head(50)

type_best[['token', 'classA_norm', 'classB_norm', 'classB_ratio']].head(20)
type_best = tokens[(tokens['classA_proportion'] > 0.005) & (tokens['token'].str.len() > 2)
      ].sort_values('classA_ratio_extra', ascending=False).head(50)

type_best[['token', 'classA_norm', 'classB_norm', 'classA_ratio_extra']].head(20)