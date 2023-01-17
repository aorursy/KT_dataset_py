#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv', encoding='ISO-8859-2')

df.head()
na_percent = (df.isnull().sum()/len(df))[(df.isnull().sum()/len(df))>0].sort_values(ascending=False)



missing_data = pd.DataFrame({'Missing Percentage':na_percent*100})

missing_data
na = (df.isnull().sum() / len(df)) * 100

na = na.drop(na[na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12,8))

sns.barplot(x=na.index, y=na)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.title('Percentage Missing', fontsize=15)
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]

print(categorical_cols)

print(numerical_cols)
# filling missing values with NA

df[['Role Category', 'Key Skills', 'Role', 'Location', 'Job Title', 'Industry', 'Functional Area', 'Job Experience Required', 'Job Salary']] = df[['Role Category', 'Key Skills', 'Role', 'Location', 'Job Title', 'Industry', 'Functional Area', 'Job Experience Required', 'Job Salary']].fillna('NA')
from scipy import stats
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
summary = resumetable(df)

summary
total = len(df)

plt.figure(figsize=(12,6))



g = sns.countplot(x='Role Category', data=df, color='green')

g.set_title("Role Category DISTRIBUTION", fontsize = 20)

g.set_xlabel("Role Category Values", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

sizes=[] # Get highest values in y

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights



#plt.show()
nom_cols = ['Uniq Id', 'Crawl Timestamp', 'Job Title', 'Job Salary', 'Job Experience Required', 'Key Skills', 'Role Category', 'Location', 'Functional Area', 'Industry', 'Role']
def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):

    

    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart

    plt.figure(figsize=(17, 35)) # size of figure



    # loop to get column and the count of plots

    for n, col in enumerate(df[cols]): 

        tmp = pd.crosstab(df[col], df['Role Category'], normalize='index') * 100

        tmp = tmp.reset_index()

        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



        ax = plt.subplot(grid[n]) # feeding the figure of grid

        sns.countplot(x=col, data=df, order=list(tmp[col].values) , color='green') 

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Role Category', fontsize=18) # title label

        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



        # twinX - to build a second yaxis

        gt = ax.twinx()

        gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

        gt.set_ylabel("Role Category %True(1)", fontsize=16)

        sizes=[] # Get highest values in y

        for p in ax.patches: # loop to all objects

            height = p.get_height()

            sizes.append(height)

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center", fontsize=14) 

        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





    plt.subplots_adjust(hspace = 0.5, wspace=.3)

    plt.show()
#ploting_cat_fet(df, nom_cols, vis_row=5, vis_col=2)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Location)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTOcGUMagvBVr1Zvm9bjekYwJaSaLcimmiFmQ&usqp=CAU',width=400,height=400)