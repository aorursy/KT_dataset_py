import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy  import stats
import glob
from tqdm import tqdm
from gensim.models import Word2Vec 
import random
import umap
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE  # final reduction
import ipywidgets as widgets

import sys
from tabulate import tabulate
import os

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as py
import plotly.express as px

init_notebook_mode(connected=True)

plt.style.use('seaborn-ticks')
# print(plt.style.available)



li = []
for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_excel(os.path.join(dirname, filename), index_col=None, header=0)
        li.append(df)


li
#  Training,Education, skills, and Experience
train_exp = pd.concat([li[0],li[1]], sort = True)
train_exp.head()
# Fix column Names
train_exp.columns = train_exp.columns.str.lower()
train_exp.columns = train_exp.columns.str.replace(' ', '_',)

# Rename column
train_exp.rename(columns={'o*net-soc_code':'job_id'} ,inplace= True)
# Drop Colunms
train_exp.drop(['lower_ci_bound', 'n', 'not_relevant','upper_ci_bound','recommend_suppress','standard_error'], axis= 1, inplace=True)
# Create new Column
train_exp['normalized'] = (train_exp['data_value'] - train_exp['data_value'].min()) / (train_exp['data_value'].max() - train_exp['data_value'].min())
train_exp.head()
#  Training, and Experience Categories
categories = li[2]
# Fix Coulmn names
categories.columns = categories.columns.str.lower()
categories.columns = categories.columns.str.replace(' ', '_',)
# dataframe Head
categories.head()
# subset from the original dataframe
mergedStuff = pd.merge(train_exp, categories, on=['category'], how='inner')
# train_exp.to_csv('training_experience_dataset.csv')
train_exp.shape
train_exp.describe()
train_exp.info()
print('Missing data [%]')
round(train_exp.isnull().sum() / len(train_exp) * 100, 4)
train_exp['job_id'].nunique()
train_exp['scale_id'].nunique()
train_exp['element_id'].nunique()
train_exp['title'].nunique()
g = sns.pairplot(train_exp)
skills = pd.DataFrame(train_exp.groupby('element_name')['data_value'].sum()).reset_index()
skills.sort_values(by = 'data_value',ascending=False,inplace = True )
vals_skl = [x for x in train_exp.element_name]
vals_job = [x for x in train_exp.title]
custom_stopword = [x for x in skills.element_name[skills.data_value >= skills.data_value.mean()]]
len(str(vals_skl)),len(str(vals_job)),len(str(custom_stopword))
for x in custom_stopword:
    STOPWORDS.add(x)
wordcloud = WordCloud(max_font_size=50, max_words=400000, 
                      background_color="#fff",stopwords=STOPWORDS, 
                      colormap=plt.cm.ocean).generate(str(set(vals_job)).lower(
))
plt.figure(figsize=(15,50))
plt.imshow(wordcloud, interpolation="bicubic")
plt.title('\n Common Job Titles \n ' ,fontsize=34)
plt.axis("off")
plt.show()
wordcloud = WordCloud(max_font_size=100, max_words=7007000, 
                      background_color="#fff",stopwords=STOPWORDS, 
                      colormap=plt.cm.ocean).generate(str(set(vals_skl)))
plt.figure(figsize=(15,50))
plt.imshow(wordcloud, interpolation="bicubic")
plt.title('\n Common Requirment  \n ' ,fontsize=34)
plt.axis("off")
plt.show()
plt.figure(figsize=(10,8))
sns.barplot(data = skills[skills.data_value <= skills.data_value.mean()][:30],
            x= 'data_value',y='element_name',palette="GnBu_d")
plt.title('\n Top 30 Skills \n ' ,fontsize=25);

plt.figure(figsize=(10,6))
sns.barplot(data = skills[skills.data_value >= skills.data_value.mean()],
            x= 'data_value',y='element_name',palette="GnBu_d")
plt.title('\n Training Methods And Experience Requirement\n ' ,fontsize=25);
merge_sub = mergedStuff[['category' ,'category_description' ,'data_value']]
grouped = pd.DataFrame(merge_sub.groupby(['category','category_description'])['category'].sum()).rename(columns={'category':'Value'}).reset_index()

fig = px.bar( grouped , x='category', y='Value',
             hover_data=['category', 'category_description', 'Value'],
             height=600,color='category',color_continuous_scale=px.colors.sequential.Teal,
             title=' Formal Education And Experience Requirement')
fig.show()
edu = mergedStuff['category'].values
labels = (np.array(mergedStuff.category_description))
values = (np.array((edu / edu.sum())*100))
plt.figure(figsize=(10,10))
sns.barplot(x= values,y=edu,palette="GnBu_d",orient= 'h')
plt.title('\n Mean Value Of The Formal Education And Experience Requirement \n ' ,fontsize=25);

rating_job = pd.DataFrame(train_exp.groupby(['job_id','title','scale_id'])['data_value'].sum()).reset_index(['scale_id','title'])
rating_job.sort_values(by = 'data_value', inplace = True)
fig = px.scatter( rating_job , x='title', y='data_value',
             hover_data=['title'],size= 'data_value',  size_max=10,
             height=700,width= 1000,color='scale_id',color_discrete_sequence=px.colors.diverging.Portland,
             title=' Job Titles And Training Scale')
fig.show()
avg_rate  = rating_job.loc[rating_job['data_value'] > rating_job['data_value'].min()]
plt.figure(figsize=(10,8))
sns.barplot(y = avg_rate.scale_id,x = 'data_value',data=avg_rate ,palette="GnBu_d" )
plt.yticks(fontsize = 15)
plt.title('Data Value Scale \n' ,fontsize = 25);
code_job = pd.DataFrame(train_exp.groupby(['job_id','title'])['element_id'].count()).reset_index('title')

code_job.shape
# Check
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(train_exp.groupby('scale_id')['data_value'].sum(), ax =ax[1])
ax[1].set_title('Distribution of Job Requirments Scale') 
sns.distplot(train_exp.groupby('element_id')['data_value'].sum(), ax =ax[0] )
ax[0].set_title('Distribution of Job skills') 
plt.show()

#  job-code 
job_code = train_exp['element_id'].unique().tolist()
# shuffle job-code 
random.shuffle(job_code)

# extract 70% of job-code 
train = [job_code[i] for i in range(round(0.7*len(job_code)))]

# split data into train and validation set
train_df = train_exp[train_exp['element_id'].isin(train)]
validation_df = train_exp[~train_exp['element_id'].isin(train)]
len(train_df)

len(validation_df)
# list to capture history of the employees
train_li = []

for i in tqdm(train):
    temp = train_df[train_df["element_id"] == i]["job_id"].tolist()
    train_li.append(temp)
# Test_df

test_li = []

for i in tqdm(validation_df["element_id"].unique()):
    temp = validation_df[validation_df["element_id"] == i]["job_id"].tolist()
    test_li.append(temp)
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(train_li, progress_per=200)

model.train(train_li, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)
# save word2vec model
model.save("word2vec_2.model")
# # L2-normalized vectors.

model.init_sims(replace=True)

print(model)

# # extract all vectors
X = model[model.wv.vocab]

print(X.shape)
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)

def aggregate_vectors(test):
    test_vec = []
    for i in test:
        try:
            test_vec.append(model[i])
        except KeyError:
            continue
        
    return np.mean(test_vec, axis=0)
aggregate_vectors(test_li[0]).shape
df_select = train_exp[['title','job_id','scale_id','scale_name','element_name','data_value','category']]

def recommend (v , n=0):
    ms_2 = model.wv.similar_by_vector(v, topn= n+1)[:]
    df= pd.DataFrame()
    li =[]
    sim =[]
   
    for j in ms_2:
        entry = {}
        li.append(j[0])
        sim.append(j[1])
        entry['job_id'] = li
        entry['similarty'] = sim

        df = df.append(pd.DataFrame(entry), sort=True)
        df_me = pd.merge(df_select, df, on=['job_id'], how='inner' )
        df_me['evaluate'] = df_me['similarty'] *df_me['data_value']
        df_me.sort_values(by = 'similarty' , ascending = False , inplace =True)


        dataframe = pd.merge(df_me,categories[['scale_id','category_description','category']] , on = ['scale_id','category'], how = 'outer').sort_values(by = 'similarty', ascending = False)
        dataframe.category_description.fillna('Skills',inplace= True)
        
        dataframe.set_index('scale_id' , inplace =True)
    print(' Job Title : {} \n   Similarity : {} \n  Resultes From : {}, Similar Jobs \n'.format( str(df_me['title'].unique().tolist()[0]) ,
          df_me['similarty'].unique().tolist()[0] , n ))
    print(df_me['title'].unique().tolist())
    
    
    Data = dataframe[['element_name' ,'category_description','evaluate']].sort_values(by = 'evaluate', ascending= False).drop_duplicates()
    
    df_oj = Data[(Data.index == 'OJ') |(Data.index == 'PT') ].drop_duplicates('category_description')[:3]
    
    df_lv = Data[Data.index == 'LV'].drop_duplicates('element_name')[:10]
    
    df_im = Data[Data.index == 'IM'].drop_duplicates('element_name')[:10]
    
    df_rw = Data[Data.index == 'RW'].drop_duplicates('category_description')[:2]
    
    df_rl = Data[Data.index == 'RL'].drop_duplicates('category_description')[:2]


    print(tabulate(df_oj,  tablefmt='fancy_grid'))
    print(tabulate(df_lv,   tablefmt="fancy_grid"))
    print(tabulate(df_im,   tablefmt="fancy_grid"))
    print(tabulate(df_rw,   tablefmt="fancy_grid"))
    print(tabulate(df_rl,   tablefmt="fancy_grid"))
    
    return dataframe.drop_duplicates()
    

# Train Data
tr_acuracy = recommend(model['13-1199.01'],10)
test_li[0][1935:1937]
#  Test/ Validation Data 
ab_test = recommend(aggregate_vectors(test_li[0][1000:1002]),10)

def plot(freq=1., color='blue', lw=2, grid=True):
    
    x_2 = ab_test['similarty']
    y_2 = ab_test.job_id
    
    x_3 = tr_acuracy['similarty']
    y_3 = tr_acuracy.job_id
    
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    sns.barplot(x= x_2,y = y_2,lw=lw, color=color , ax =ax[1])
    ax[1].set_title('Similarity Between Validation Data') 
    ax[1].grid(grid)
    sns.barplot(x= x_3,y = y_3,lw=lw, color=color , ax =ax[0])
    ax[0].set_title('Similarity Between Trained Data') 
    ax[0].grid(grid)
    fig.autofmt_xdate(rotation=45)
plot()
# df_select.job_id[df_select['title'] == 'Energy Auditors'].unique().tolist()[0]
