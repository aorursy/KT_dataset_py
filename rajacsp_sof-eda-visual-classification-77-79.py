# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def showver(col):

    try:

        print("{} version: {}". format(col.__name__, col.__version__))

    except AttributeError:

        try:

            print("{} version: {}". format(col.__name__, col.version))

        except AttributeError:

            #print('{} Error', format(col))

            pass # Skip it

        

import sys #access to system parameters https://docs.python.org/3/library/sys.html

showver(sys)    



import numpy as np # linear algebra

showver(np)



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

showver(pd)



import missingno as miss

showver(miss)



import matplotlib.pyplot as plt

# showver(matplotlib)



import squarify

showver(squarify)



import random

showver(random)



import datetime

showver(datetime)



import re

showver(re)



from collections import Counter

showver(Counter)



from nltk.corpus import stopwords #removes and, in, the, a ... etc

showver(stopwords)



import plotly.express as px

showver(px)



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



print('-' * 43)

FILEPATH = '/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv'
df = pd.read_csv(FILEPATH)
df.describe()
df.info()
df.isnull().sum()
df.isna().sum()
# import missingno as miss
miss.matrix(df)
df.head()
# df.duplicated(subset=None, keep='first')
len(df[df.duplicated()])
def get_tech_keys(tag):

    

    if(not tag):

        return tag

    

    tag = tag.replace('><', ',')

    

    tag = tag.replace('<', '')

    

    tag = tag.replace('>', '')

    

    return tag
df['TechKeys'] = df['Tags'].apply(get_tech_keys)
df.head()
df_tech_keys = df[['TechKeys']]
df_tech_keys.head()
tech_key_list   = []

tech_key_values = None

index_counter = 0

tech_key_index_list = []

for (columnName, columnData) in df_tech_keys.iteritems():

    tech_key_values = columnData.values

    

for item in tech_key_values:

    item_parts = item.split(',')

    

    for item_ in item_parts:

        

        tech_key_index_list.append(index_counter)

        tech_key_list.append(item_)

        

        index_counter += 1

    

# tech_key_list
data = {'id' : tech_key_index_list, 'tech_key' : tech_key_list} 

  

# Create DataFrame 

df_tech_key_new = pd.DataFrame(data) 
df_tech_key_new.head()
len(df_tech_key_new)
df_tech_key_new.tech_key.value_counts().nlargest(10)
def get_tags_counts(col):

    

    if(not col):

        return 0

    

    tags_count = len(col.split(','))

    

    return tags_count
df['TagsCount'] = df['TechKeys'].apply(get_tags_counts)
df.head()
df_sub = df[['Id', 'Title', 'Tags', 'TagsCount']][0:25]
df_sub.head()
def highlight_max_custom(s, color = 'lightgreen'):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: '+color if v else '' for v in is_max]
df_sub.style.apply(highlight_max_custom, color = '#CFFE96',  axis = 0, subset=['TagsCount'])
df.Y.unique()
def get_question_level(level):

    

    if(not level):

        return level

    

    if(level == 'LQ_CLOSE'):

        return 3

    

    if(level == 'LQ_EDIT'):

        return 2

    

    if(level == 'HQ'):

        return 1

    

    return level
df['Level'] = df['Y'].apply(get_question_level)
df.head()
# import matplotlib.pyplot as plt



def show_donut_plot(col):

    

    rating_data = df.groupby(col)[['Id']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot: SOF Questions by ' +str(col), loc='center')

    

    plt.show()
show_donut_plot('Level')
show_donut_plot('TagsCount')
# import squarify



def show_treemap(col, max_labels = 10):

    df_type_series = df.groupby(col)['Id'].count().sort_values(ascending = False).head(20)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:max_labels],  # show labels for only first 10 items

                  alpha=.2 )

    

    plt.title('TreeMap: SOF Questions by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap('Level')
# print random body (random column data)

# import random



# df.at[df.index[random.randint(0, len(df))], 'Body']
def code_available(content):

    

    if('<code>' in content):

        return True

    

    return False
df['code_available'] = df['Body'].apply(code_available)
df.head()
show_donut_plot('code_available')
# import datetime



def get_week(col):

    

    return col.strftime("%V")
# Create new columns for Month, Year Created

df['CreationDatetime'] = pd.to_datetime(df['CreationDate']) 

df['CreationMonth'] = df['CreationDatetime'].dt.month.astype(int)

df['CreationYear'] = df['CreationDatetime'].dt.year.astype(int)

df['CreationWeek'] = df['CreationDatetime'].apply(get_week).astype(int)
# df.info()
show_donut_plot('CreationMonth')
show_treemap('CreationMonth')
show_donut_plot('CreationYear')
show_donut_plot('CreationWeek')
show_treemap('CreationYear')
show_treemap('CreationWeek', 18)
df_tech_key_new
def show_donut_plot_techkey(col):

    

    rating_data = df_tech_key_new.groupby(col)[['id']].count().head(50)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot by ' +str(col), loc='center')

    

    plt.show()
show_donut_plot_techkey('tech_key')
def show_treemap_tech_key(col):

    df_type_series = df_tech_key_new.groupby(col)['id'].count().sort_values(ascending = False).head(50)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:25],  # show labels for only first 10 items

                  alpha=.2 )

    plt.title('TreeMap by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap_tech_key('tech_key')
def show_donut_plot_2cols(col1, col1_val, col2):

    

    df1 = df[df[col1] == col1_val]

    

    rating_data = df1.groupby(col2)[['Id']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot by ' +str(col1) + ' and ' +str(col2), loc='center')

    

    plt.show()
show_donut_plot_2cols('CreationYear', 2016, 'Level')
show_donut_plot_2cols('CreationYear', 2016, 'code_available')
df.head()
df.Y.unique()
import re



code_start = '<code>'

code_end   = '</code>'



def get_codes(content):

    

    if('<code>' not in content):

        return None

    

    code_list = []

    

    loop_counter = 0

    while(code_start in content):



        code_start_index = content.index(code_start)

        if(code_end not in content):

            code_end_index = len(content)

        else:

            code_end_index = content.index(code_end)



        substring_1 = content[code_start_index : (code_end_index + len(code_end) )]

 

        code_list.append(substring_1)

        

        content = content.replace(substring_1, '')

        

        loop_counter += 1



    

    return ' '.join(code_list)



def  clean_text(content):

    

    content = content.lower()

    

    content = re.sub('<.*?>+', '', content)

    

    content = re.sub(r"(@[A-Za-z0-9]+)|^rt|http.+?", "", content)

    content = re.sub(r"(\w+:\/\/\S+)", "", content)

    content = re.sub(r"([^0-9A-Za-z \t])", " ", content)

    content = re.sub(r"^rt|http.+?", "", content)

    content = re.sub(" +", " ", content)



    # remove numbers

    content = re.sub(r"\d+", "", content)

    

    return content



# Clean the data

def clean_text_simple(text):

    text = text.lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text



def get_non_codes(content):

    

    loop_counter = 0

    while(code_start in content):



        code_start_index = content.index(code_start)

        if(code_end not in content):

            code_end_index = len(content)

        else:

            code_end_index = content.index(code_end)



        substring_1 = content[code_start_index : (code_end_index + len(code_end) )]



        content = content.replace(substring_1, ' ')

        

        loop_counter += 1

        

    content = clean_text_simple(content)



    return content
df['Body_code'] = df['Body'].apply(get_codes)

df['Body_content'] = df['Body'].apply(get_non_codes)
# from collections import Counter

# from nltk.corpus import stopwords



stopwords1 = stopwords.words('english')







df['content_words'] = df['Body_content'].apply(lambda x:str(x).split())
def remove_short_words(content):



    new_content_list = []

    for item in content:

        

        if(len(item) > 2):

            new_content_list.append(item)

    

    return new_content_list

    
df['content_words'] = df['content_words'].apply(remove_short_words)
df.head()
words_collection = Counter([item for sublist in df['content_words'] for item in sublist if not item in stopwords1])

freq_word_df = pd.DataFrame(words_collection.most_common(30))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
# import plotly.express as px



fig = px.scatter(freq_word_df, x="frequently_used_word", y="count", color="count", title = 'Frequently used words - Scatter plot')

fig.show()
fig = px.pie(freq_word_df, values='count', names='frequently_used_word', title='Stackoverflow Questions - Frequently Used Word')

fig.show()
fig = px.sunburst(df, path=['CreationYear', 'CreationMonth'], values='Level',

                  color='Level', hover_data=['Level'])

fig.show()
fig = px.sunburst(df, path=['CreationYear', 'CreationMonth'], values='code_available',

                  color='code_available', hover_data=['code_available'])

fig.show()
fig = px.strip(df, x="CreationMonth", y="code_available", orientation="h", color="CreationYear")

fig.show()
df.head()
df_text = df.drop(['Title', 'Body', 'Tags', 'CreationDate', 'TechKeys', 'TagsCount', 

                   'Y', 'code_available', 'CreationDatetime', 'CreationMonth', 'CreationYear', 

                   'CreationWeek', 'Body_code', 'content_words', 'Id'], axis = 1)
df_text.head()
# Define how much percent data you wanna split

split_count = int(0.23 * len(df_text))
# Shuffles dataframe

df_text = df_text.sample(frac=1).reset_index(drop=True)



# Training Sets

train = df_text[split_count:]

trainX = train['Body_content']

trainY = train['Level'].values



# Test Sets

test = df_text[:split_count]

testX = test['Body_content']

testY = test['Level'].values



print(f"Training Data Shape: {testX.shape}\nTest Data Shape: {testX.shape}")
from sklearn.feature_extraction.text import TfidfVectorizer



# Load the vectorizer, fit on training set, transform on test set

vectorizer = TfidfVectorizer()

trainX = vectorizer.fit_transform(trainX)

testX = vectorizer.transform(testX)
# commenting out as it is taking too much time

# from sklearn.svm import SVC



# svm_model = SVC(kernel='rbf', random_state=0, gamma=1, C=1).fit(trainX, trainY)



# svm_model
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier()

dt_model = dt_model.fit(trainX, trainY)



dt_model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



knn_model = knn.fit(trainX, trainY)



knn_model
models = [

#     svm_model,

    dt_model,

    knn_model

]
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
best_model_accuracy = 0

best_model = None



for model in models:

    

    model_name = model.__class__.__name__

    

    predY = model.predict(testX)

    accuracy = accuracy_score(testY, predY)

    

    print("-" * 43)

    print(model_name + ": " )

    

    if(accuracy > best_model_accuracy):

        best_model_accuracy = accuracy

        best_model = model_name

    

    print("Accuracy: {:.2%}".format(accuracy))
print("Best Model : {}".format(best_model))

print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))