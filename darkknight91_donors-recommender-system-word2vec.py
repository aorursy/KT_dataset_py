import numpy as np
import pandas as pd
import time
import os
import pickle
import math
from gensim.models import Word2Vec,KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

# stop words are not used in the kernel anymore since I am using pre-trained model.
stop = set(stopwords.words('english'))
stop = stop.union(set(string.punctuation))
stop = text.ENGLISH_STOP_WORDS.union(stop)

translator = str.maketrans('', '', string.punctuation)

glove_input_file = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt' 
word2vec_output_file = 'glove_w2v.txt'
projects_file = "projects"
model_file = 'model'
update = False # if for some reason you want to update the loaded objects make it True
# ['glove.6B.50d.txt', 'glove.6B.200d.txt', 'glove.6B.300d.txt', 'glove.6B.100d.txt'] these are the available pre-trained models
def print_time(tag, start_time):
    """
    To print time difference with a text
    """
    print(tag,(time.time()-start_time))
    
def get_avg_vec(words):
    """
    This method accepts a string with multiple words and returns the 
    average of there word vectors.
    """
    avg = np.zeros((len(model["book"]),))
    for word in words.split():
        try:
            avg = avg + model[word.lower()]
        except:
            print("word not found",word)

    avg = avg/len(words.split())
    if(np.isnan(avg).any()):
        print("Nan",words,len(words.split()))
    return avg

def smooth_donor_preference(x):
    """
    To reduce large numbers to smaller, comparable values. I had found this idea in one of the kernels some time back.
    I do not take credit for this smoothening idea.
    """
    return math.log(1+x, 2)

def build_df_groupy_donor(df):
    df["eventStrength"] = df["Donation Amount"]
    return df.groupby(['Project ID','Donor ID'])['eventStrength'].sum().apply(smooth_donor_preference).reset_index()
print(os.listdir("../input/io"))
print(os.listdir("../input/glove-global-vectors-for-word-representation"))
model = None
if(not os.path.isfile(word2vec_output_file) or update):
    print("Glove model not pre-loaded. Loading now...")
    glove2word2vec(glove_input_file, word2vec_output_file)
    load_time = time.time()
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print_time("model load time",load_time)
    pickle.dump(model,open(model_file,'wb')) # store the word2vec model in output folder after building for first time to save time

if(model is None): # if model is already available in output just load it
    load_time = time.time()
    model = pickle.load(open(model_file,'rb'))
    print_time("model pickle load time",load_time)
print("size of output file",os.path.getsize(word2vec_output_file)//(1024*1024),"MB")
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1) # just to check if model is actually working
print("Testing Word2Vec model (Result should be 'queen')",result)
print(model.most_similar(positive=["book"],topn=2))
from sklearn.model_selection import train_test_split
df_projects = None
if(not os.path.isfile(projects_file) or update):
    print("No pre-loaded projects found. Loading now...")
    df_projects = pd.read_csv("../input/io/Projects.csv", low_memory=False)
    df_projects = df_projects[0:50000] # reading only 50K rows to save time
    pickle.dump(df_projects,open(projects_file,'wb'))

print("size of projects file",os.path.getsize(projects_file)//(1024*1024),"MB")
if(df_projects is None): # if already in storage, load it.
    df_projects = pickle.load(open(projects_file,'rb'))

df_projects["cat_res"] = df_projects["Project Subject Category Tree"]+" "+df_projects["Project Resource Category"]
df_projects,df_te = train_test_split(df_projects,train_size=0.9,test_size=0.1)
print("split sizes",len(df_projects),len(df_te))
# drop projects with no category
df_projects.dropna(subset=['Project Subject Category Tree',"Project Resource Category"],inplace=True)
categories = df_projects["cat_res"].unique() # df_projects["Project Subject Category Tree"].unique()
print("These are the overall categories+resource in use till now",len(categories))
print(categories)

load_time = time.time()
df_donations = pd.read_csv("../input/io/Donations.csv", low_memory=False)
df_gp = build_df_groupy_donor(df_donations)
#df_gp.set_index("Project ID",inplace=True)
display(df_gp[0:50])
print_time("Donations load time",load_time)
#display(df_projects[df_projects["Project Subject Category Tree"] == 'Math & Science, Literacy & Language'].head())
text_ms_ll = df_projects[df_projects["Project Subject Category Tree"] == 'Math & Science, Literacy & Language'].iloc[0]["Project Essay"]
print(text_ms_ll)
stime = time.time()
cat_vec = np.zeros((len(categories),len(model["book"])))
print("Number of categories",len(categories))
count = 0
"""
'categories' holds all the unique categories used in projects dataset.
we calculate average word vectors for each of them and store in 'cat_vec'
"""
for cat in categories:
    #print("Real-",cat.strip(),"--trans-",cat.strip().translate(translator))
    words = cat.strip().translate(translator)
    cat_vec[count] = get_avg_vec(words)
    count = count + 1

print("category vectors calculated",cat_vec.shape)

def get_similar_category(cats):
    """
    parameter : 'n' category strings
    returns : list of most similar category (from projects dataset) to each of the 'n' categories sent as parameter
    """
    #arr.argsort()[-3:]
    sim_cats = []
    for cat in cats:
        words = cat.strip().translate(translator)
        avg = get_avg_vec(words)
        res = cosine_similarity(cat_vec,avg.reshape(1,-1))
        max_ind = np.argmax(res)
        print("Given category- ",cat,", Most similar category- ",categories[max_ind])
        sim_cats.append(categories[max_ind])
    return sim_cats

test_cats = list(df_te.sample(3)["cat_res"]) # storing as list so that we can append other values which are not in dataset
test_cats.append("Math & Science Books")
test_cats.append("Musical instrument Others")
res = get_similar_category(test_cats) # array of matching categories
print("\n")
for cat in res:
    print("For category",cat)
    df_temp = df_projects[df_projects["cat_res"] == cat].reset_index() # 'Project Subject Category Tree' keep only fields needed
    #display(df_temp)
    df_cat_projs = df_gp[df_gp["Project ID"].isin(df_temp["Project ID"])]#.sort_values("eventStrength",ascending=False)
    df_cat_projs = df_cat_projs.groupby(["Project ID","Donor ID"]).agg({'eventStrength':sum})
    df_cat_projs.sort_values("eventStrength",ascending=False,inplace=True)
    # suggest donors who have donated most genorously for better turnover and brevity of display
    df_cat_projs = df_cat_projs[df_cat_projs["eventStrength"] > 7]
    print("Suggested donors ({0}) for category {1}".format(len(df_cat_projs),cat))
    display(df_cat_projs[0:20])

print_time("End time",stime)
# 'Math & Science, Literacy & Language', 'Health & Sports, History & Civics'
#next think about collborative filtering with matrix [projects, donors]