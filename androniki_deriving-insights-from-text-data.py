# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# endoder for the categorical columns
from sklearn.preprocessing import LabelBinarizer
from math import sin, cos, sqrt, atan2, radians

#word cloud
from os import path
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
import random
import scipy.stats as st

# for doc2vec - use of loan
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join

#plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
kiva_mpi_locations_df = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/loan_theme_ids.csv")
kiva_loans_df = pd.read_csv("../input/kiva_loans.csv")
#loan_theme_ids_df.head()
#kiva_mpi_locations_df.head()
#kiva_loans_df.head()
def number_of_borrowers(x):
    if type(x['borrower_genders']) in [float]:
        return 0
    else:
        y= x['borrower_genders'].split(',')
        return len(y)

def number_of_fem_borrowers(x):
    if type(x['borrower_genders']) in [float]:
        return 0
    else:
        y = x['borrower_genders'].split(',')
        y = [str(n.strip()) for n in y]
        return y.count('female')
    

def distance(x1,x2,y1,y2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(x1)
    lon1 = radians(x2)
    lat2 = radians(y1)
    lon2 = radians(y2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance
tokenizer = RegexpTokenizer('\w+')
stopword_set = set(stopwords.words("english"))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.TaggedDocument(doc,[self.labels_list[idx]])
def plot_num_borrowers():
    '''understanding the sense of community in each place is an important factor to understand how best to allocate loans, 
    we look here which countries create cooperatives and if there is any relationship with the loan funded'''
    cnt_srs = kiva_loans_df1.groupby('country')["number_of_borrowers"].mean().sort_values(ascending=False).head(20)
    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        orientation = 'h',
        marker=dict(
            color=cnt_srs.values[::-1],
            colorscale = 'Viridis',
            reversescale = True
        ),
    )

    layout = go.Layout(
        title='Mean number of borrowers per loan',
        width=700,
        height=1000,
        )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename="CountryLoan")
    
def scatter():
    r = random.sample(range(1, kiva_loans_df1.shape[0]), 10000)
    x = kiva_loans_df1['number_of_borrowers'][r]
    y = np.log(kiva_loans_df1['loan_amount'][r])
    s = np.random.rand(*x.shape) * 0.2 + 50

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.xlabel("Number of borrowers")
    plt.ylabel("log of Loan amount")
    plt.legend(loc=2)
    plt.show()
    
def word_cloud():
        # Simple WordCloud
    plt.figure(figsize=(20,10))
    r = random.sample(range(1, df2.shape[0]), 15000)
    text = ' '.join(str(m) for m in df2['use'][r])

    wordcloud = WordCloud(
                          relative_scaling = 1.0,
                          stopwords = {'when','they','his','her','to', 'of','in','their','for','my','more','and','the','such','as'} # set or space-separated string
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
#BINARISE SECTOR
lb_style = LabelBinarizer()

lb_results1 = lb_style.fit_transform(kiva_loans_df["sector"])
X1=pd.DataFrame(lb_results1,columns=lb_style.classes_)

#BINARISE REPAYMENT INTERVAL
lb_results2 = lb_style.fit_transform(kiva_loans_df["repayment_interval"])
X2=pd.DataFrame(lb_results2,columns=lb_style.classes_)

# PERCENTAGE OF LOAN FUNDED by KIVA
kiva_loans_df["Perc_loan_funded"] = kiva_loans_df["funded_amount"]/kiva_loans_df["loan_amount"]*100.0

# BORROWERS GENDER AND NUMBER
kiva_loans_df['number_of_borrowers'] = kiva_loans_df.apply(number_of_borrowers,axis=1)
kiva_loans_df['number_of_fem_borrowers'] = kiva_loans_df.apply(number_of_fem_borrowers,axis=1)

XX = pd.concat([X1,X2],axis=1,join="inner")
kiva_loans_df1 = pd.concat([kiva_loans_df.drop(["sector","repayment_interval","borrower_genders"],axis=1),XX],axis=1,join="inner")
# join the loans with the purpose of the loan

df10 = pd.merge(kiva_loans_df1,loan_theme_ids_df,on="id",how="left")
df11 = pd.merge(df10,kiva_mpi_locations_df,on="country",how="left")


#BINARIZE WORLD REGIONS
df11["world_region"] = df11.world_region.replace(np.NaN, "other")
lb_results3 = lb_style.fit_transform(df11["world_region"])
X3=pd.DataFrame(lb_results3,columns=lb_style.classes_)

df2 = pd.concat([df11.drop("world_region",axis=1),X3],axis=1)
st.pearsonr(np.log(kiva_loans_df1['loan_amount']),kiva_loans_df1['number_of_borrowers'])
df2.head()
from numpy import genfromtxt,array,linalg,zeros,apply_along_axis

data = df2.drop(["id","activity","use","country_code","country","region_x","currency","partner_id","posted_time","disbursed_time","funded_time","tags",\
                 "date","Loan Theme ID","Loan Theme Type","Partner ID","LocationName","ISO","region_y","geo"],axis=1)
# normalization to unity of each pattern in the data
data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)
data = ''.join(str(m) for m in df2['use'])
data = data.split(".")
docLabels = []
docLabels = [str(f) for f in range(0,len(data))]
data = nlp_clean(data)
it = LabeledLineSentence(data, docLabels)
#model = gensim.models.Doc2Vec(vector_size=20, min_count=5, alpha=0.025, min_alpha=0.025)
#model.build_vocab(it)
#training of model
#for epoch in range(1):
# print("iteration " + str(epoch+1))
# model.train(it,total_examples=model.corpus_count,epochs=model.epochs)
#    #saving the created model
# model.save("doc2vec.model")
#print("model saved")