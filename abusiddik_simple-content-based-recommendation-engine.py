import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv("../input/sample-data.csv")

ds.shape
ds.head()

ds.description
import re
p = re.compile(r'<.*?>')
ds['Cleaned_description'] = [p.sub('',each) for each in ds.description]
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['Cleaned_description'])
tfidf_matrix

cos_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
print(len(cos_similarities))
cos_similarities
for i,j in ds.iterrows():
    print(i,j)
    
    break

result_dict = {}

for idx, row in ds.iterrows():
    similar_ind = cos_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cos_similarities[idx][i], ds['id'][i]) for i in similar_ind]
    result_dict[row['id']] = similar_items[1:]

# return name of the Item
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary. No real logic here.
def recommend(item_id):
    num = 100
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = result_dict[item_id][:num]
    count = 0
    for rec in recs:
        if rec[0] > 0.1:
            if count < 11:
                print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
                count+=1
# Just plug in any item id here (1-500), and the number of recommendations you want (1-99)
# You can get a list of valid item IDs by evaluating the variable 'ds', or a few are listed below

recommend(item_id= np.random.randint(1,500,1)[0])