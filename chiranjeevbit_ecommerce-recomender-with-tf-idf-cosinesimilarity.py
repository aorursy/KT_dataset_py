import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
from __future__ import division

ds = pd.read_csv("../input/sample-data.csv")


ds.head(20)
print(ds['description'][0])
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])



cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}


for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results[row['id']] = similar_items[1:]
print('done!')

#declare list for generateing testing sorce
score_count = []

# hacky little function to get a friendly item name from the description field, given an item ID
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary. No real logic here.
def recommend(item_id, num):
    print("Recommending ----->>>>>>> " + str(num) + " products similar to " + item(item_id) + "...")
    recs = results[item_id][:num]
    i=0
    for rec in recs:
        ## print recommendaed items 
        print("Recommended items: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
#### Start testign logic
        if rec[0] >0.15:
            i+=1
    print ("Recommendation score: "+ str(i))
    score_count.append(str(i))
    return Counter(score_count)
#### End Testing logic
    


## Testing on all items so generate recommendation for all items

for i in range(1,100):
    # print 
    score_counter_variable = recommend(item_id=i, num=5)
    print("-----------Recommendation for a single item ends here-----------")
    print()
    
## Generate test score in percentage
no_of_item =100
score_dict = dict(score_counter_variable)              
score_0 = (score_dict.get("0")/no_of_item)*100
score_1 = (score_dict.get("1")/no_of_item)*100
score_2 = (score_dict.get("2")/no_of_item)*100
score_3 = (score_dict.get("3")/no_of_item)*100
score_4 = (score_dict.get("4")/no_of_item)*100
score_5 = (score_dict.get("5")/no_of_item)*100

score_0 =round((score_0),2)
score_1 =round((score_1),2)
score_2 =round((score_2),2)
score_3 =round((score_3),2)
score_4 =round((score_4),2)
score_5 =round((score_5),2)


print ("0 useful recommendation: "+ str(score_0)+'%')
print ("1 useful recommendation: "+ str(score_1)+'%')
print ("2 useful recommendation: "+ str(score_2)+'%')
print ("3 useful recommendation: "+ str(score_3)+'%')
print ("4 useful recommendation: "+ str(score_4)+'%')
print ("5 useful recommendation: "+ str(score_5)+'%')



