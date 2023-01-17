# Currently WIP
import pandas as pd

import numpy as np
df = pd.read_csv('../input/redbubble_featured2 (1).csv')
df.head()
# replace HTML code with numerical 1-5 rating

for column in range(len(df['star_rating'])):

    df['star_rating'][column] = df['star_rating'][column].count("255")





# df['star_rating'][1] = df['star_rating'][1].count("255")



df.head()
import matplotlib.pyplot as plt



objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')

y_pos = np.arange(len(objects))

performance = [10,8,6,4,2,1]



plt.figure(figsize=(20,10))

plt.bar(df['star_rating'].sort_values().unique(), df['star_rating'].value_counts().sort_index())

plt.show()
import re 

  

# Function to clean the names. Removes all excess text so only the name of the reviewer is given

def clean_reviewer(reviewer): 

    # Search for opening bracket in the name followed by 

    # any characters repeated any number of times 

    if re.search('by ', reviewer): 

  

        # Extract the position of end and then beginning of pattern 

        pos = re.search('by ', reviewer).end() 

        pos2 = re.search(' on', reviewer).start() 

  

        # return the cleaned name 

        return reviewer[pos:pos2] 

  

    else: 

        # if clean up needed return the same name 

        return reviewer 
for column in range(len(df['star_rating'])):

    df['reviewer'][column] = clean_reviewer(df['reviewer'][column])

    

df.head()
df['reviewer'].value_counts()
df['review'].value_counts()
reviewer_dict = {}



for value in df['reviewer']:

    if value not in reviewer_dict:

        reviewer_dict[value] = 1

    else:

        reviewer_dict[value] += 1



reviewer_dict
df_reviewer = pd.DataFrame(list(reviewer_dict.items()), columns = ['User', '# Reviews'])

df_reviewer.sort_values('# Reviews', ascending = False).head(20)
# Sanity checking above code

(df['reviewer'] == 'Amy').sum()
df.sort_values('link').head()
# Total of 59 different products

df['link'].nunique()
from fastai.collab import *

from fastai.tabular import *
data = CollabDataBunch.from_df(df, seed=42, valid_pct=0.1, user_name = 'reviewer', item_name = 'link', rating_name = 'star_rating')
data
data.show_batch()
y_range = [1.0,5.0]
learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 5e-3)
learn.model
learn.model.u_bias
# Actual first 5 ratings

df['star_rating'].head()
# Predicted first 5 ratings

for i in range(0,5):

    print(learn.predict(df.iloc[i]))
g
g = df.groupby('link')['star_rating'].count()

top_items = g.sort_values(ascending=False).index.values[:1000]

# top_items = g.sort_values(ascending=False).index



top_items[:10]
# TODO: BLOCKED. Should be creating an array of tensors but is not?

learn.bias(top_items)
item_bias = learn.bias(top_items, is_item=True)

item_bias.shape