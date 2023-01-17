import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
news_data = pd.read_csv("../input/news-data.csv")
news_data.shape
news_data.head()
NUM_SAMPLES = 12000 # The number of sample to use 
sample_df = news_data.sample(NUM_SAMPLES, replace=False).reset_index(drop=True)
sample_df.shape
sample_df.head()
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
dtm = cv.fit_transform(sample_df['headline_text'])
dtm
feature_names = cv.get_feature_names()
len(feature_names) # show the total number of distinct words
feature_names[6500:]
NUM_TOPICS = 7 
LDA_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=30, random_state=42)
LDA_model.fit(dtm)
len(feature_names)
import random 
for index in range(15):
    random_word_ID = random.randint(0, 6506)
    print(cv.get_feature_names()[random_word_ID])
len(LDA_model.components_[0])
# Pick a single topic 
a_topic = LDA_model.components_[0]

# Get the indices that would sort this array
a_topic.argsort()
# The word least representative of this topic
a_topic[597]
# The word most representative of this topic
a_topic[3598]
top_10_words_indices = a_topic.argsort()[-10:]

for i in top_10_words_indices:
    print(cv.get_feature_names()[i])
for i, topic in enumerate(LDA_model.components_):
    print("THE TOP {} WORDS FOR TOPIC #{}".format(10, i))
    print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])
    print("\n")
final_topics = LDA_model.transform(dtm)
final_topics.shape
final_topics[0]
final_topics[0].argmax()
sample_df["Topic N°"] = final_topics.argmax(axis=1)
sample_df.head()
import pyLDAvis.sklearn
pyLDAvis.enable_notebook() # To enable the visualization on the notebook
panel = pyLDAvis.sklearn.prepare(LDA_model, dtm, cv, mds='tsne') # Create the panel for the visualization
panel
