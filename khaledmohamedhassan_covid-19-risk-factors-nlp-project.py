# Member_1_Email: omar.mahmoudsaleh@student.guc.edu.eg
# Member_1_Name: Omar Ashraf Mahmoud Saleh
# Member_1_ID: 37-15763
    
# Member_2_Email: khaled.abdelawaad@student.guc.edu.eg
# Member_2_Name: Khaled Mohamed Ahmed Hassan Abdelawaad
# Member_2_ID: 37-2956

# Member_3_Email: yara.dorgham@student.guc.edu.eg
# Member_3_Name: Yara Gamal Dorgham
# Member_3_ID: 37-2464

# Member_4_Email: karim.attia@student.guc.edu.eg
# Member_4_Name: Karim Mohamed Nashaat
# Member_4_ID: 37-10478

# Member_5_Email: khaled.khaled@student.guc.edu.eg
# Member_5_Name: Khaled Ahmed Khaled
# Member_5_ID: 37-8970


import string
import math
import json
import spacy
import os
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer




# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens
relative_words = ['smoke','lungs','lung'
                  'pulmonary','disease','neonates','pregnant','incubation','fatality','hospitalized'
                 'asthma', 'chronic' 'bronchitis', 'bronchiectasis','copd', 'tobacco', 'smoking','pneumonia'
                 ,'viral', 'infections', 'infection','sbestosis','asthma','bronchiectasis','bronchitis','transmission',
                  'basic', 'reproductive','serial', 'interval','high-risk']

covid = ['corona','covid','coronavirus','covid-19','COVID','COVID-19']
# # final cell all in one (final run )

relevent = []
irrelevent = []
partialy_relevent = [] 
all_data = []
i = 0



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pdf = os.path.join(dirname, filename)
        try :
            # using the first 20,000 documents 
            if(i<20000):
                if ".json" in pdf :
                    with open(pdf) as json_file:
                        data = json.load(json_file)
                    text = str((data['body_text']))
                    sample = ""
                    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
                    if(len(text)<1000000):
                        cleaned_text = spacy_tokenizer(text)
                        for txt in cleaned_text:
                            sample = sample+" "+txt


                        # just send in all your docs here
                        tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform([sample])
                        first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

                        # place tf-idf values in a pandas data frame
                        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), columns=["tfidf"])

                        df.unstack()
                        df.sort_values(by=["tfidf"],ascending=False)
                        df.insert(0, "Word", tfidf_vectorizer.get_feature_names(), True)
                        tempr = np.array([data['paper_id'], data['metadata']['title'],df])
                        
                        mean = tempr[2].mean().get(0)
                        tf = tempr[2]
                        temp = (tf.loc[tf['Word'].isin(covid)]).mean().get(0)
                        if ( math.isnan(temp) | (temp<mean) ):
                            irrelevent.append(tempr)
                        else :

                            if (temp>=mean):
                                temp2 = (tf.loc[tf['Word'].isin(relative_words)]).mean().get(0)
                                
                                if( math.isnan(temp2) | (temp2<mean)):
                                    partialy_relevent.append(tempr)                                    

                                else:
                                    if(temp2>=mean):

                                        relevent.append(tempr)                             

                        
                        
                        all_data.append(tempr)
                    i = i +1
                    

                    
        except Exception:
            pass

np.save('all_data', all_data)
np.save('relevent', relevent)
np.save('irrelevent', irrelevent)
np.save('partialy_relevent', partialy_relevent)


rel = np.load('relevent.npy',allow_pickle=True)




ids = []
titles = []
idfs = []

for r in rel:
    ids.append(r[0])
    titles.append(r[1])
    idfs.append(r[2].loc[r[2]['Word'].isin(covid)].mean().get(0))
    
relevent_df = pd.DataFrame()

relevent_df.insert(0, "paper_id", ids, True)
relevent_df.insert(1, "paper_title", titles, True)    
relevent_df.insert(2, "tf-idf", idfs, True)
relevent_df = relevent_df.sort_values(by=['tf-idf'],ascending=False)
relevent_df = relevent_df.reset_index(drop=True)
relevent_df.head(20)