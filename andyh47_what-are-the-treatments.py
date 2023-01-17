%matplotlib inline

from functools import reduce

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import recall_score,precision_score,f1_score

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from IPython.core.display import display, HTML
# read a list of 200 common drugs

drugs = pd.read_html('https://clincalc.com/DrugStats/Top200Drugs.aspx')[0]

# add known interesting theraputics

promising = pd.Series(['chloroquine','hydrochloroquine','remdesivir','quercetin']).to_frame()

promising.columns =  ['Drug Name']

promising['Rank'] = np.nan

promising['Total Prescriptions (2017)'] = np.nan

promising['Annual Change'] = np.nan

drugs = drugs.append(promising,ignore_index=True,sort=False)
drugs.head()
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

display(HTML(f'meta data shape: {metadata.shape}'))

has_abstract = metadata.abstract.apply(lambda x: str(x)!='nan')

metadata = metadata.iloc[has_abstract.values,:]

display(HTML(f'meta data shape after dropping docs without abstracts: {metadata.shape}'))

metadata.head(n=2).transpose()
def has_word(text_string,filter_words):

    def has_this_word(has_one,word):

        if has_one:

            return True

        else:

            if word in text_string:

                return True

            else:

                return False

    return reduce(has_this_word,filter_words,False)



FILTER_WORDS = ['SARS','MERS','corona','Cov','COV'] #keyword strings used to filter titles



# filter to titles with covid words

have_filter_word = metadata.abstract.apply(lambda x: has_word(x,FILTER_WORDS)) 

metadata_has_corona = metadata[have_filter_word]



# filter to drug words

have_filter_word = metadata.abstract.apply(lambda x: has_word(x,drugs['Drug Name']))  

metadata_has_drug = metadata[have_filter_word]



# filter to theraputics

theraputic_words = ['anti-viral','antiviral']

have_filter_word = metadata.abstract.apply(lambda x: has_word(x,theraputic_words))  

metadata_has_theraputic = metadata[have_filter_word]



# filter for antivirals and arb_blockers

def regex_search(string,pattern):

    return True if re.search(pattern,string) else False



have_filter_word = metadata.abstract.apply(lambda x: regex_search(x,'[a-zA-Z]+vir ')) 

metadata_has_antiviral = metadata[have_filter_word]



# filter for arb blockers

have_filter_word = metadata.abstract.apply(lambda x: regex_search(x,'[a-zA-Z]+sartan ')) 

metadata_has_arb_blocker = metadata[have_filter_word]



#build positives

X1 = metadata_has_drug.append(metadata_has_antiviral)

X2 = X1.append(metadata_has_arb_blocker)



#build negatives 

negatives_index = np.random.choice(metadata.index,size=1000,replace=False)

negatives_index = [x for x in negatives_index if x not in X2.index]

X = X2.append(metadata.loc[negatives_index]).abstract



# build ground truth

y = pd.Series([1]*len(X2) + [0]*(len(X) - len(X2)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(ngram_range=(1,1),stop_words='english',max_df=.75,min_df=1)

bow = pd.DataFrame((tfidf.fit_transform(X_train).todense()))

display(HTML( f'Bag of Words shape {bow.shape}'))



rf = RandomForestClassifier(n_estimators=300,

    min_samples_leaf=5,

    oob_score=True)

rf.fit(bow,y_train)

display(HTML(f'RandomForest Out-of-Bag Score: {rf.oob_score_}'))
bow_test = pd.DataFrame((tfidf.transform(X_test).todense()))

y_prob = rf.predict_proba(bow_test)[:,1]

y_pred = [1 if x > .5 else 0 for x in y_prob]

_ = plt.hist(y_prob[y_test==1],bins=20,label='Positives',alpha=.5)

_ = plt.hist(y_prob[y_test==0],bins=20,label='Negatives',alpha=.5)

plt.legend(loc='upper right')

_ = plt.title('Class Probability Distribution')

plt.gcf().set_size_inches((8,5))
thresh = .5  # adjust to tune precision, recall

y_pred = pd.Series([1 if x > thresh else 0 for x in y_prob])

display(HTML('Confusion Matrix' ))

display_df = pd.DataFrame(confusion_matrix(y_test,y_pred))

display_df.columns = ['Predicted 0','Predicted 1']

display_df.index = ['True 0','True 1']

display(HTML(display_df.to_html()))
display(HTML(f'Recall {recall_score(y_test,y_pred)}'))

display(HTML(f'Precision {precision_score(y_test,y_pred)}'))

bow_covid = tfidf.transform(metadata_has_corona.abstract)

y_prob = rf.predict_proba(bow_covid)[:,1]

y_pred = [True if x > thresh else False for x in y_prob]

display(HTML( f'Bag of Words shape {bow_covid.shape}'))
def print_sample(docs,n=20):

    for d in docs[0:n]:

        print(d + '\n')



print_sample(metadata_has_corona.abstract.loc[y_pred])
display(HTML(f'Found {np.sum(y_pred)} papers'))
df_s = metadata_has_corona.loc[y_pred,['title','abstract','doi']]

#convert to html

df_s['title'] = '<span style="float: left; width: 100%; text-align: left;">' + df_s['title'] + '</span>'

df_s['abstract'] = '<span style="float: left; width: 80%; text-align: left;">' + df_s['abstract'] + '</span>'

df_s['doi'] = '<a href = "https://doi.org' + df_s['doi'] + '" target="_blank">link</a>'

result = HTML(df_s.to_html(escape=False))

display(result)

display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
# Save file

metadata_has_corona.loc[y_pred].to_csv('/kaggle/working/theraputics_compounds_alh.csv')