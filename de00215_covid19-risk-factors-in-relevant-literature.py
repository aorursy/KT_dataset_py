import spacy
import gzip, pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
vocabularies=[
    'viruses',
    'diseases',
    'bacteria'
]
def read_df(file_name):
    with tqdm.wrapattr(gzip.open(file_name, 'rb'), "read", desc='read from ' +file_name) as file:
        return pickle.load(file)
def get_vocabulary_id_map(v):
    id_map={}
    with open(f'../input/snmi-disease-vocabularyjson/{v}.json') as file:
        v_json = json.load(file)
        for key, value in v_json.items():
            id_map[value['ID']]=value
    return id_map
all_vocabulary_id_map={}
for v in vocabularies:
    all_vocabulary_id_map[v]=get_vocabulary_id_map(v)
    print('load vocabulary for', v, len(all_vocabulary_id_map[v].keys()))

def wordcloud(df, name, title = None):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white",
        max_words=200,
        max_font_size=40,
        scale=4,
        random_state=0
    ).generate_from_frequencies(df)

    wc.recolor(color_func=wcolors)
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wc),
    wc.to_file(name)
    plt.show()

def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]
    return np.random.choice(colors)
for vocabulary in vocabularies: 
    # load pre-built DataFrame
    df = read_df(f'../input/cord19precomputeddata/corona_mentioned_with_{vocabulary}_mentioned.df.pklz')
    # get id map for combining text matched
    id_map=all_vocabulary_id_map[f'{vocabulary}']
    # combine text matched to the label of the term.
    df[f'{vocabulary}_label_combined']=df[f'{vocabulary}_mentions'].apply(lambda mentions: [id_map[mention['id']]['properties']['label'] for mention in mentions])
    
    # concatenate all labels
    res = np.concatenate([labels for labels in df[f'{vocabulary}_label_combined']])

    # remove common terms for viruses, diseases and bacteria
    if vocabulary == 'viruses':
        res = [r for r in res if r != 'Coronavirus' and r != 'Virus' ]
    if vocabulary == 'diseases':
        res = [r for r in res if r != 'Disease']
    if vocabulary == 'bacteria':
        res = [r for r in res if r != 'Bacterium']
        
    freqs = pd.Series(res).value_counts()
    wordcloud(freqs, f'word-cloud/corona-{vocabulary}.png', f'Most frequent words for matched {vocabulary}')
# import required libraries
import json
import spacy
from unidecode import unidecode
import glob
import json
import os
import sys
import pickle, gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
nlp = spacy.load("en_core_web_sm")
# read all data files
all_data_files=glob.glob(f'data/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*.json', recursive=True)
len(all_data_files)
# read all vocab files
all_vocab_files=glob.glob(f'Owncloud/CORD-19-Hackathon/V-team/vocabulary/*.json', recursive=True)
len(all_vocab_files)
print(all_vocab_files)
# get all labels from all vocab json
all_labels = []
for file in all_vocab_files:
    #print(file)
    with open(file) as json_file :
        virus_data = json.load(json_file)
        max_count = max(virus_data.keys())
        for count in range(0, int(max_count)+1):
            all_labels.append(virus_data["{}".format(count)]["properties"]["label"])
        print(all_labels)
# remove unwanted labels from all_labels
new_all_labels = []
removed_labels = []
unwanted_labels = ['coronavirus','Coronavirus','disease','Disease','bacteria','Bacteria','virus','Virus']
for label in all_labels:
    if label in unwanted_labels:
        removed_labels.append(label)
print("labels removed are - ", removed_labels)
new_all_labels = list(set(all_labels) - set(removed_labels)) 
print(new_all_labels)
# get all vocab labels with synonyms
synonyms = []
synonyms_dict = {}
for file in all_vocab_files:
    print(file)
    with open(file) as json_file :
        virus_data = json.load(json_file)
        max_count = max(virus_data.keys())
        for count in range(0, int(max_count)+1):
            #labels.append(virus_data["{}".format(count)]["properties"]["label"])
            #synonyms.extend(virus_data["{}".format(count)]["properties"]["synonyms"])
            synonyms_dict.update({virus_data["{}".format(count)]["properties"]["label"]: virus_data["{}".format(count)]["properties"]["synonyms"]})
#print(labels)
#print(synonyms)
print(synonyms_dict)
# pickles file generated locally
with gzip.open('NLP/mentions/all_data_corona_mentioned.pklz') as file:
    df=pickle.load(file)
df.head()
corona_mentioned_paper_ids = df['paper_id']
#print(corona_mentioned_paper_ids)
corona_mentioned_paper_ids_list = df['paper_id'].tolist()
print(corona_mentioned_paper_ids_list)
# list of all corona_mentioned_files
corona_mentioned_file_ids = []
corona_mentioned_files = []
for file in all_data_files:
    with open(file) as json_file :
        file_name = file.split("/")[-1].split(".")[0]
        #print(file_name)
        if file_name in corona_mentioned_paper_ids_list:
            corona_mentioned_file_ids.append(file_name)
            corona_mentioned_files.append(file)
#print(corona_mentioned_files)
print("no. of corona_mentioned_files = ", len(corona_mentioned_files))
print("no. of all_data_files = ", len(all_data_files))
# occurence of all labels in corona_mentioned_paper_ids_list

final_dict = {}
id_twc_dict = {}

for file in corona_mentioned_files:
    label_dict = {}
    with open(file) as json_file :
        file_name = file.split("/")[-1].split(".")[0]
        #print(file_name)
        data = json.load(json_file)
        str = ""
        try:
            for k in data['abstract']:
                str = str + k['text']                     
            doc = nlp(unidecode(str)) 
            #print (doc,"\n")
            
            new_all_labels = set(new_all_labels)
            #print(new_all_labels)
            for label in new_all_labels:
                counter=0
                for word in doc:             
                    if(word.text.lower() == label.lower()):
                        counter = counter + 1
                        for lbl in synonyms_dict[label]:
                            if (word.text == lbl):
                                counter = counter + 1
                if len(doc) > 0:
                    freq = (counter/len(doc))*100
                else:
                    freq = 0
                if counter > 0:
                    label_dict.update({label:counter})
                    print(label,"appears in",file_name,counter,"times",", i.e.(",freq,"%, as this text has",len(doc),"\"tokens\")")
                    id_twc_dict.update({file_name:len(doc)})
                    final_dict[file_name] = label_dict
        except:
            print(sys.exc_info[0])

print(id_twc_dict)   
print(final_dict)

with open('total_word_count_in_corona_mentioned_files.json', 'w') as outfile:
    json.dump(id_twc_dict, outfile)

with open('all_labels_occurences_in_corona_mentioned_files.json', 'w') as outfile:
    json.dump(final_dict, outfile)
#print(final_dict)
# pandas df for labels and occurences in corona_mentioned_files
labels_occurences_corona_df = pd.read_json (r'all_labels_occurences_in_corona_mentioned_files.json')
print(labels_occurences_corona_df)
labels_occurences_corona_df.to_csv('labels_occurences_corona_df.csv')
# occurence of all labels in all research papers

final_dict = {}

#id_twc_df = pd.DataFrame(columns=['paper_id','total_word_count'])
id_twc_dict = {}

for file in all_data_files:
    label_dict = {}
    with open(file) as json_file :
        file_name = file.split("/")[-1].split(".")[0]
        #print(file_name)
        data = json.load(json_file)
        str = ""
        try:
            for k in data['abstract']:
                str = str + k['text']                     
            doc = nlp(unidecode(str)) 
            #print (doc,"\n")

            new_all_labels = set(new_all_labels)
            #print(new_all_labels)
            for label in new_all_labels:
                counter=0
                for word in doc:             
                    if(word.text.lower() == label.lower()):
                        counter = counter + 1
                        for lbl in synonyms_dict[label]:
                            if (word.text == lbl):
                                counter = counter + 1
                if len(doc) > 0:
                    freq = (counter/len(doc))*100
                else:
                    freq = 0

                if counter > 0:
                    label_dict.update({label:counter})

                    print(label,"appears in",file_name,counter,"times",", i.e.(",freq,"%, as this text has",len(doc),"\"tokens\")")
                    id_twc_dict.update({file_name:len(doc)})
                    final_dict[file_name] = label_dict
                    #final_dict.update({file_name:label_dict})
        except:
            print(sys.exc_info[0])                

print(id_twc_dict)   
print(final_dict)

with open('total_word_count_in_all_files.json', 'w') as outfile:
    json.dump(id_twc_dict, outfile)

with open('all_labels_occurences_in_all_files.json', 'w') as outfile:
    json.dump(final_dict, outfile)
#print(final_dict)
# pandas df for labels and occurences in all files
labels_occurences_all_df = pd.read_json (r'all_labels_occurences_in_all_files.json')
print(labels_occurences_all_df)
labels_occurences_all_df.to_csv('labels_occurences_all_df.csv')
import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
# loading data from precomputed files
root="postsubmission/"
#reading noncommon-use-subset
ncu_labelocc = pd.read_csv(root + "ncu-labels_occurences_all_df.csv")
ncu_labelocc.rename(columns={ ncu_labelocc.columns[0]: "term" }, inplace = True)
ncu_labelocc_c=pd.read_csv(root + "ncu-labels_occurences_corona_df.csv")
ncu_labelocc_c.rename(columns={ ncu_labelocc_c.columns[0]: "term" }, inplace = True)
#ncu_labelocc_c)

#reading biorxiv
bio_labelocc = pd.read_csv(root +"biorxiv-labels_occurences_all_df.csv")
bio_labelocc.rename(columns={ bio_labelocc.columns[0]: "term" }, inplace = True)
bio_labelocc_c=pd.read_csv(root +"biorxiv-labels_occurences_corona_df.csv")
bio_labelocc_c.rename(columns={ bio_labelocc_c.columns[0]: "term" }, inplace = True)
#print(bio_labelocc_c)
#print(ncu_labelocc_c)
# joining ncu and bio dataframes

labelocc = pd.concat([ncu_labelocc, bio_labelocc], ignore_index=True)
#labelocc=ncu_labelocc
labelocc=labelocc.groupby("term").sum().reset_index()
labelocc.dropna(inplace=True)


labelocc_c = pd.concat([ncu_labelocc_c, bio_labelocc_c], ignore_index=True)
#labelocc_c=ncu_labelocc
labelocc_c=labelocc_c.groupby("term").sum().reset_index()
labelocc_c.dropna(inplace=True)


#optional: save joined dfs to files
#labelocc.to_csv("labels_occurences_corona_df_allds.csv")
#labelocc_c.to_csv("labels_occurences_all_df_allds.csv")

#import precomputed abstract word counts
with open(root +'ncu-bio-total_word_count_in_all_files.json', 'r') as f:
    wordcountdict = json.load(f)

#print(len(wordcountdict))
# collecting terms
terms = list(labelocc_c["term"])

# collecting IDs of all papers, whole subset
IDList = list(labelocc)
IDList.pop(0)

# collecting IDs of all papers, corona subset
IDList_c = list(labelocc_c)
IDList_c.pop(0)


#print(len(IDList_c))
#print(len(IDList))
#print(len(IDrest))

IDrest = []
nomatch = 0
for paper1 in IDList:
    nomatch = 0
    for paper2 in IDList_c:
        if paper1!=paper2:
            nomatch +=1
            if nomatch==len(IDList_c):
                IDrest.append(paper1)
                

def getWordCount(term, IDList, sourcedf) :
    wordcount = 0
    for paper in IDList:
        #print(paper)
        wc_paperID = sourcedf[paper][sourcedf["term"] == term].values[0]
        #print(wc_paperID)
        wordcount = wordcount + wc_paperID
    return int(wordcount)


def getTotWords(term, IDListin, sourcedf, sourcedict=wordcountdict) :
    total_words = 0;
    tokens_of_term = len(term.split())
    #add check if 
    for paper in IDListin :
        #wctot_paperID = sourcedf[sourcedf["term"] == term][paper][0]
        wctot_paperID = sourcedict[paper]
        #print(wctot_paperID)
        total_words = total_words + wctot_paperID
        #if tokens_of_term > 1:
        #    total_words = total_words + int(wctot_paperID)
        #    #compute the total number of tokens considering the overlapping window
        #    total_words = wctot_paperID - tokens_of_term +1
    return total_words


#print(getTotWords("Influenza", IDList_c, labelocc_c))
EFDict = {}; pDict = {}; logEFDict ={};					# Other variables

####### Probability of finding the term corona in entire literature #######

# taken from previous calculations (see above)
corona_count = 11130 # mentions of "corona" or it's synonyms in all biorxiv and non-common-subset articles
corona_tot_wc = getTotWords("Corona", IDList, labelocc) # total word count of biorxiv and non-common-subset articles, that mentioned corona
 
pCorona = corona_count / corona_tot_wc

print ("Prob of finding the term corona in entire literature = " + str(pCorona))

###############################################################################

for term in terms: 
    if True :
        wc1 = getWordCount(term, IDList, labelocc); #print (wc1)
        wc_tot = getTotWords(term, IDList, labelocc);
        pL = wc1 / wc_tot;  # Prob of finding the term across entire Lit
        #print ("Prob of finding the term "+term+" in the entire literature =" + str(pL))
        #print(wc1)
        #print(wc_tot)
        #print(pL)

        wc2 = getWordCount(term, IDList_c, labelocc_c)
        wc_totC = getTotWords(term, IDList_c, labelocc_c)
        pC = wc2 / wc_totC   # Prob of finding the term across CORONA Lit
        #print ("Prob of finding the term "+term+" in the CORONA literature =" + str(pC))
        #print(wc2)
        #print(wc_totC)
        #print(pC)

    ######## Preferential association of vocab terms with CORONA literature #########
        if pL != 0:
            EF = pC/pL
            #print ("CORONA Lit has "+str(EF)+ " fold higher probability of finding the term "+term)
            if EF>0: 
                #print (math.log10(EF))  # as Log10(0) is undefined
                EFDict[term] = EF
                logEFDict[term] = math.log10(EF)
            posteriorT = (pC*pCorona)/pL
            #print(posteriorT)
            #print(pCorona)
            pDict[term] = posteriorT
        else :
            #print ("The term "+term+" does not appear in any literature and hence will be neglected!\n")
            pass


#collecting results and saving to a dataframe
resDf = pd.DataFrame(columns=["term", "EF", "EFlog", "posterior", ])
for term in terms:
    resDf = resDf.append({"term": term, "EF":EFDict[term], "EFlog": logEFDict[term], "posterior":pDict[term]}, ignore_index=True)
resDf = resDf.sort_values("EF", ascending=False).reset_index(drop=True)   
print(resDf)

plt.figure(figsize=(10,20))


ax=sns.barplot(data=resDf, x="EF", y="term")
ax.set_ylabel("")
ax.set_xlabel("Enrichment Factor")
ax.axvline(1, ls='--')
