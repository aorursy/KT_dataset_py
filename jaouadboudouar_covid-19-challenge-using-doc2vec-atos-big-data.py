#Install spacy & spacylangdetect to take only english articles 
!pip install spacy
!pip install spacy-langdetect
!mkdir output
import re
from spacy_langdetect import LanguageDetector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json
from nltk.corpus import stopwords
from copy import deepcopy
from langdetect import detect
import numpy as np
import os
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from tqdm.notebook import tqdm
import string  
import gensim
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import NearestNeighbors
import re
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

class CleanAndTokenizeText(BaseEstimator, TransformerMixin):
    
    def tokenizer(self, input_text):
       tokens = re.split('\W+', input_text)
       return tokens

    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans('', '', punct)
        return input_text.translate(trantab)
    
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords_and_non_latin_words(self, words):
        stopwords_list = stopwords.words('english')
        stopwords_list.append('al')
        stopwords_list.append('et')
        stopwords_list.append('also')
        whitelist=[]
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return clean_words
    
    def stemming(self, words):
        porter = PorterStemmer()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def lemma(self,words):
        lemmatizer = WordNetLemmatizer()
        stemmed_words = [lemmatizer.lemmatize(word) for word in words]
        return stemmed_words 
        
    def english_only(self, words):
      english_words = []
      for word in words:
        if detect(word) == 'en':
          english_words.append(word)
      return english_words


def transform(text):
  ct = CleanAndTokenizeText() 
  text_st = str(text)
  clean_x = ct.remove_urls(text_st)
  clean_x = ct.remove_punctuation(clean_x)
  clean_x = ct.remove_digits(clean_x)
  clean_x = ct.to_lower(clean_x)
  clean_x = ct.tokenizer(clean_x)
  clean_x = ct.remove_stopwords_and_non_latin_words(clean_x)
  clean_x = ct.lemma(clean_x)
  return clean_x
#Reading Dataset
df_clean_biorxiv = pd.read_csv('../input/covid19-challenge-dataset/biorxiv_clean.csv')
df_clean_pmc = pd.read_csv('../input/covid19-challenge-dataset/clean_pmc.csv')
df_clean_ncu = pd.read_csv('../input/covid19-challenge-dataset/clean_noncomm_use.csv')
df_clean_cu = pd.read_csv('../input/covid19-challenge-dataset/clean_comm_use.csv')

#Concatenate all datasets in one dataframe
final_frames = [df_clean_biorxiv, df_clean_pmc, df_clean_ncu, df_clean_cu]
df_final = pd.concat(final_frames)

#remove null data from title, texte, abstract columns
df_final['title'] = df_final['title'].fillna('')
df_final['text'] = df_final['text'].fillna('')
df_final['abstract'] = df_final['abstract'].fillna('')


#detect language of articles using title
df_final['lang'] = df_final['title'].apply(lambda title : nlp(title)._.language['language'])
df_final.head(3)
#Clean title, text and abstract 
df_final['title_tokenized'] = df_final['title'].apply(lambda x : transform(x))
df_final['text_tokenized'] = df_final['text'].apply(lambda x : transform(x))
df_final['abstract_tokenized'] = df_final['abstract'].apply(lambda x : transform(x))
#Combine title, text, and abstract
df_final['complete_text_tokenized'] = df_final['title_tokenized'] + df_final['text_tokenized'] + df_final['abstract_tokenized']
#Take only entries with more than 200 keywords
df_final = df_final[df_final['complete_text_tokenized'].map(len) > 200]
#Take only english entries
df_final = df_final[df_final['lang'] == 'en']
#Describing our final dataframe.
df_final['complete_text_tokenized'].describe

#Prepare corpus 
def read_corpus(df, column):
    for i, line in enumerate(df[column]):
        yield gensim.models.doc2vec.TaggedDocument(line, [i])


#Take 100 % of our dataset
train_df  = df_final.sample(frac=1, random_state=42)

#train corpus
train_corpus = (list(read_corpus(train_df, 'complete_text_tokenized'))) 
# Doc2VEC : using distributed memory model
model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=300, min_count=10, epochs=20, seed=42, workers=10)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
#Task 1 
task_1 = "improve clinical processes disease models including animal models varies across age personal protective equipment monitor phenotypic change inform decontamination efforts infected person implementation community settings role movement control strategies community settings effectiveness virus immune response prevent secondary transmission immunity effectiveness control range transmission tools stainless steel reduce risk provide information potential adaptation physical science natural history nasal discharge long individuals health status health care fecal matter environmental survival different materials charge distribution affected areas viral shedding asymptomatic shedding phobic surfaces infection prevention incubation periods environmental stability disease virus transmission shedding surfaces stability infection incubation usefulness urine substrates studies sputum sources seasonality recovery products prevalence plastic persistence multitude learned know hydrophilic humans even environment diagnostics covid coronavirus copper contagious blood adhesion 19"
task_2 = "populations public health mitigation measures fatality among symptomatic hospitalized patients risk patient groups susceptibility potential risks factors smoking existing pulmonary disease co pregnant women socio epidemiological studies data environmental factors severity basic reproductive number 19 risk factors viral infections make existing respiratory behavioral factors including risk serial interval morbidities neonates incubation period transmission dynamics economic impact disease co infections transmission including economic whether virus virulent understand transmissible pre modes high effective differences covid could control"
task_3 = "temporal diverse sample sets sustainable risk reduction strategies behavioral risk factors nagoya protocol could test host range animal interface real understand geographic distribution whether farmers could animal host whether farmers livestock could determine whether whole genomes track variations southeast asia receptor binding rapid dissemination one strain mixed wildlife management measures livestock farms lateral agreements humans socioeconomic genomic differences genetic sequencing experimental infections epidemic appears virus genetics time tracking field surveillance continued spill virus origin geographic virus time surveillance spill origin therapeutics serve sars role reservoir played pathogen multi mechanism leveraged information inform infected human evolution evidence diagnostics development cov coronaviruses coordinating circulation access 2 "
task_4 = "investigate less common viral inhibitors methods evaluating potential complication could include identifying approaches alongside suitable animal models standardize challenge studies efforts develop prophylaxis clinical studies evaluate vaccine immune response develop animal models include antiviral agents healthcare workers approaches best animal models published concerning research may exert effects aid decision makers clinical effectiveness studies expanding production capacity universal coronavirus vaccine newly proven therapeutics viral replication evaluate risk alternative models therapeutics effectiveness production ramps vaccine recipients human vaccine evaluation efforts efforts targeted vaccination assays timely distribution predictive value ensure equitable distribute scarce dependent enhancement bench trials treat covid process development enhanced disease discover therapeutics 19 patients clinical vaccine efforts therapeutics disease discover development covid 19 vaccines use tried therapeutic prioritize populations need naproxen minocyclinethat exploration drugs developed determining conjunction clarithromycin capabilities antibody ade "
task_5 = "vary among different populations models health care delivery system capacity would include identifying policy policy changes necessary public health advice support real time give us time excellence could potentially critical government services health insurance status financial costs may social distancing approaches compare npis currently critical household supplies needed care health diagnoses critical shortfalls predict costs immigration status housing status employment status various sizes take account school closures rapid design rapid assessment qualified participants programmatic alternatives potential interventions pharmaceutical interventions people fail mobilize resources mitigate risks mass gatherings limited resources likely efficacy geographic location geographic areas gain consensus food distribution establish funding economic impact dhs centers coordinated way travel bans consistent guidance supplies social npis guidance bans ways want underserved treatment states spread scale respond research regardless race pay pandemic non methods leveraged lessen infrastructure individuals increase income implemented identified high factors experiments execution examine even equity enhance enable effectiveness disability control conduct comply compliance communities collaboration cases benefits barriers authorities authoritative age access ability "
task_6 = "epidemic preparedness innovations could provide critical funding help quickly migrate assays onto predict clinical outcomes )? make immediate policy recommendations g ., heavily trafficked companion species ), inclusive potential testers using pcr predict severe disease progression targeted surveillance experiments calling states might leverage universities ad hoc local interventions understanding best clinical practice public health surveillance perspective experiments could aid public health officials one health surveillance sufficient viral load streamlined regulatory environment published concerning systematic occupational risk factors improve response times host response markers collecting longitudinal samples determine asymptomatic disease existing surveillance platforms widespread current exposure existing diagnostic platforms rapid influenza test new diagnostic tests assay development issues detect early disease new platforms therapeutic interventions public ). potential sources local expertise best practices side tests ongoing exposure early detection care test diagnostic testing operational issues latency issues rapid design rapid bed unknown pathogens transmission hosts target regions supplies associated specific entity sampling methods rapidly sharing private sector private laboratories particular variant occurring pathogens national guidance mitigation measures market forces large scale holistic approaches holistic approach genetic drift future spillover future pathogens future diseases farmed wildlife extent possible explore capabilities evolutionary hosts enhance capabilities domestic food distinguishing naturally detection schemes defined area critical coupling genomics avoid locking allow specificity advanced analytics accelerator models testing purposes start testing mass testing rapid sequencing including swabs including legal including demographics technology roadmap technology crispr increase capacity enhance capacity specific reagents future coalition environmental sampling surveillance disease understanding states development detect testing including technology capacity sequencing reagents environmental coalition would virus use tradeoffs track terms tap support speed separation screening scaling role report recruitment recognizing protocols policies point people pathogen organism opportunities needed mutations mitigate mechanism like intentional instruments information important impact humans guidelines genome execution evolution ethical employ efforts efficacy effects e diagnostics devices developing denominators demographic data cytokines covid coordination communications biological bioinformatics barriers accuracy accessibility able 19 "
task_7 = "viral etiologies extracorporeal membrane oxygenation organ failure – particularly acute respiratory distress syndrome core clinical outcome set support skilled nursing facilities published concerning alternative methods long term care facilities time health care delivery published concerning surge capacity 19 patients outcomes data published concerning processes surge medical staff personal protective equipment might potentially work mechanical ventilation adjusted infection prevention control public health interventions published concerning efforts adjusted mortality data simple things people high flow oxygen based support resources across state boundaries supply chain management overwhelmed communities age g ., eua best telemedicine practices inform clinical care clinical trials efforts data across clinical outcomes outcomes data nursing homes medical care infected patients enhance capacity best practices clinical characterization trials efforts take care sick people inform allocation hospital flow care level adapt care supportive interventions evaluate interventions workforce protection workforce allocation specific actions scarce resources save thousands risk factors regulatory standards possible cardiomyopathy oral medications natural history n95 masks maximize usability innovative solutions elastomeric respirators done manually determine adjunctive critical challenges crisis standards cardiac arrest address shortages manage disease disease management extrapulmonary manifestations published care 19 outcomes efforts resources management g age disease manifestations without within way virus use transmission technologies steroids remove real range production payment organization mobilization limited knowledge know including improve home guidance frequency facilitating faciitators expand encouraging efficiency ecmo e develop define covid course could community clia barriers ards approaches application ai advise ability "
task_8 = "implementing public health measures affects systematically collect information related develop qualitative assessment frameworks published concerning ethical considerations integrated within multidisciplinary research translate existing ethical principles published concerning social sciences public health measures health seeking behaviors novel ethical issues support sustained education expanded global networks embed ethics across social sciences psychological health social media salient issues ethics efforts underlying drivers thematic areas surgical masks secondary impacts school closures rapid identification providing care operational platforms minimize duplication local barriers immediate needs fuel misinformation capacity building 19 patients oversight efforts 2019 efforts outbreak response measures research existing efforts outbreak use uptake team stigma standards srh rumor responding prevention physical particularly must modification includes identify g fear establish engage enablers e covid coordinate control connect burden articulate arise area anxiety adherence addressed access "
task_9 = "baseline public health response infrastructure preparedness modes sharing response information among planners local public health surveillance systems public health emergency response health care workers ). risk populations ’ families governmental public health public health capability misunderstanding around containment agendas incorporate attention indicates potential risk including academic ). understanding coverage policies clarify community measures data systems coordinate local risk populations information sharing nation ’ disadvantaged populations access information risk communication underrepresented minorities sectoral collaboration research priorities reach marginalized population groups mitigating threats mitigate gaps include targeting incarcerated people equity considerations data standards coordinating data assuring access action plan public target high standardized nomenclature mitigating barriers capacity relevant surveillance sharing ). information care understanding measures nomenclature high communication capacity barriers value understand treatment testing supported support state related recruit published providers profit problems private prevention others opportunities non needs need mitigation methods know investments inter integration inequity guidelines gathering funding follow federal expertise ensure elderly easy disease diagnosis covid communicating commercial citizens circumstances 19 "

list_of_tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9]

def get_doc_vector(doc):
    tokens = transform(doc) 
    vector = model.infer_vector(tokens)
    return vector

array_of_tasks = [get_doc_vector(task) for task in list_of_tasks]

train_df['complete_text_vector'] = [vec for vec in model.docvecs.vectors_docs]
train_array = train_df['complete_text_vector'].values.tolist()

#Apply KNN to extract 50 neighbors
ball_tree = NearestNeighbors(algorithm='ball_tree', leaf_size=20).fit(train_array)
distances, indices = ball_tree.kneighbors(array_of_tasks, n_neighbors=50)

df_output = pd.DataFrame(columns=['Task','Result_Paper_ID','complete_text_tokenized'])

for i, info in enumerate(list_of_tasks):
    df =  train_df.iloc[indices[i]]
    dist = distances[i]
    papers_ids = df['paper_id']
    titles = df['title']
    complete_texts_tokenized = df['complete_text_tokenized']
    for l in range(len(dist)):
        df_output = df_output.append({'Task': i, 'Result_Paper_ID' : papers_ids.iloc[l], 'complete_text_tokenized' : complete_texts_tokenized.iloc[l]}, ignore_index=True)
df_output.to_csv('./output/final_output.csv', sep=',', encoding='utf-8')

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from ast import literal_eval
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
df_output_results = pd.read_csv('../input/covid19-challenge-dataset/final_output.csv')
df_output_results['complete_text_tokenized'].describe()
%matplotlib inline 
stopwords = set(STOPWORDS)
new_stopwords = ['copyright', 'dq', 'license', 'display', 'author', 'preprint', 'patient', 'authorfunder','ef','using', 'new', 'set', 'yet', 'fully', 'expected', 'medrxiv', 'available', 'granted','futhermore']
new_stopwords_list = stopwords.union(new_stopwords)
lem = WordNetLemmatizer()
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=new_stopwords_list,
        max_words=200,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)
  
    plt.imshow(wordcloud)
    plt.show()
df_task1 = df_output_results['complete_text_tokenized'][0:10] # What is known about transmission, incubation, and environmental stability?
df_task2 = df_output_results['complete_text_tokenized'][50:60] # What do we know about COVID-19 risk factors?
df_task3 = df_output_results['complete_text_tokenized'][100:110] # What do we know about virus genetics, origin, and evolution?
df_task4 = df_output_results['complete_text_tokenized'][150:160] # What do we know about vaccines and therapeutics?
df_task5 = df_output_results['complete_text_tokenized'][200:210] # What has been published about medical care?
df_task6 = df_output_results['complete_text_tokenized'][250:260] # What do we know about non-pharmaceutical interventions?
df_task7 = df_output_results['complete_text_tokenized'][300:310] # What do we know about diagnostics and surveillance?
df_task8 = df_output_results['complete_text_tokenized'][350:360] # What has been published about ethical and social science considerations?
df_task9 = df_output_results['complete_text_tokenized'][400:410] # What has been published about information sharing and inter-sectoral collaboration?
liste1 = []
for el in df_task1 : 
  liste_of_keywords = set(literal_eval(el))
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste1.append(lem.lemmatize(el2))

wordsT1 = ' '.join(liste1)

liste2 = []
for el in df_task2 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste2.append(lem.lemmatize(el2))

wordsT2 = ' '.join(liste2)

liste3 = []
for el in df_task3 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste3.append(lem.lemmatize(el2))

wordsT3 = ' '.join(liste3)

liste4 = []
for el in df_task4 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste4.append(lem.lemmatize(el2))

wordsT4 = ' '.join(liste4)

liste5 = []
for el in df_task5 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste5.append(lem.lemmatize(el2))

wordsT5 = ' '.join(liste5)

liste6 = []
for el in df_task6 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste6.append(lem.lemmatize(el2))

wordsT6 = ' '.join(liste6)

liste7 = []
for el in df_task7 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste7.append(lem.lemmatize(el2))

wordsT7 = ' '.join(liste7)

liste8 = []
for el in df_task8 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste8.append(lem.lemmatize(el2))

wordsT8 = ' '.join(liste8)

liste9 = []
for el in df_task9 : 
  liste_of_keywords = literal_eval(el)
  for el2 in liste_of_keywords:
    if(nlp(el2)._.language['language'] == 'en'):
      liste9.append(lem.lemmatize(el2))

wordsT9 = ' '.join(liste9)
show_wordcloud(wordsT1, title = 'Task : What is known about transmission, incubation, and environmental stability? - wordcloud (10 samples)')
show_wordcloud(wordsT2, title = 'Task : What do we know about COVID-19 risk factors? - wordcloud (10 samples)')
show_wordcloud(wordsT3, title = 'Task : What do we know about virus genetics, origin, and evolution? - wordcloud (10 samples)')
show_wordcloud(wordsT4, title = 'Task : What do we know about vaccines and therapeutics? - wordcloud (10 samples)')
show_wordcloud(wordsT5, title = 'Task : What has been published about medical care? - wordcloud (10 samples)')
show_wordcloud(wordsT6, title = 'Task : What do we know about non-pharmaceutical interventions? - wordcloud (10 samples)')
show_wordcloud(wordsT7, title = 'Task : What do we know about diagnostics and surveillance? - wordcloud (10 samples)')
show_wordcloud(wordsT8, title = 'Task : What has been published about ethical and social science considerations? - wordcloud (10 samples)')
show_wordcloud(wordsT9, title = 'Task : What has been published about information sharing and inter-sectoral collaboration? - wordcloud (10 samples)')
