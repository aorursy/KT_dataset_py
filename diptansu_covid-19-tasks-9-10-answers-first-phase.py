import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
meta_df.info()
# load the nltk

from nltk.tokenize.punkt import PunktSentenceTokenizer

data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
sentTokenizer = PunktSentenceTokenizer()
print(sentTokenizer.span_tokenize(data))
# helper functions

def mergeConsecutiveSpans(spans):
    newSpans = []
    for ((start, _), (_, end)) in zip(spans, spans[1:]):
        newSpans.append((start, end))
    return newSpans

mergeConsecutiveSpans([(1,2), (3, 5), (4, 7)])
def findSentence(text, sentence_spans, ref):
    if 'text' in ref:
        ref_info = ref['text']
    elif 'mention' in ref:
        ref_info = ref['mention']
    else:
        ref_info = f"Unknown info from {ref}"
    start = ref['start']
    end = ref['end']
    sentence = ""
    for (sent_start, sent_end) in sentence_spans:
        if sent_start <= start and sent_end >= end:
            sentence = text[sent_start:sent_end]
            break
    if len(sentence) == 0:
        for (sent_start, sent_end) in mergeConsecutiveSpans(sentence_spans):
            if sent_start <= start and sent_end >= end:
                sentence = text[sent_start:sent_end]
                break
        
    if len(sentence) == 0:
        sentence = f"Couldn't find ({start}, {end}) in sentences:\n"
        for (sent_start, sent_end) in sentence_spans:
            sentence += f"\t({sent_start}, {sent_end}): {text[sent_start:sent_end]}\n"
    return sentence
    

with open('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/42a44518a00b962207226bf61b4d71a0f596e2a1.json') as file:
    sentTokenizer = PunktSentenceTokenizer()
    finfo = json.load(file)
    body_text = finfo['body_text']
    #print(body_text)
    for idx, item in enumerate(body_text):
        text = item['text']
        sentence_spans = sentTokenizer.span_tokenize(text)
        #print(f"sentence spans = {sentence_spans}")
        for ref in item['ref_spans']:
            sentence = findSentence(text, sentence_spans, ref)
            print(f"In {idx}th reference text: {sentence} ref_id: {ref['ref_id']}")
        for cite in item['cite_spans']:
            sentence = findSentence(text, sentence_spans, cite)
            print(f"In {idx}th citation: {sentence} cite_id: {cite['ref_id']}")
def getCitationPoints(sentTokenizer, finfo):
    body_text = finfo['body_text']
    #print(body_text)
    citations = dict()
    for idx, item in enumerate(body_text):
        text = item['text']
        sentence_spans = sentTokenizer.span_tokenize(text)
        #print(f"sentence spans = {sentence_spans}")
        for cite in item['cite_spans']:
            ref_id = cite['ref_id']
            if ref_id is None:
                continue
            sentence = findSentence(text, sentence_spans, cite)
            #print(f"In {idx}th citation: {sentence} cite_id: {cite['ref_id']}")
            sentences = citations.get(cite['ref_id'], list())
            sentences.append(sentence)
            citations[cite['ref_id']] = sentences
    return citations
                
with open('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/42a44518a00b962207226bf61b4d71a0f596e2a1.json') as file:
    sentTokenizer = PunktSentenceTokenizer()
    finfo = json.load(file)
    print(len(finfo['bib_entries']))
    print(getCitationPoints(sentTokenizer, finfo))
from tqdm.notebook import tqdm
meta_title = meta_df.set_index('title')

# add new column
meta_title['json_present'] = False

# drop na
meta_title = meta_title.loc[meta_title.index.dropna()]

# remove duplicates
meta_title = meta_title[~meta_title.index.duplicated(keep='first')]
meta_title.index.is_unique

# sort index
meta_title = meta_title.sort_index()
def checkBibs(fname, meta_title, titles_from_json, bibs, paper_sentences, duplicate_jsons):
    with open(fname) as file:
        sentTokenizer = PunktSentenceTokenizer()
        finfo = json.load(file)
        title = finfo['metadata']['title']
        if title not in meta_title.index:
            #print(f"{title} not found in {finfo['metadata']}, paper_id = {finfo['paper_id']}")
            return
        meta_title.loc[title, "json_present"] = True
        bib_entries = finfo.get('bib_entries', None)
        cit_points = getCitationPoints(sentTokenizer, finfo)
        # some papers dont have citation points, so for now we are going to not use those
        #if (bib_entries is not None and len(bib_entries) > 0 ) and len(cit_points) == 0:
        #    print(f"no citation_points found for {fname}")
        if bib_entries is None and len(cit_points) > 0:
            print(f"no bib_entries found for {fname}")
            return
        if title in titles_from_json:
            #print(f"{title} already exists!")
            duplicate_jsons.add(title)
        titles_from_json.add(title)
        bibs[title] = bib_entries
        paper_sentences[title] = cit_points
        
bibs = dict()
titles_from_json = set()
paper_sentences = dict()
duplicate_jsons = set()
checkBibs('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/42a44518a00b962207226bf61b4d71a0f596e2a1.json', 
          meta_title, titles_from_json, bibs, paper_sentences, duplicate_jsons)
print(titles_from_json)
print(bibs)
print(duplicate_jsons)
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
# test with a small number first
for fname in tqdm(all_json[:4]):
    checkBibs(fname, meta_title, titles_from_json, bibs, paper_sentences, duplicate_jsons)
print(titles_from_json)
print(bibs)
print(duplicate_jsons)
# now for all the files
bibs = dict()
titles_from_json = set()
paper_sentences = dict()
duplicate_jsons = set()
for fname in tqdm(all_json):
    checkBibs(fname, meta_title, titles_from_json, bibs, paper_sentences, duplicate_jsons)
print(f"# of unique titles: {len(titles_from_json)}")
print(f"# of unique bib entries {len(bibs)}")
print(f"# of duplicate entries {len(duplicate_jsons)}")
# filter out the metadata for files not present
meta_title = meta_title[meta_title['json_present'] == True]
import itertools
ref_graph = list()
for title in tqdm(paper_sentences):
    refs_to_sentences = paper_sentences[title]
    bibs_to_titles = bibs[title]
    for ref in refs_to_sentences:
        sentences = refs_to_sentences[ref]
        if ref in bibs_to_titles:
            totitle = bibs_to_titles[ref]['title']
            #import pdb; pdb.set_trace()
            if totitle in meta_title.index:
                for s in sentences:
                    ref_graph.append([title, totitle, s])
    #break
ref_graph = pd.DataFrame(ref_graph, columns=['fromtitle', 'totitle', 'ref_sentence'])
ref_graph
len(ref_graph['totitle'].unique())
len(ref_graph['fromtitle'].unique())
grouped_ref_sentences = ref_graph.pivot_table(values='ref_sentence', columns=['totitle'], aggfunc='. '.join).transpose()
grouped_ref_sentences
ref_graph.memory_usage()
paper_sentences[list(paper_sentences.keys())[1]]
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import spacy
import scispacy
from tqdm import tqdm

nlp = spacy.load("en_core_sci_lg")
# sub questions from the 2 tasks. 

questions = [
    # What has been published about ethical and social science considerations?
    "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019",
    "Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",
    "Efforts to support sustained education, access, and capacity building in the area of ethics",
    "Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",
    "Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",
    "Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",
    "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.",
    # What has been published about information sharing and inter-sectoral collaboration?
    "Methods for coordinating data-gathering with standardized nomenclature."
    "Sharing response information among planners, providers, and others.",
    "Understanding and mitigating barriers to information-sharing.",
    "How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",
    "Integration of federal/state/local public health surveillance systems.",
    "Value of investments in baseline public health response infrastructure preparedness",
    "Modes of communicating with target high-risk populations (elderly, health care workers).",
    "Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).",
    "Communication that indicates potential risk of disease to all population groups.",
    "Misunderstanding around containment and mitigation.",
    "Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",
    "Measures to reach marginalized and disadvantaged populations.",
    "Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",
    "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",
    "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care",
]
qvectors = list()
for q in questions:
    qvectors.append(nlp(q).vector)
qvectors = np.array(qvectors)
qvectors.shape
def cosine_similarity(u, v):
    """
    Arguments:
        u -- a word vector of shape (m,n)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0

    #import pdb; pdb.set_trace()
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v.T)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u, axis=1))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity
from tqdm.notebook import tqdm
question_answers = None
for idx in tqdm(grouped_ref_sentences.index):
    text = grouped_ref_sentences.loc[idx, 'ref_sentence']
    text = text[:1000000]
    textvector = nlp(text).vector
    similarities = cosine_similarity(qvectors, textvector).reshape(1, -1)
    #import pdb; pdb.set_trace()
    if question_answers is None:
        question_answers = similarities
    else:
        question_answers = np.append(question_answers, similarities, axis=0)
question_answers.shape
question_matches = pd.DataFrame(question_answers, columns=["q" + str(i) for i, _ in enumerate(questions)])
# find top 5 matches for each question
question_matches.idxmax()
top5 = pd.DataFrame(dict([(q, grouped_ref_sentences.iloc[question_matches.sort_values('q' + str(i), 
                                                                             ascending=False).head(5).index].index.values) for i, q in enumerate(questions)])).T
top5
