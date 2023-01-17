
import os
import json
import pandas as pd

biorxiv_dir = '../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
comm_use_dir = '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'
#non_comm_use_dir = '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'
#custom_license_dir = '../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

data_dirs = [biorxiv_dir,comm_use_dir]#, non_comm_use_dir, custom_license_dir]


def refine_text(json_object):
    # dropping in-text references
    # considering abstract and body_text as of now
    abstract = json_object['abstract']
    body_text = json_object['body_text']

    def get_combined_text(text_dict):
        return " ".join(p['text'] for p in text_dict)
    
    return " ".join([get_combined_text(abstract), get_combined_text(body_text)])

def loadData(directory):
    biorxiv_pdfs = os.listdir(directory)
    print(f'----- Total pdf files in %s {len(biorxiv_pdfs)} -----'%(directory.split('/')[-3]))

    sample_file_path = os.path.join(directory, biorxiv_pdfs[0])
    sample_json = json.load(open(sample_file_path, 'r'))
    print(f'\n** Keys in pdf file - \n{sample_json.keys()}\n**')

    with open('sample.json', 'w') as f:
        json.dump(sample_json, f)

    refined_pdfs = []
    for pdf in biorxiv_pdfs:
        json_pdf = json.load(open(os.path.join(directory, pdf), 'r'))
        json_pdf['text_'] = refine_text(json_pdf)
        refined_pdfs.append(json_pdf)

    dataframe = pd.DataFrame.from_records(refined_pdfs)
    return dataframe
import numpy as np
import io

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    embeddings_index = {}
    f = io.open(gloveFile, 'r', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Done.",len(embeddings_index)," words loaded!")
    return embeddings_index

def loadGloveEmbedding():
    MODEL = loadGloveModel("../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt")
    DEFAULT = np.mean([MODEL[k] for k in MODEL],axis = 0)
    return MODEL,DEFAULT

MODEL,DEFAULT = loadGloveEmbedding()



import numpy as np
import string

def noobEmbedding(sentence, context):
    embedding = np.zeros(26)
    for char in sentence:
        if (char <= 'z' and char >= 'a'):
            embedding[string.ascii_lowercase.index(char)] += 1
        elif (char <= 'Z' and char >= 'A'):
            embedding[string.ascii_uppercase.index(char)] += 1
    return embedding, context

def getEmbedding(sentence, context):
    return gloveEmbedding(sentence, context)

def gloveEmbedding(sentence, context):
    words = sentence.split()
    if words:
        vectors = np.stack([MODEL[k] if k in MODEL else DEFAULT for k in words],axis=1)
        embedding = np.mean(vectors,axis = 1)
    else:
        embedding = DEFAULT
    return embedding, context
    
    
def cosineSimilarity(a, b):
    normAB = np.linalg.norm(a)*np.linalg.norm(b)
    if normAB != 0:
        return np.dot(a, b)/(normAB)
    else:
        return 0

def similarity(a, b):
    return cosineSimilarity(a, b)
def getQueries():
    return ["virus genetics and evolution"]
def rateParagraph(para, embeddingQuery):
    score = 0
    sentences = getSentencesDotSeparated(para)
    context = [0]
    for sentence in sentences:
        embeddingSentence, context = getEmbedding(sentence, context)
        score += similarity(embeddingQuery, embeddingSentence)
    return score/len(sentences)

def score(paper, query):
    result = {}

    context = [0]
    embeddingQuery, _ = getEmbedding(query, context)

    maxScore = 0
    resultAbstract = []
    for idx, para in enumerate(paper['abstract']):
        score = rateParagraph(para['text'], embeddingQuery)
        resultAbstract.append({'idx' : idx, 'score' : score})
        maxScore = max(maxScore, score)
    
    resultBody = []
    for idx, para in enumerate(paper['body_text']):
        score = rateParagraph(para['text'], embeddingQuery)
        resultBody.append({'idx' : idx, 'score' : score})
        maxScore = max(maxScore, score)
    resultAbstract.sort(key = lambda x:x['score'], reverse=True)
    resultBody.sort(key = lambda x:x['score'], reverse=True)

    result['abstract'] = resultAbstract
    result['body'] = resultBody
    result['score'] = maxScore
    return result
PUNCTUATIONS = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
REPLACE = ' '*len(PUNCTUATIONS)

def getSentences(para):
    return para.split("\n")

def getSentencesDotSeparated(para):
    replaceMap = para.maketrans(PUNCTUATIONS,REPLACE)
    para = para.translate(replaceMap)
    return para.lower().split(". ")
data = loadData(data_dirs[0])
for dirs in data_dirs[1:]:
    data = data.append(loadData(dirs),ignore_index = True)
# print(data.iloc[0]['text_'])
print(data.keys())
data.shape
queries = getQueries()
print (queries)
result = []
for idx, query in enumerate(queries):
    result.append([])
    for index, paper in data.iterrows():
        result[idx].append({'idx' : index, 'scores' : score(paper, query)})
        if index%1000 ==0:
            print(index)
# sort result by paper score
[res.sort(key = lambda item : item['scores']['score'], reverse=True) for res in result]
# print(result)
topN = 2
topM = 2

for idx, query in enumerate(queries):
    print("query:", query, "\n")
    for i in range(topN):
        paperScoreData = result[idx][i]
        paper = data.iloc[paperScoreData['idx']]
        scores = paperScoreData['scores']
        print('paper_id : ', paper['paper_id'], "\n")
        print("abstract : ")
        for j in range(topM):
            try:
                print("score :", scores['abstract'][j]['score'])
                print("text : ", paper['abstract'][scores['abstract'][j]['idx']]['text'])
                print()
            except:
                pass
        print("body : ")
        for j in range(topM):
            try:
                print("score :", scores['body'][j]['score'])
                print("text : ", paper['body_text'][scores['abstract'][j]['idx']]['text'])
                print()
            except:
                pass
        print("------------------------------------------")
    print("************************************")
