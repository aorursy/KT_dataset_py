!pip install transformers
!pip install rpforest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean, cosine
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import softmax
import numpy as np

from tqdm import tqdm
import math
from google.colab import drive
drive.mount('/content/drive')
root_dir = '/content/drive/Shared drives/COVID-19-Research/data/'
df = pd.read_csv(f'{root_dir}/metadata.csv', header=0, usecols=['title', 'abstract', 'journal'])
df['title'].isnull().sum()
print(len(df))
df = df[df['title'].notna()]
df = df[df['abstract'].notna()]
print(len(df))
df['title'].fillna('', inplace=True)
# combine title and abstract
df['title+abstract'] = df['title'] + '\n' + df['abstract']
print(len(df))
print(len(df[df['title+abstract']=='']))

df.head()
df.reset_index(drop=True, inplace=True)
df.head()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RobertaModel.from_pretrained('roberta-base').to(device)
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def get_embed_matrix():
  save_path = f'{root_dir}outputs_hong/roberta.npy'
  try:
    return np.load(save_path)
  except:
    print(f'Cannot directly load numpy roberta.npy')
    token_info = tokenizer.batch_encode_plus(df['title+abstract'].tolist(), max_length=300, pad_to_max_length=True, return_attention_masks=True) 

    input_idss = torch.tensor(token_info['input_ids'], dtype=torch.long)
    attention_masks = torch.tensor(token_info['attention_mask'], dtype=torch.long)
    print(f'input tensor size: {input_idss.size()}')

    dataloader = DataLoader(TensorDataset(input_idss, attention_masks), batch_size=32, shuffle=False)
    # run model
    feature_vec = []
    with torch.no_grad():
      for input_ids, attention_mask in tqdm(dataloader):
        feature_vec.append(model(input_ids.to(device), attention_mask=attention_mask.to(device))[0][:,0,:].cpu())
    feature_vec = torch.cat(feature_vec, dim=0)
    print(f'output feature vec size: {feature_vec.size()}')
    np.save(save_path, feature_vec.numpy())
    return feature_vec

embed_matrix = get_embed_matrix()
embed_matrix.shape
# train tf-idf vectorizers and fit documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, stop_words='english', max_features=800)

def get_dfidf_matrices():
  return tfidf_vectorizer.fit_transform(df['title+abstract'])


tfidf_matrix = get_dfidf_matrices()
print(f'tf-idf matrix shape: {tfidf_matrix.shape}')
def query(query, method):

  if method == 'tf-idf':    
    vec = tfidf_vectorizer.transform([query]).toarray()[0] * 100
    matrix = tfidf_matrix.toarray() * 100
  else:
    with torch.no_grad():
      input_ids = torch.tensor(tokenizer.encode(query, add_special_tokens=True)).unsqueeze(0).to(device)
      vec = model(input_ids)[1].squeeze().cpu().numpy() * 100
    matrix = embed_matrix * 100
  
  # print(f'{method} distance: {1 - cosine(vec, matrix[0])}')

  score = np.asarray([1 - cosine(vec, doc_vec) for doc_vec in matrix])
  # print(f'{method}: {score}')
  return score

def score_document(vaccine_queries, therap_queries):
  def compute_other_score(a, b):
    sqrt_func = np.vectorize(math.sqrt)
    return 0.5 * (sqrt_func((1-a) * (1-b)) - sqrt_func((1+a) * (1+b)))
    # return -(a + b)

  vaccine_scoress, therap_scoress = [], []
  from itertools import product
  for method, vaccine_query, therap_query in product(['tf-idf'], vaccine_queries, therap_queries):
    vaccine_scoress.append(query(vaccine_query, method))
    therap_scoress.append(query(therap_query, method))

  final_scores = []
  for vaccine_scores, therap_scores in zip(vaccine_scoress, therap_scoress):
    other_scores = compute_other_score(vaccine_scores, therap_scores)
    # compute softmax scores
    scores = np.stack([other_scores, vaccine_scores, therap_scores], axis=0)
    scores = softmax(scores, axis=0)
    final_scores.append(scores)


  # majority votes
  final_votes = [np.argmax(final_score, axis=0) for final_score in final_scores]

  final_votes = np.stack(final_votes)
  print(f'finalvotes shape: {final_votes.shape}')
  print(final_votes[:,:20])
  from scipy.stats import mode
  final_votes = mode(final_votes, axis=0)[0].squeeze()
  print(f'finalvotes shape: {final_votes.shape}')
  return final_votes
vaccine_queries = ('vaccine vaccination dose antitoxin serum immunization inoculation for COVID-19 or coronavirus related research work',
                  'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients of COVID-19 or coronavirus',
                   'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients',
                   'Exploration of use of best animal models and their predictive value for a human vaccine')
therap_queries = ('Therapeutics treatment therapy drug antidotes cures remedies medication prophylactic restorative panacea for COVID-19 or coronavirus',
                   'Effectiveness of drugs like naproxen, clarithromycin, and minocyclinethat being developed that may exert effects on viral replication and tried to treat COVID-19 patients',
                  'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.',
                  'Effectiveness of drugs being developed and tried to treat COVID-19 patients')
final_votes = score_document(vaccine_queries, therap_queries)

from collections import Counter
print(Counter(final_votes))
final_votes[:20]

# save as json
import json
json_dict = {}
for i, row in df.iterrows():
  json_dict[row['title']] = int(final_votes[i])
with open('/content/drive/Shared drives/COVID-19-Research/results/Hong/votes.json', 'w') as fp:
    json.dump(json_dict, fp)
save_path = '/content/drive/Shared drives/COVID-19-Research/results/Hong/votes'
np.save(save_path, final_votes)
df.iloc[24]['title']
get_score('Immune-mediated diseases of the dog and cat III. Immune-mediated diseases of the integumentary, urogenital, endocrine and vascular systems')
