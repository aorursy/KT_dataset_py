import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from pathlib import Path
import random
from random import choice

random.seed(2020)
import os
os.listdir('../input/')
class VoteEnsemble:
  def __init__(self):
    self.root_dir = Path('../input/')
    self.jsons = []
    self.majority_vote = None

  def _load_json(self):
    if self.jsons:
      return

    with open(self.root_dir / 'results/Hong/votes.json') as f:
      self.jsons.append([json.load(f)])
    with open(self.root_dir / 'results/Mihir/ensemble_results.json') as f:
      self.jsons.append([json.load(f)])
    with open(self.root_dir / 'results/Jitesh/elasticbert_labels.json') as f1:
      with open(self.root_dir / 'results/Jitesh/google_labels.json') as f2:
        self.jsons.append([json.load(f1), json.load(f2)])
    with open(self.root_dir / 'results/Rishab/bert_1.json') as f1:
      with open(self.root_dir / 'results/Rishab/biobert_1.json') as f2:
        with open(self.root_dir / 'results/Rishab/fine_tuned_1.json') as f3:
          self.jsons.append([json.load(f1), json.load(f2), json.load(f3)])
    # load Ashwin's results
    label_map = {'O':0, 'VC':1, 'TR':2}
    with open(self.root_dir / 'results/Ashwin/JSON_Files_For_Ensemble/labels/COVID_Labels_After_Classification_CH_labels.json') as f1:
      json1 = json.load(f1)
      json1 = {key: label_map[value] for key, value in json1.items()}
      with open(self.root_dir / 'results/Ashwin/JSON_Files_For_Ensemble/labels/Filter_After_FineTuning_labels.json') as f2:
        json2 = json.load(f2)
        json2 = {key: label_map[value] for key, value in json2.items()}
        with open(self.root_dir / 'results/Ashwin/JSON_Files_For_Ensemble/labels/Filter_Before_FineTuning_labels.json') as f3:
          json3 = json.load(f3)
          json3 = {key: label_map[value] for key, value in json3.items()}       
          self.jsons.append([json1, json2, json3]) 
  
  def _get_majority_vote(self, votes):
    """
    Args: 
      votes: a list of dictionary {'title': class}, where class is one of {0, 1, 2},
             0 for not related, 1 for vaccine related, 2 for therapeutics related
    Return:
      vote: a single dictionary {'title': class}
    """
    all_titles = set()
    for v in votes:
      all_titles |= set(v.keys())

    # combining votes
    vote = {title: Counter([v[title] for v in votes if title in v]) for title in all_titles}
    # pick highest vote:
    result = {}
    for title, counter in vote.items():
      freq = counter.most_common(1)[0][1]
      candidate_cls = [cls for cls, num_votes in counter.items() if num_votes == freq]
      result[title] = choice(candidate_cls)

    return result

  def _combine_individual_votes(self):
    if not self.jsons:
      self._load_json()

    for i, votes in enumerate(self.jsons):
      if len(votes) == 1:
        self.jsons[i] = votes[0]
      else:
        self.jsons[i] = self._get_majority_vote(votes)

  def majority_vote_ensemble(self):
    if self.majority_vote is not None:
      return self.majority_vote

    self._combine_individual_votes()
    self.majority_vote = self._get_majority_vote(self.jsons)
    label_map = {0: 'Other', 1: 'Vaccine', 2: 'Therapeutics'}
    self.majority_vote = {key: label_map[value] for key, value in self.majority_vote.items()}
    return self.majority_vote


# get statistics
ensemble = VoteEnsemble()
Counter(ensemble.majority_vote_ensemble().values())

ensemble.majority_vote
class ScoreEnsemble:
  def __init__(self):
    self.df = None
    self.weights = None
    self.scores = None
    self.combined_scores = None
    self.sorted_vaccine_titles = None
    self.sorted_theurap_titles = None
    self.root_path = '' ## need specify root_path

  def _load_metadata_df(self, df_path): ## need specify df_path
    if self.df:
      return self.df
    
    self.df = pd.read_csv(f'{self.root_path}/df_path')
    return self.df

  def _load_scores_from_json(self):
    """ Load scores that are stored in json files """
    if self.scores:
      return self.scores

    file_names = ['1.json', '2.json', '3.json', '4.json', '5.json']
    scores = []
    for file_name in file_names:
      scores.append(json.load(f'{self.root_path}/file_name'))
    self.scores = scores
    return self.scores

  def _train_weights(self): # need implementation if want to train weights
    if self.weights:
      return self.weights

    self.weight = [1] * 5
    return self.weights

  def _aggregate_scores(self):
    if self.combined_scores:
      return self.combined_scores

    weights = self._train_weights()
    scores = self._load_scores_from_json()
    # get all keys/titles
    all_titles = set()
    for score in self.scores:
      all_titles |= set(score.keys())
    # combine scores
    combined_scores = defaultdict(float)
    for title in titles:
      for weight, score in zip(weights, scores):
        score4title = score.get(title)
        score4title = np.asarray(score4title) if score4title else np.asarray((1/3, 1/3, 1/3))
        combined_scores[title] += weight * score4title
    self.combined_scores =  combined_scores
    return self.combined_scores
    
  def _sort_by_score(self):
    if self.sorted_vaccine_titles and self.sorted_theurap_titles:
      return self.sorted_vaccine_titles, self.sorted_theurap_titles
      
    combined_scores = _aggregate_scores()
    self.sorted_vaccine_titles = [k for k, v in sorted(combined_scores.items(), reverse=True, key=lambda item: item[1][0])]
    self.sorted_theurap_titles = [k for k, v in sorted(combined_scores.items(), reverse=True, key=lambda item: item[1][1])]
    return self.sorted_vaccine_titles, self.sorted_theurap_titles
  
  def get_top_K(self, cls='vaccine', K=100):
    sorted_vaccine_titles, sorted_theurap_titles = _sort_by_score()
    if cls == 'vaccine':
      top_k_titles = set(sorted_vaccine_titles[:K])
    elif cls == 'therap' or cls == 'therapeutics':
      top_k_titles = set(sorted_theurap_titles[:K])
    else:
      raise NotImplementedError(f'Not type call \"{cls}\", please set cls to \"vaccine\" or \"therap\"')

    df = self._load_metadata_df()
    return df.loc[df['title'].isin(top_k_titles)]

  def get_score_given_title(self, title):
    combined_scores = _aggregate_scores()
    score = combined_scores[title]
    print(f'vaccines: {score[0]}, therapeutics: {score[1]}, other: {score[2]}')
    return score