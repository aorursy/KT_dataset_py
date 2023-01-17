# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import numpy as np
import random
from collections import defaultdict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
class PredictorBase:
    """
    This class contains implementation of experiment design.
    It passes few instances with classes to the `train` method, then it passes validation instances to predict
    method.
    """
    
    def __init__(self):
        pass
    
    def clear(self):
        """
        Clear info about previous bathces
        """
        pass
    
    def train(self, X, y):
        """
        Train on few entities. Override this method in real implementation
        """
        pass
    
    def predict(self, X):
        """
        Predict classes of the given entities
        """
        pass
class Evaluator:
    """
    This class may be used to test any solution with following experiment design:
    
    * Read the next N records from dataset (N = 1000)
    * Group records by concept
    * Split each group to Train and Test, so there will be at least one train and one test record for each group.
    * Use all train records to 'learn' model. Note that 'learning' should be fast on this stage.
    * Predict concepts for all test records and calculate FScore
    """
    # Default batch size
    batch_size = 1000
    test_fraction = 0.3
    
    @classmethod
    def read_lines(cls, fd):
        for line in fd:
            if not isinstance(line, str):
                line = line.decode()
            yield line.strip('\n').split('\t')
            
    @classmethod
    def prepare_data(cls, group):
        """
        This method converts concept groups into feature and target tables.
        """
        X, y = [], []
        for label, entities in group.items():
            for entity in entities:
                X.append((entity[1], entity[3])) # Use right and left contexts only
                y.append(label)
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)
        return X, y
    
    
    def __init__(self, filename="./shuffled_dedup_entities.tsv.gz"):
        self.fd = open(filename, 'r')
        self.reader = self.read_lines(self.fd)
        
        
    def read_batch(self, size=None):
        """
        This methos reads lines from dataset and create train and test groups
        """
        batch = list(itertools.islice(self.reader, size or self.batch_size))
        
        groups = defaultdict(list)
        for entity in batch:
            groups[entity[0]].append(entity)
            
        train_groups = {}
        test_groups = {}
        for etype, entities in groups.items():
            if len(entities) * self.test_fraction > 1:
                test_size = int(len(entities) * self.test_fraction)
                test_groups[etype] = entities[:test_size]
                train_groups[etype] = entities[test_size:]
        
        return train_groups, test_groups
    

    def eval_batched(self, model, metric, entities_count, count):
        """
        This method evaluates given model and calculates given metric for model predictions.
        """
        metrics = []
        for batch_id in range(count):
            train, test = eva.read_batch(entities_count)
            X, y = Evaluator.prepare_data(train)
            X_test, y_test = Evaluator.prepare_data(test)
            model.train(X, y)
            pred = model.predict(X_test)
            score = metric(pred, y_test)
            metrics.append(score)
        return np.mean(metrics)
    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
def concat_context(X):
    return np.array(list(map(
        lambda x: x[0] + " " + x[1],
        X
    )))

class KNNBaseline(PredictorBase):
    """
    This is an implementation of simple kNN based solution.
    It uses cosine distance between Bag of Words in contexts to predict mentioned concept
    """
    def __init__(self):
        self.model = None
        self.clear()
    
    def clear(self):
        self.model = Pipeline([
            ('concat_context', FunctionTransformer(concat_context)),
            ('vectorizer', CountVectorizer(stop_words='english')),
            ('cls', KNeighborsClassifier(metric='cosine', algorithm='brute'))
        ])
        
    def train(self, X, y):
        self.model.fit(X, y)
        
        
    def predict(self, X):
        return self.model.predict(X)
        
eva = Evaluator('../input/shuffled_dedup_entities.tsv')
mean_score = eva.eval_batched(
    model=KNNBaseline(), # used model
    metric=lambda x, y: f1_score(x, y, average='micro'), # metric function
    entities_count=1000, # number of mentions per batch
    count=50 # number of batches to evaluate on
)
mean_score
