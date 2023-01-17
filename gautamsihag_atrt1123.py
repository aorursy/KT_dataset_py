import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import math
damping = 0.85
    
def update(scope, scheduler):
    pvertex = scope.getVertex().value
    #sumval = pvertex.rank * pvertex.selfedge + sum([e.value * scope.getNeighbor(e.from).value.rank for e in scope.getInboundEdges()])
    newval = (1-damping)/scope.getNumOfVertices() + damping*sumval
    if (abs(newval-pvertex.rank)>0.00001):
        scheduler.addTaskToOutbound(scope)
        pvertex.rank = newval
    
    update(scope, scheduler)
