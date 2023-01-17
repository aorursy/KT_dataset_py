import sys
!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import f1_score, accuracy_score
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
import cuml; cuml.__version__
