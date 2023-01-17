import json



import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



sns.set_palette('husl')

sns.set_style("whitegrid")



from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, PowerTransformer

from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV



from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
! ls ../input/yelp-dataset
reviews = []



with open('../input/yelp-dataset/yelp_academic_dataset_review.json') as file:

    for line in file:

        reviews.append(json.loads(line))

        

review_df = pd.DataFrame(reviews)

review_df.head()