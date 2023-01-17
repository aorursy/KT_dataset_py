import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
races = pd.read_csv(r"../input/hkracing/races.csv", delimiter=",", header=0, index_col='race_id')
races_data = races[['venue', 'race_no', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'race_class']]
runs = pd.read_csv(r"../input/hkracing/runs.csv", delimiter=",", header=0)
runs_data = runs[['race_id', 'result', 'won', 'horse_age', 'horse_country', 'horse_type', 'horse_rating',
                  'declared_weight', 'actual_weight', 'draw', 'win_odds']] 
data = runs_data.join(races_data, on='race_id')
# drop race_id after join because it's not a feature
data = data.drop(columns=['race_id'])
print(data.head())
# remove rows with NaN
print(data[data.isnull().any(axis=1)])
print('data shape before drop NaN rows', data.shape)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
print('data shape after drop NaN rows', data.shape)

# encode ordinal columns: config, going, horse_ratings
encoder = preprocessing.OrdinalEncoder()
data['config'] = encoder.fit_transform(data['config'].values.reshape(-1, 1))
data['going'] = encoder.fit_transform(data['going'].values.reshape(-1, 1))
data['horse_ratings'] = encoder.fit_transform(data['horse_ratings'].values.reshape(-1, 1))

# encode labels
lb_encoder = preprocessing.LabelEncoder()
data['horse_country'] = lb_encoder.fit_transform(data['horse_country'])
data['horse_type'] = lb_encoder.fit_transform(data['horse_type'])
data['venue'] = lb_encoder.fit_transform(data['venue'])

print(data.dtypes)
print(data.head())
# feature selection
# result and won are outputs, the rest are inputs
X = data.drop(columns=['result', 'won'])
y = data['won']

# apply SelectKBest class to extract top 10 best features
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization 
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['features', 'score']  
print(feature_scores.nlargest(10, 'score')) 

# choose the top 10 features only
X = data[['win_odds', 'draw', 'declared_weight', 'actual_weight', 'horse_rating', 
          'horse_country', 'venue', 'race_no', 'horse_ratings', 'race_class']]
mnb = MultinomialNB()
scores = cross_val_score(mnb, X, y, cv=10, scoring='precision')
average_precision = sum(scores) / len(scores) 
print(f'MultinomialNB average precision: {average_precision}')