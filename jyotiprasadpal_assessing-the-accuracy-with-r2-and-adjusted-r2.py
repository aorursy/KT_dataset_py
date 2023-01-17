import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, train_test_split
df = pd.read_csv('../input/ISLR-Auto/Credit.csv')

df.head()
df.drop(columns='Unnamed: 0', inplace=True, axis=1)
df.info()
df.isna().sum()
sns.pairplot(df)
fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax)
X = df.loc[:, 'Income':'Ethnicity']

y = df.loc[:, 'Balance']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
NUM_FEATURES = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']

CAT_FEATURES = ['Gender', 'Student', 'Married', 'Ethnicity']



num_pipe = Pipeline(steps=[

    ('scale', StandardScaler()),   

])

cat_pipe = Pipeline(steps=[    

    ('encode', OneHotEncoder(drop='first')),   

    ('scale', StandardScaler(with_mean=False)),

])



preprocessor = ColumnTransformer(transformers=[

    ('num', num_pipe, NUM_FEATURES),

    ('cat', cat_pipe, CAT_FEATURES),

], remainder='drop')
r2scores=[]

adjustedr2 = []

feature_names=[]

for i in range(1, 10):   

    reduce_dim_pipe = Pipeline(steps=[

        ('preprocess', preprocessor),

        ('reduce_dim', SelectKBest(k=i, score_func=f_regression)),       

    ])

    

    pipeline = Pipeline(steps=[

        ('reduce_dim_pipe', reduce_dim_pipe),       

        ('regress', LinearRegression())

    ])

    

    #calculate cross validated R2

    R2 = cross_val_score(pipeline, X=X_train, y=y_train,cv=10, scoring='r2').mean()    

    r2scores.append(R2)

        

    #calculate Adj R2

    n= len(X_train)

    p = i #len(X.columns)

    adj_R2 = 1- ((1-R2) * (n-1)/(n-p-1)) #Adj R2 = 1-(1-R2)*(n-1)/(n-p-1)

#     print(r2, adjustedr2)

    adjustedr2.append(adj_R2)

    

    reduce_dim_pipe.fit(X=X_train, y=y_train)

    # Get columns to keep    

    cols = reduce_dim_pipe.named_steps['reduce_dim'].get_support(indices=True)

    # Create new dataframe with only desired columns

#     print(cols)

    features_df_new = X_train.iloc[:, cols]

    best_features = list(features_df_new.columns)

#     print(best_features)

    feature_names.append(best_features)
scoring_df = pd.DataFrame(np.column_stack((r2scores, adjustedr2)), columns=['R2', 'Adj_R2'])

scoring_df['feature_names'] = feature_names

scoring_df['features'] = range(1, 10)

scoring_df
fig, ax = plt.subplots(figsize=(8, 6))

#convert data frame from wide format to long format so that we can pass into seaborn line plot function to draw multiple line plots in same figure

# https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn

long_format_df = pd.melt(scoring_df.loc[:, ['features','R2', 'Adj_R2']], ['features'])

sns.lineplot(x='features', y='value', hue='variable', data=long_format_df, ax=ax)

ax.set_xlabel('No of features')

ax.set_ylabel('Cross validated R2 and Adj R2 scores')

ax.set_title('Plot between number of features and R2/Adj R2 scores')