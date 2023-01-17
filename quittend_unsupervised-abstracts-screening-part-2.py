import pandas as pd

import numpy as np



from sklearn.mixture import GaussianMixture

from sklearn.model_selection import RandomizedSearchCV, KFold

from scipy.stats import loguniform, uniform, randint



RANDOM_STATE = 1563
df = pd.read_csv('/kaggle/input/cleaning-cord-19-metadata/cord_metadata_cleaned.csv')

print(f'There are {len(df)} studies.')
embeddings = np.load('/kaggle/input/biowordvec-precomputed-cord19/biowordvec.npy')

print(f'Embedding matrix has shape: {embeddings.shape}')
estimator = GaussianMixture(

    n_components=10,

    covariance_type='full', 

    max_iter=100, 

    n_init=1, 

    init_params='kmeans', 

    random_state=RANDOM_STATE, 

)
N_ITER = 20

N_SPLITS = 4



param_distributions = {

    "n_components": randint(2, 256),

    "covariance_type": ['diag', 'full', 'spherical'],

}



cv = KFold(

    n_splits=N_SPLITS, 

    shuffle=True, 

    random_state=RANDOM_STATE

)



hp_search = RandomizedSearchCV(

    estimator=estimator,

    param_distributions=param_distributions,

    n_iter=N_ITER,

    n_jobs=N_SPLITS,

    cv=cv,

    verbose=1,

    random_state=RANDOM_STATE,

    return_train_score=True,

    refit=True

)



hp_search.fit(embeddings)

best_model = hp_search.best_estimator_
print(f'Best validation likelihood: {hp_search.best_score_}')
print(f'Best params: {hp_search.best_params_}')
df['cluster'] = best_model.predict(embeddings)
cluster_count = df['cluster'].value_counts().sort_values()



ax = cluster_count.plot(kind='bar', figsize=(15, 5))

ax.set_xticks([])

ax.set_xlabel("Cluster id")

ax.set_ylabel("Count")

ax.grid(True)
(df

    .drop(columns=['title_lang', 'abstract_lang', 'distance'])

    .to_csv('/kaggle/working/cord_metadata_word2vec.csv', index=False)

)