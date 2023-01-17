import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.multioutput import MultiOutputClassifier
comp_path = '../input/lish-moa/'
tf_df = pd.read_csv(comp_path+'train_features.csv',index_col='sig_id')
pca = PCA(n_components=50)

pca.fit(tf_df.loc[:,'g-0':'c-99'])

pca.explained_variance_ratio_
reduced_comps = pca.transform(tf_df.loc[:,'g-0':'c-99'])
reduced_comps = reduced_comps[:,:6]
tts_df = pd.read_csv(comp_path+'train_targets_scored.csv',index_col='sig_id')
tts_df.columns
nb_model = MultiOutputClassifier(GaussianNB())

nb_model.fit(reduced_comps,tts_df)

tts_pred = nb_model.predict(reduced_comps)

metrics.log_loss(tts_df,tts_pred)
(pd.DataFrame(tts_pred,columns=tts_df.columns,index=tts_df.index)).to_csv(path_or_buf=

            '/kaggle/working/submission.csv')