import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
%matplotlib inline
sns.set()
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_roc_curve
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
tf_df = pd.read_csv('../input/lish-moa/train_features.csv',index_col='sig_id')
pca = PCA(n_components=50)
pca.fit(tf_df.loc[:,'g-0':'c-99'])
pca.explained_variance_ratio_
reduced_comps = pca.transform(tf_df.loc[:,'g-0':'c-99'])
reduced_comps.shape
reduced_comps = reduced_comps[:,:6]
reduced_comps.shape
tts_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv',index_col='sig_id')
moa1 = tts_df.iloc[:,0]
print(moa1.head())
tts_df.columns
len(moa1[moa1==1])
logit_model = LogisticRegression(penalty='l1',solver='saga',max_iter=5000)
logit_model.fit(reduced_comps,moa1)
logit_model.score(reduced_comps,moa1)
metrics.log_loss(moa1,logit_model.predict(reduced_comps))
ax = plt.gca()
plot_roc_curve(logit_model,reduced_comps,moa1,ax=ax)
plt.show()
multilogistic_model = MultiOutputClassifier(LogisticRegression(penalty='l1',solver='saga',max_iter=500))
multilogistic_model.fit(reduced_comps,tts_df)
tts_pred = multilogistic_model.predict(reduced_comps)
print(multilogistic_model.score(reduced_comps,tts_df))
metrics.log_loss(tts_df,tts_pred)
metrics.log_loss(moa1,tts_pred[:,0])
pred_df = pd.DataFrame(tts_pred,index=tts_df.index,columns=tts_df.columns)
pred_df.to_csv('/kaggle/working/submission.csv')