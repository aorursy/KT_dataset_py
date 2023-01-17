# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
df.head(10)
output=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
res_df1=df.loc[(df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'] == 1)] 
res_df1.loc[:,'label'] = 1
res_df2=df.loc[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)] 
res_df2.loc[:,'label'] = 0
df=res_df1.append(res_df2)
df = df.sample(frac=1).reset_index(drop=True)
df.head(5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'],df['label'], test_size=0.3, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(min_df=2,max_df=0.5,ngram_range=(1,1),max_features=10000,stop_words='english')
features= tfidf.fit_transform(X_train)


features=features.todense()
test_tfidf= tfidf.transform(X_test)
test_tfidf=test_tfidf.todense()
#random
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=15, n_estimators=40)
clf.fit(features, y_train)
test_data=X_test.tolist()
test_data[12445]
import shap
explainer = shap.TreeExplainer(clf)
choosen_instance = test_tfidf[12445]
choosen_instance = np.squeeze(np.asarray(choosen_instance))
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0],choosen_instance,feature_names=tfidf.get_feature_names())
test_data[37]
choosen_instance = test_tfidf[37]
choosen_instance = np.squeeze(np.asarray(choosen_instance))
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1],choosen_instance,feature_names=tfidf.get_feature_names())
test_data[75]
choosen_instance = test_tfidf[75]
choosen_instance = np.squeeze(np.asarray(choosen_instance))
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1],choosen_instance,feature_names=tfidf.get_feature_names())
