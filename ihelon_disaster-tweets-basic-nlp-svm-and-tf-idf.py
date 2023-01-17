import numpy as np 

import pandas as pd 

import os



from sklearn import model_selection as sk_model_selection

from sklearn.feature_extraction import text as sk_fe_text

from sklearn import svm as sk_svm

from sklearn import metrics as sk_metrics
base_dir = '../input/nlp-getting-started/'

df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))

df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))

df_submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
print(f'df_train shape: {df_train.shape}')

df_train.head()
df_train.isna().sum()
X_train = df_train["text"]

y_train = df_train["target"].values
tfidf = sk_fe_text.TfidfVectorizer(stop_words = 'english')

tfidf.fit(X_train)

X_train = tfidf.transform(X_train)
parameters = { 

    'C': [0.01, 0.1, 1],

    'gamma': [0.7, 1, 'auto', 'scale']

}



model = sk_svm.SVC(

    kernel='rbf', 

    class_weight='balanced',

    random_state=42,

)



model = sk_model_selection.GridSearchCV(

    model, 

    parameters, 

    cv=5,

    scoring='f1',

    n_jobs=-1,

)



model.fit(X_train, y_train)



print(f'Best parameters: {model.best_params_}')

print(f'Mean cross-validated F1 score of the best_estimator: {model.best_score_:.3f}')
X_test = df_test["text"]

X_test = tfidf.transform(X_test)

y_test_pred = model.predict(X_test)
df_submission["target"] = y_test_pred

df_submission.to_csv("submission.csv",index=False)
df_submission