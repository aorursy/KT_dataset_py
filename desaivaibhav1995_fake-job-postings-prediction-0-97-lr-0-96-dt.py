import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
data.shape
data.isnull().sum()
cols = ["title", "company_profile", "description", "requirements", "benefits"]
for c in cols:
    data[c] = data[c].fillna("")

def extract_features(df):    
    for c in cols:
        data[c+"_len"] = data[c].apply(lambda x : len(str(x)))
        data[c+"_wc"] = data[c].apply(lambda x : len(str(x.split())))

    
extract_features(data)
data.head()
data['combined_text'] = data['company_profile'] + " " + data['description'] + " " + data['requirements'] + " " + data['benefits']

n_features = {
    "title" : 100,
    "combined_text" : 500
}

for c, n in n_features.items():
    tfidf = TfidfVectorizer(max_features=n, norm='l2', stop_words = 'english')
    tfidf.fit(data[c])
    tfidf_train = np.array(tfidf.transform(data[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c]):
        data[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
cat_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]

for c in cat_cols:
    encoded = pd.get_dummies(data[c])
    data = pd.concat([data, encoded], axis=1)
drop_cols = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits', 'combined_text']
drop_cols += cat_cols
data = data.drop(drop_cols, axis = 1)
train = data.loc[:9000,:]
test = data.loc[9000:,:]
y = train['fraudulent']
X = train.drop(columns = ['fraudulent'])
y_test = test['fraudulent']
X_test = test.drop(columns = ['fraudulent'])
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)
model_lr = lr.fit(X,y)
pred_lr = model_lr.predict(X_test)
print("Accuracy score (LR) : {:.2f}".format(accuracy_score(pred_lr, y_test)))
print(classification_report(pred_lr,y_test))
SVC = svm.SVC()
model_SVC = SVC.fit(X,y)
SVC_pred = model_SVC.predict(X_test)
print("Accuracy score (SVM) : {:.2f}".format(accuracy_score(SVC_pred, y_test)))
