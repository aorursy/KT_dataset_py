import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Read input files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_resource = pd.read_csv("../input/resources.csv")
#Join the duplicate rows in resources CSV and get the sum of quantity and price
train_resource_grouped = train_resource.groupby(['id'], as_index=False)[['quantity','price']].sum()
test_resource_grouped = train_resource.groupby(['id'], as_index=False)[['quantity','price']].sum()
#Join the train and test resources to respective dataframes
train_joined = pd.merge(train,train_resource_grouped,on='id')
test_joined = pd.merge(test,test_resource_grouped,on='id')
#since we dont have values for project_essay 3&4 after some date I'm merging essays 1&3 and 2&4 to avoid NaN values
train_joined['projectessay_1_3'] = train_joined['project_essay_1']+train_joined['project_essay_3'].replace(np.nan, '', regex=True)
train_joined['projectessay_2_4'] = train_joined['project_essay_2']+train_joined['project_essay_4'].replace(np.nan, '', regex=True)
test_joined['projectessay_1_3'] = test_joined['project_essay_1']+test_joined['project_essay_3'].replace(np.nan, '', regex=True)
test_joined['projectessay_2_4'] = test_joined['project_essay_2']+test_joined['project_essay_4'].replace(np.nan, '', regex=True)
#Remove unwanted columns
columns_to_remove = ["id","project_essay_1","project_essay_2","project_essay_3","project_essay_4"]
train_joined.drop(columns_to_remove,inplace=True,axis=1)
test_joined.drop(columns_to_remove,inplace=True,axis=1)
categorical_columns = ['teacher_prefix','school_state', 'project_grade_category','project_subject_categories', 'project_subject_subcategories']
non_cat_columns = ["project_submitted_datetime","teacher_number_of_previously_posted_projects","quantity","price"]
text_columns = ["project_title","project_resource_summary","projectessay_1_3","projectessay_2_4"]
train_cat = train_joined[categorical_columns]
train_non_cat = train_joined[non_cat_columns]
train_text = train_joined[text_columns]
test_cat = test_joined[categorical_columns]
test_non_cat = test_joined[non_cat_columns]
test_text = test_joined[text_columns]
y = train_joined['project_is_approved']
train_cat = pd.get_dummies(train_cat)
test_cat = pd.get_dummies(test_cat)
### Handling missing Categories since some categories are missing in test
test_columns = test_cat.columns 
train_columns = train_cat.columns
all_columns = train_columns.union(test_columns)
train_add_columns = all_columns.difference(train_columns)
test_add_columns = all_columns.difference(test_columns)
test_copy = test_cat
test_cat = pd.concat([test_copy , test_copy.reindex(columns = test_add_columns, fill_value = 0.0)], axis = 1)
train_copy = train_cat
train_cat = pd.concat([train_copy , train_copy.reindex(columns = train_add_columns, fill_value = 0.0)], axis = 1)
from datetime import datetime
#Convert column to datetime column
train_non_cat["project_submitted_datetime"] = pd.to_datetime(train_non_cat["project_submitted_datetime"])
test_non_cat["project_submitted_datetime"] = pd.to_datetime(test_non_cat["project_submitted_datetime"])
#Get month year from the datetome column
train_non_cat["Project_submitted_month"] = train_non_cat["project_submitted_datetime"].map(lambda x: x.month)
train_non_cat["Project_submitted_year"] = train_non_cat["project_submitted_datetime"].map(lambda x: x.year)
test_non_cat["Project_submitted_month"] = test_non_cat["project_submitted_datetime"].map(lambda x: x.month)
test_non_cat["Project_submitted_year"] = test_non_cat["project_submitted_datetime"].map(lambda x: x.year)
train_non_cat.drop(["project_submitted_datetime"],inplace=True,axis=1)
test_non_cat.drop(["project_submitted_datetime"],inplace=True,axis=1)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
train_non_cat_scaled = pd.DataFrame(std.fit_transform(train_non_cat),columns=train_non_cat.columns)
test_non_cat_scaled = pd.DataFrame(std.fit_transform(test_non_cat),columns=test_non_cat.columns)

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    lowercase=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
#Vectorize text columns
vectorized_columns = []
vectorized_columns_test = []
for text_column in text_columns:
    word_vectorizer.fit(pd.concat([train_text[text_column],test_text[text_column]]))
    vectorized_columns.append(word_vectorizer.transform(train_text[text_column]))
    vectorized_columns_test.append(word_vectorizer.transform(test_text[text_column]))
train_features = hstack(vectorized_columns+[csr_matrix(train_non_cat_scaled)]+[csr_matrix(train_cat)], 'csr')
test_features = hstack(vectorized_columns_test+[csr_matrix(test_non_cat_scaled)]+[csr_matrix(test_cat)], 'csr')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(n_jobs=-1)
model.fit(train_features,y)
predicted_proba = model.predict_proba(test_features)
submission = pd.DataFrame.from_dict({'id': test['id']})
submission['project_is_approved'] = predicted_proba[:,1]
submission.to_csv("submission.csv",index=False)
