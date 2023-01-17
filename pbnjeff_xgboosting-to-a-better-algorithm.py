# data storage and viz
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from textblob import TextBlob

# model imports
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# for diagnostics
nrows = None

data = pd.read_csv('../input/train.csv', nrows=nrows).drop('teacher_id', axis=1)
resource_data = pd.read_csv('../input/resources.csv')

y = data['project_is_approved']
X = data.drop('project_is_approved', axis=1)
resource_data = pd.read_csv('../input/resources.csv')
resource_data['total_price'] = resource_data['quantity'] * resource_data['price']
resource_data['description'] = resource_data['description'] + ' '
resource_data = pd.DataFrame(resource_data.groupby('id').apply(lambda x: x.sum()))
resource_data = resource_data.reset_index()
resource_data = resource_data.rename(columns={'level_1':'attribute', 0:'value'})
resource_data = resource_data.pivot(index='id', columns='attribute', values='value').drop(columns=['id','price']).reset_index()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, stemmer, stop_words, max_features):
        super(StemmedTfidfVectorizer, self).__init__(stop_words=stop_words, max_features=max_features)
        self.stemmer = stemmer
        
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc:(self.stemmer.stem(w) for w in analyzer(doc))
def join_resource_features(df, resource_data):
    joined_df = df.merge(resource_data, on='id', how='left')
    return joined_df


def combine_essays(row):
    """Cleans up the project essays by combining 4 essays into 2.
    
    See competition description for details.
    """
    if row['project_submitted_datetime'] < '2016-05-17':
        row['project_essay_1'] = row['project_essay_1'] \
                                        + ' ' + row['project_essay_2']
        row['project_essay_2'] = row['project_essay_3'] \
                                        + ' ' + row['project_essay_4']
    return row

def preprocess_text(X_train, X_test):
    """Cleans and processes the text data for BOTH train and test sets.
    
    This combines essays 1/2 & 3/4 then fits the vectorizer to
    the training data.
    """
    
    X_train.apply(lambda row: combine_essays(row), axis=1)
    X_test.apply(lambda row: combine_essays(row), axis=1)
    
    X_train['description'] = X_train['description'].fillna(' ')
    X_test['description'] = X_test['description'].fillna(' ')
    
    X_train['description'] = X_train['description'].astype(str)
    X_test['description'] = X_test['description'].astype(str)
    
    X_train = X_train.drop(columns=['project_essay_3','project_essay_4'], axis=1)
    X_test = X_test.drop(columns=['project_essay_3','project_essay_4'], axis=1)
    
    stemmer = EnglishStemmer()
    
    count_vec_title = CountVectorizer(stop_words='english', max_features=50)
    count_vec_title.fit(X_train['project_title'])
    tfidf_vec_title = StemmedTfidfVectorizer(stemmer=stemmer, stop_words='english', max_features=500)
    tfidf_vec_title.fit(X_train['project_title'])
    
    count_vec_essay1 = CountVectorizer(stop_words='english', max_features=50)
    count_vec_essay1.fit(X_train['project_essay_1'])
    tfidf_vec_essay1 = StemmedTfidfVectorizer(stemmer=stemmer, stop_words='english', max_features=500)
    tfidf_vec_essay1.fit(X_train['project_essay_1'])
    
    count_vec_essay2 = CountVectorizer(stop_words='english', max_features=50)
    count_vec_essay2.fit(X_train['project_essay_2'])
    tfidf_vec_essay2 = StemmedTfidfVectorizer(stemmer=stemmer, stop_words='english', max_features=500)
    tfidf_vec_essay2.fit(X_train['project_essay_2'])
    
    count_vec_resource = CountVectorizer(stop_words='english', max_features=50)
    count_vec_resource.fit(X_train['project_resource_summary'])
    tfidf_vec_resource = StemmedTfidfVectorizer(stemmer=stemmer, stop_words='english', max_features=500)
    tfidf_vec_resource.fit(X_train['project_resource_summary'])
    
    count_vec_resource_desc = CountVectorizer(stop_words='english', max_features=50)
    count_vec_resource_desc.fit(X_train['description'])
    tfidf_vec_resource_desc = StemmedTfidfVectorizer(stemmer=stemmer, stop_words='english', max_features=500)
    tfidf_vec_resource_desc.fit(X_train['description'])
    
    return (X_train, X_test,
            count_vec_title, tfidf_vec_title,
            count_vec_essay1, tfidf_vec_essay1,
            count_vec_essay2, tfidf_vec_essay2,
            count_vec_resource, tfidf_vec_resource,
            count_vec_resource_desc, tfidf_vec_resource_desc)


def process_features(df,
                    count_vec_title, tfidf_vec_title,
                    count_vec_essay1, tfidf_vec_essay1,
                    count_vec_essay2, tfidf_vec_essay2,
                    count_vec_resource, tfidf_vec_resource,
                    count_vec_resource_desc, tfidf_vec_resource_desc):
    """Takes features and changes them into model-usable format.
    
        Returns a numpy array.
    """
    
    dropped_columns = ['project_submitted_datetime','project_subject_categories','project_subject_subcategories',
                      'project_title','project_essay_1','project_essay_2','project_resource_summary','description']
    
    ids = df['id']
    df = df.drop('id', axis=1)
    
    df['quantity'] = df['quantity'].astype(float)
    df['total_price'] = df['total_price'].astype(float)
    
    df = pd.get_dummies(df, columns=['teacher_prefix'])
    df = pd.get_dummies(df, columns=['school_state'])
    df = pd.get_dummies(df, columns=['project_grade_category'])
    
    # Subjects and subject subcategories taken from donorschoose.org
    
    subjects = ['Applied Learning', 'Health & Sports', 'History & Civics', 'Literacy & Language', 'Math & Science', 'Music & The Arts', 'Special Needs', 'Warmth, Care & Hunger']
    
    subject_subcategories = { 'Applied Learning': ['Character Education', 'College & Career Prep', 'Community Service', 'Early Development', 'Extracurricular', 'Other', 'Parent Involvement'],
                            'Health & Sports': ['Gym & Fitness', 'Health & Wellness', 'Nutrition Education', 'Team Sports'],
                            'History & Civics': ['Civics & Government', 'Economics', 'Financial Literacy', 'History & Geography', 'Social Sciences'],
                            'Literacy & Language': ['ESL', 'Foreign Languages', 'Literacy', 'Literature & Writing'],
                            'Math & Science': ['Applied Sciences', 'Environmental Science', 'Health & Life Science', 'Mathematics'],
                            'Music & The Arts': ['Music', 'Performing Arts', 'Visual Arts'],
                            'Special Needs': [],
                            'Warmth, Care & Hunger': [] }
    
    for subject in subjects:
        df['Subject: ' + subject] = df['project_subject_categories'].str.contains(subject).astype(int)
        for subcategory in subject_subcategories[subject]:
            df['Subcategory: ' + subcategory] = df['project_subject_subcategories'].str.contains(subcategory).astype(int)

    df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime'])
    df['submit_year'] = df['project_submitted_datetime'].dt.year
    df['submit_month'] = df['project_submitted_datetime'].dt.month
    df['submit_day'] = df['project_submitted_datetime'].dt.day
    df['submit_dayofweek'] = df['project_submitted_datetime'].dt.weekday_name
    df['submit_hour'] = df['project_submitted_datetime'].dt.hour
    
    df = pd.get_dummies(df, columns=['submit_dayofweek'])
    
    pattern = r'\\([a-zA-Z]|")'
    df['project_title'] = df['project_title'].str.replace(pattern,'')
    
    df['title_polarity'] = df['project_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['project1_polarity'] = df['project_essay_1'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['project2_polarity'] = df['project_essay_2'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['resource_polarity'] = df['project_resource_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    title_dtm = count_vec_title.transform(df['project_title'])
    project1_dtm = count_vec_essay1.transform(df['project_essay_1'])
    project2_dtm = count_vec_essay2.transform(df['project_essay_2'])
    resource_dtm = count_vec_resource.transform(df['project_resource_summary'])
    resource_desc_dtm = count_vec_resource_desc.transform(df['description'])
    
    title_tfidf_dtm = tfidf_vec_title.transform(df['project_title'])
    project1_tfidf_dtm = tfidf_vec_essay1.transform(df['project_essay_1'])
    project2_tfidf_dtm = tfidf_vec_essay2.transform(df['project_essay_2'])
    resource_tfidf_dtm = tfidf_vec_resource.transform(df['project_resource_summary'])
    resource_desc_tfidf_dtm = tfidf_vec_resource_desc.transform(df['description'])
    
    df['project_title'] = df['project_title'].apply(str.lower)
    df['project_title'] = df['project_title'].apply(word_tokenize)
    
    df['has_exclamation'] = df['project_title'].apply(lambda x: '!' in x).astype(int)
    df['title_token_count'] = df['project_title'].apply(len)
    
    df = df.drop(dropped_columns, axis=1)
    
    #processed_array = hstack( (title_dtm, project1_dtm, project2_dtm,
    #                           resource_dtm, title_tfidf_dtm,
    #                           project1_tfidf_dtm, project2_tfidf_dtm,
    #                           resource_tfidf_dtm, resource_desc_dtm, resource_desc_tfidf_dtm, np.array(df)[:,:]) )
    
    processed_array = hstack( (title_tfidf_dtm,
                               project1_tfidf_dtm, project2_tfidf_dtm,
                               resource_tfidf_dtm, resource_desc_dtm, resource_desc_tfidf_dtm, np.array(df)[:,:]) )
    
    return (ids, processed_array)
X_train = X
X_test = pd.read_csv('../input/test.csv', nrows=nrows).drop('teacher_id', axis=1)
X_train = join_resource_features(X_train, resource_data)
X_test = join_resource_features(X_test, resource_data)

(X_train_preprocessed, X_test_preprocessed,
count_vec_title, tfidf_vec_title,
count_vec_essay1, tfidf_vec_essay1,
count_vec_essay2, tfidf_vec_essay2,
count_vec_resource, tfidf_vec_resource,
count_vec_resource_desc, tfidf_vec_resource_desc) = preprocess_text(X_train, X_test)

(X_train_ids, X_train) = process_features(X_train_preprocessed,
                                        count_vec_title, tfidf_vec_title,
                                        count_vec_essay1, tfidf_vec_essay1,
                                        count_vec_essay2, tfidf_vec_essay2,
                                        count_vec_resource, tfidf_vec_resource,
                                        count_vec_resource_desc, tfidf_vec_resource_desc)

(X_test_ids, X_test) = process_features(X_test_preprocessed,
                                        count_vec_title, tfidf_vec_title,
                                        count_vec_essay1, tfidf_vec_essay1,
                                        count_vec_essay2, tfidf_vec_essay2,
                                        count_vec_resource, tfidf_vec_resource,
                                        count_vec_resource_desc, tfidf_vec_resource_desc)
#xgb_model = XGBClassifier(max_depth=2, learning_rate=0.20, objective='binary:logistic')
#parameters = {
#                'num_estimators': range(650)
#             }
#clf = GridSearchCV(xgb_model, parameters, verbose=10, cv=6, scoring='roc_auc')
#clf.fit(X_train, y)
#print(clf.best_estimator_)
xgb_model = XGBClassifier(max_depth=2, n_estimators=1000, learning_rate=0.20, objective='binary:logistic')
xgb_model.fit(X_train, y)
predictions = xgb_model.predict_proba(X_test)
output_df = pd.DataFrame(columns=['id','project_is_approved'])
output_df['id'] = X_test_ids
output_df['project_is_approved'] = predictions[:,1]
output_df.to_csv('submission.csv',index=False)

