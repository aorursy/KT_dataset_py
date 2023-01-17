import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from scipy import stats

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.csv')
df_predict = pd.read_csv('../input/test.csv')
# Generating additional new features and cleaning text with RegEx

def add_features(df):
    # new column for exclamation mark
    df['ex_mark'] = df['comment_text'].str.findall('\!+')
    df['ex_mark'] = df['ex_mark'].apply(lambda x: len(x))
    df['ex_mark'][ df['ex_mark']>  df['ex_mark'].quantile(.9)] = df['ex_mark'].quantile(.9) #remove outsiders
    
    # new column for question mark
    df['qu_mark'] = df['comment_text'].str.findall('\?+')
    df['qu_mark'] = df['qu_mark'].apply(lambda x: len(x))
    df['qu_mark'][ df['qu_mark']>  df['qu_mark'].quantile(.9)] = df['qu_mark'].quantile(.9) #remove outsiders
    
    # new column for *
    df['star_mark'] = df['comment_text'].str.findall('\*+')
    df['star_mark'] = df['star_mark'].apply(lambda x: len(x))

    # new columns for smileys
    smileys_good = r'((:|;|X)-?(\)|P|D))\W'
    smileys_bad =  r'((:|;)-?(\())\W'
    df['smileys_good'] = df['comment_text'].str.extract(smileys_good, expand=True)[0].fillna(0)
    df['smileys_bad'] = df['comment_text'].str.extract(smileys_bad, expand=True)[0].fillna(0)

    df['smileys_good'][df['smileys_good']!=0] = 1
    df['smileys_bad'][df['smileys_bad']!=0] = 1
    
    # new column link_count
    df['link_count'] = df['comment_text'].str.findall(r'\wwww\.')
    df['link_count'] = df['link_count'].apply(lambda x: len(x))
    
    # new column quote_count
    df['quote_count'] = df['comment_text'].str.findall(r'(\'+|\"+)')
    df['quote_count'] = df['quote_count'].apply(lambda x: len(x))
    df['quote_count'][ df['quote_count']>  df['quote_count'].mean()*2] = df['quote_count'].mean()*2
    
    # new column comma_count
    df['comma_count'] = df['comment_text'].str.findall(r'\,+')
    df['comma_count'] = df['comma_count'].apply(lambda x: len(x))
    df['comma_count'][ df['comma_count']>  df['comma_count'].mean()*2] = df['comma_count'].mean()*2
    
    
    # cleaning text
    df['comment_text'] = df['comment_text'].str.replace(r'a*h+a+h+a+', 'haha')
    df['comment_text'] = df['comment_text'].str.replace(r'a+hh+', 'ahh')
    df['comment_text'] = df['comment_text'].str.replace(r'(l+o+l+\s?)+', 'lol')
    df['comment_text'] = df['comment_text'].str.replace(r'a+b+c\w*', 'abc')
    df['comment_text'] = df['comment_text'].str.replace(r'a+r+g+h+', 'argh')
    df['comment_text'] = df['comment_text'].str.replace(r'a+w+e+s+o+m+e+', 'awesome')
    df['comment_text'] = df['comment_text'].str.replace(r'\ba*f+u+c*k*\b', 'fuck')
    df['comment_text'] = df['comment_text'].str.replace(r'aa+ww+', 'aww')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e*a+y+', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e+a+h+', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e{2,}s{2,}', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'ass', 'azz')
    
    # replace char repetitions
    df['comment_text'] = df['comment_text'].str.replace(r'(.)\1+', r"\1")
        
    return df
    
df_train = add_features(df_train)
df_predict = add_features(df_predict)

df_train.describe()
# Fit the vectorizer to whole data set
all_text = pd.concat([df_train['comment_text'], df_predict['comment_text']])
# Here comparing binarized scores with Tfidf scores
bin_vect = TfidfVectorizer(min_df=4, ngram_range=(1,2), stop_words='english', lowercase=True, binary=True).fit(all_text)
vect = TfidfVectorizer(min_df=4, ngram_range=(1,2), stop_words='english', lowercase=True, binary=False).fit(all_text)
# Split data into X and Y
X_train = df_train['comment_text']
X_predict = df_predict['comment_text']
Y = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# transform the documents in the training data to a sparse matrix
X_train_vectorized = vect.transform(X_train)
X_train_bin_vectorized = bin_vect.transform(X_train)
X_predict_vectorized = vect.transform(X_predict)
# plot influence of Chi2 dimensionality reduction
def plot_model():
    lr = LogisticRegression()
    percentiles = (60, 70, 80, 90)
    best_scores = []
    
    for y_col in Y.columns:
        score_maxs, score_means, score_mins = [], [], []
        bin_score_maxs, bin_score_means, bin_score_mins = [], [], []

        for percentile in percentiles:
            # fit chi2-Filter to binary values and apply filter to both X-sets
            chi2_filter = SelectPercentile(chi2, percentile)
            X_train_bin_vectorized_new = chi2_filter.fit_transform(X_train_bin_vectorized, Y[y_col])
            X_train_vectorized_new = chi2_filter.transform(X_train_vectorized)
            # calculate cross-validation scores for filtered sets
            bin_scores = cross_val_score(lr, X_train_bin_vectorized_new, Y[y_col], n_jobs=1)
            scores = cross_val_score(lr, X_train_vectorized_new, Y[y_col], n_jobs=1)
            bin_score_maxs.append(bin_scores.max())
            bin_score_means.append(bin_scores.mean())
            bin_score_mins.append(bin_scores.min())
            score_maxs.append(scores.max())
            score_means.append(scores.mean())
            score_mins.append(scores.min())
        #plot results 
        fig, ax = plt.subplots()
        ax.plot(percentiles, bin_score_means, label='binary')
        ax.fill_between(percentiles, bin_score_maxs, bin_score_mins, alpha=.5)
        ax.plot(percentiles, score_means, label='cont.')
        ax.fill_between(percentiles, score_maxs, score_mins, alpha=.5)
        plt.legend()
        plt.title(y_col)
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')
        plt.show()

plot_model()
        
# fit and apply Chi²-Filter
chi2_filter = SelectPercentile(chi2, 65)
X_train_filtered = chi2_filter.fit_transform(X_train_vectorized, Y)
X_predict_filtered = chi2_filter.transform(X_predict_vectorized)

# add additional features features

def plot_model2():
    add_features = ['ex_mark', 'smileys_good', 'smileys_bad', 'star_mark',
                        'qu_mark', 'comma_count', 'quote_count']

    for y_col in Y.columns:
        scores_mean, scores, scores_diff = [], [], []
        score0 = np.mean(cross_val_score(LogisticRegression(), X_train_filtered, Y[y_col], n_jobs=1))
        for feature in add_features:
            train_features =  hstack([X_train_filtered, np.array(df_train[feature].astype('int64'))[:,None]])
            cv_score = cross_val_score(LogisticRegression(), train_features, Y[y_col], n_jobs=1)
            scores_mean.append(np.mean(cv_score))
            scores_diff.append((np.max(cv_score)-np.min(cv_score))/2)
            scores.append(cv_score)

        fig, ax = plt.subplots()
        ax.bar(add_features, scores_mean, alpha=0.8, yerr=scores_diff)
        ax.plot(add_features, np.full([len(add_features)], score0), 'k')
        plt.ylim(score0-np.abs(score0-np.min(scores))*1.2, score0+np.abs(score0-np.max(scores))*1.5)
        plt.xticks(rotation='vertical')
        plt.title(y_col)
        plt.xlabel('Percentile')
        plt.ylabel('feature')
        plt.show()
    
plot_model2()

# adding ex_mark as additional feature
train_features =  hstack([X_train_filtered, np.array(df_train['ex_mark'].astype('int64'))[:,None]])
predict_features =  hstack([X_predict_filtered, np.array(df_predict['ex_mark'].astype('int64'))[:,None]])
#predict Y
#additional tuning of hyperparameters with GridsearchCV

model = LogisticRegression()
params = {'C':[1]}

Y_predicted = pd.DataFrame()
Y_predicted['id'] = df_predict['id']
scores = []

for y_col in Y.columns:
    gsCV = GridSearchCV(model, params, scoring="roc_auc").fit(train_features, Y[y_col])
    scoreX = np.max(gsCV.cv_results_['mean_test_score'])
    scores.append(scoreX)
    Y_predicted[y_col] = gsCV.predict_proba(predict_features)[:,1]
    print(y_col + ':' + str(scoreX))
print('mean score: ' + str(np.mean(scores)))
submission = Y_predicted
submission.to_csv('submission.csv', index=False)