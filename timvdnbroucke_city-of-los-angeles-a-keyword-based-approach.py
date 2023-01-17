import numpy as np

import pandas as pd

import re

import os

import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

py.init_notebook_mode(connected=True)

import copy

from scipy import sparse

from itertools import combinations

from warnings import warn



datadir=r"../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA"

keywords_file = r"../input/top-500-resume-keywords/resume_keywords_clean.txt"

# import textstat

files = [dir for dir in os.walk(datadir)]

bulletins = os.listdir(datadir + "/Job Bulletins/")

additional = os.listdir(datadir + "/Additional data/")

bulletins = os.listdir(datadir + "/Job Bulletins/")

additional = os.listdir(datadir + "/Additional data/")



csvfiles = []

for file in additional:

    if file.endswith('.csv'):

        csvfiles.append(datadir + "/Additional data/" + file)

csvfiles = []

for file in additional:

    if file.endswith('.csv'):

        csvfiles.append(datadir + "/Additional data/" + file)



job_title = pd.read_csv(csvfiles[0])

sample_job = pd.read_csv(csvfiles[1])

kaggle_data = pd.read_csv(csvfiles[2])

job_title = pd.read_csv(csvfiles[0])

sample_job = pd.read_csv(csvfiles[1])

kaggle_data = pd.read_csv(csvfiles[2])

job_title.head()

print("The are %d rows and %d cols in job_title file" % (job_title.shape))

print("The are %d rows and %d cols in sample_job file" % (sample_job.shape))

print("The are %d rows and %d cols in kaggle_data file" % (kaggle_data.shape))

print("There are %d text files in bulletin directory" % len(bulletins))



def get_headings(bulletin):

    with open(datadir + "/Job Bulletins/" + bulletins[bulletin]) as f:  ##reading text files

        data = f.read().replace('\t', '').split('\n')

        data = [head for head in data if head.isupper()]

        return data



def clean_text(bulletin):

    with open(datadir + "/Job Bulletins/" + bulletins[bulletin]) as f:

        data = f.read().replace('\t', '').replace('\n', '')

        return data



get_headings(1)

get_headings(2)



def to_dataframe(num, df):

    opendate = re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')  # match open date

    salary = re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')  # match salary

    requirements = re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')  # match requirements



    for no in range(0, num):

        with open(datadir + "/Job Bulletins/" + bulletins[no],

                  encoding="ISO-8859-1") as f:  # reading files

            try:

                file = f.read().replace('\t', '')

                data = file.replace('\n', '')

                headings = [heading for heading in file.split('\n') if heading.isupper()]  ##getting heading from job bulletin



                sal = re.search(salary, data)

                date = datetime.strptime(re.search(opendate, data).group(3), '%m-%d-%y')

                try:

                    req = re.search(requirements, data).group(2)

                except Exception as e:

                    req = re.search('(.*)NOTES?', re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',

                                                             data)[0][1][:1200]).group(1)



                duties = re.search(r'(DUTIES)(.*)(REQ[A-Z])', data).group(2)

                try:

                    enddate = re.search(

                        r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})'

                        , data).group()

                except Exception as e:

                    enddate = np.nan



                selection = [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)', data)]  ##match selection criteria



                df = df.append({'File Name': bulletins[no], 'Position': headings[0].lower(), 'salary_start': sal.group(1),

                                'salary_end': sal.group(5), "opendate": date, "requirements": req, 'duties': duties,

                                'deadline': enddate, 'selection': selection}, ignore_index=True)



                reg = re.compile(

                    r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\s(years?)\s(of\sfull(-|\s)time)')

                df['EXPERIENCE_LENGTH'] = df['requirements'].apply(

                    lambda x: re.search(reg, x).group(1) if re.search(reg, x) is not None else np.nan)

                df['FULL_TIME_PART_TIME'] = df['EXPERIENCE_LENGTH'].apply(lambda x: 'FULL_TIME' if x is not np.nan else np.nan)



                reg = re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\s|-)(years?)\s(college)')

                df['EDUCATION_YEARS'] = df['requirements'].apply(

                    lambda x: re.search(reg, x).group(1) if re.search(reg, x) is not None else np.nan)

                df['SCHOOL_TYPE'] = df['EDUCATION_YEARS'].apply(lambda x: 'College or University' if x is not np.nan else np.nan)



            except Exception as e:

                ''

                #print('Failed to read file ' + bulletins[no])



    return df



df = pd.DataFrame(

    columns=['File Name', 'Position', 'salary_start', 'salary_end', 'opendate', 'requirements', 'duties', 'deadline'])

df = to_dataframe(len(bulletins), df)

df.replace(to_replace=[None], value="N/A", inplace=True)
field="duties"

df_field=df[field].to_frame()



resume_keywords = list(pd.read_csv(keywords_file, header=None)[0].values)

vocab_keywords = dict(zip(resume_keywords, np.arange(len(resume_keywords))))



#tf-idf: count word frequencies

tfidf = TfidfVectorizer(vocabulary=vocab_keywords,ngram_range=[1,4])



# Apply fit_transform to document: csr_mat

csr_mat = tfidf.fit_transform(df_field[field])

words = tfidf.get_feature_names() #These are the words which the TfidfVectorizer detected, in our case this is simply the vocabulary we provided.
job_nr=1 #An example job description
#Alternative but faster approach similar to NMF

QQ=csr_mat.dot(csr_mat.transpose()).todense()

recommendations_job_nr=np.flip(np.argsort(np.asarray(QQ)[job_nr,:]))



print("Original job description: " + df.loc[job_nr,"Position"] + " ($" + df.loc[job_nr,"salary_start"] + " - " + df.loc[job_nr,"salary_end"] + ")")

print(df.loc[job_nr,field])

print("")



for iRecommendation in range(5):

    print("Recommendation #" + str(iRecommendation+1) + ": " + df.loc[recommendations_job_nr[iRecommendation+1],"Position"] + " ($" + df.loc[recommendations_job_nr[iRecommendation+1],"salary_start"] + " - " + df.loc[recommendations_job_nr[iRecommendation+1],"salary_end"] + ")")  #"+1" removes the original job description itself

    print(df.loc[recommendations_job_nr[iRecommendation+1],field])

    print("")





df.assign(Recommendation_1="").assign(Recommendation_2="").assign(Recommendation_3="").assign(Recommendation_4="").assign(Recommendation_5="")

for job_nr in range(len(df)):

    recommendations_job_nr=np.flip(np.argsort(np.asarray(QQ)[job_nr,:]))

    for iRecommendation in range(5):

        df.loc[job_nr,"Recommendation_" + str(iRecommendation+1)]=df.loc[recommendations_job_nr[iRecommendation+1],"Position"]
occurrences_per_keyword = np.squeeze(np.asarray(sum(csr_mat.todense()!=0)))

data = [go.Histogram(x=occurrences_per_keyword, xbins=dict(start=0,end=140,size= 1),opacity=0.75)]

layout = go.Layout(title='Number of occurrences per keyword',

    xaxis=dict(title='Number of occurrences in different vacancies'),yaxis=dict(title='Number of keywords'),bargap=0.05)

py.iplot(go.Figure(data=data, layout=layout))
data = go.Scatter(x = df.index.values,y = occurrences_per_keyword, mode = 'markers', text=resume_keywords, opacity=0.75)

layout = go.Layout(title='Jobscan.co keyword ranking vs. City of LA occurrences',

    xaxis=dict(title='Jobscan.co keyword ranking'),yaxis=dict(title='Number of occurrences in City of LA vacancies'))

py.iplot(go.Figure(data=[data], layout=layout))
keywords_per_vacancy = np.squeeze(np.asarray(sum(csr_mat.todense().transpose()!=0)))

data = [go.Histogram(x=keywords_per_vacancy, xbins=dict(start=0,end=99,size= 1),opacity=0.75)]

layout = go.Layout(title='Number of keywords per vacancy',

    xaxis=dict(title='Number of keywords'),yaxis=dict(title='Number of vacancies'),bargap=0.05)

py.iplot(go.Figure(data=data, layout=layout))

keywords_per_vacancy==0

df.loc[keywords_per_vacancy==0,'Position'].head(5)

df['Matching_keywords']=keywords_per_vacancy
df_cm=pd.DataFrame(csr_mat.todense())

df_cm.index.name = 'Vacancies'

cm=sns.clustermap(df_cm,figsize=(15, 15))

cm.fig.suptitle('Clustered vacancies based on keywords') 

cm.ax_heatmap.set_xlabel('Keywords')

cm.ax_heatmap.set_ylabel('Vacancies')
print(df.loc[cm.dendrogram_row.reordered_ind,'Position'].head(20))
def remove_words(text, word_list):

    text = text.lower()

    for word in word_list:

        pattern = r"\b" + word.lower() + r"\b"

        text = re.sub(pattern, "", text)

    return text



# from DataCamp course "Machine learning with the experts"

def combine_text_columns(data_frame, to_keep):

    """ converts all text in each row of data_frame to single vector"""

    text_data = data_frame[to_keep]



    # Replace nans with blanks

    text_data.fillna("", inplace=True)



    # Join all text items in a row that have a space in between

    return text_data.apply(lambda x: " ".join(x), axis=1)



# from github pjbull/SparseInteractions.py

class SparseInteractions(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2, feature_name_separator="_"):

        self.degree = degree

        self.feature_name_separator = feature_name_separator



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        if not sparse.isspmatrix_csc(X):

            X = sparse.csc_matrix(X)



        if hasattr(X, "columns"):

            self.orig_col_names = X.columns

        else:

            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])



        spi = self._create_sparse_interactions(X)

        return spi



    def get_feature_names(self):

        return self.feature_names



    def _create_sparse_interactions(self, X):

        out_mat = []

        self.feature_names = self.orig_col_names.tolist()



        for sub_degree in range(2, self.degree + 1):

            for col_ixs in combinations(range(X.shape[1]), sub_degree):

                # add name for new column

                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])

                self.feature_names.append(name)



                # get column multiplications value

                out = X[:, col_ixs[0]]

                for j in col_ixs[1:]:

                    out = out.multiply(X[:, j])



                out_mat.append(out)



        return sparse.hstack([X] + out_mat)



# from github drivendataorg/box-plots-sklearn

def multilabel_sample(y, size=1000, min_count=5, seed=None):

    """ Takes a matrix of binary labels `y` and returns

        the indices for a sample of size `size` if

        `size` > 1 or `size` * len(y) if size =< 1.

        The sample is guaranteed to have > `min_count` of

        each label.

    """

    try:

        if (np.unique(y).astype(int) != np.array([0, 1])).any():

            raise ValueError()

    except (TypeError, ValueError):

        raise ValueError('multilabel_sample only works with binary indicator matrices')



    if (y.sum(axis=0) < min_count).any():

        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')



    if size <= 1:

        size = np.floor(y.shape[0] * size)



    if y.shape[1] * min_count > size:

        msg = "Size less than number of columns * min_count, returning {} items instead of {}."

        warn(msg.format(y.shape[1] * min_count, size))

        size = y.shape[1] * min_count



    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))



    if isinstance(y, pd.DataFrame):

        choices = y.index

        y = y.values

    else:

        choices = np.arange(y.shape[0])



    sample_idxs = np.array([], dtype=choices.dtype)



    # first, guarantee > min_count of each label

    for j in range(y.shape[1]):

        label_choices = choices[y[:, j] == 1]

        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)

        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])



    sample_idxs = np.unique(sample_idxs)



    # now that we have at least min_count of each, we can just random sample

    sample_count = int(size - sample_idxs.shape[0])



    # get sample_count indices from remaining choices

    remaining_choices = np.setdiff1d(choices, sample_idxs)

    remaining_sampled = rng.choice(remaining_choices,

                                   size=sample_count,

                                   replace=False)



    return np.concatenate([sample_idxs, remaining_sampled])



def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):

    """ Takes a features matrix `X` and a label matrix `Y` and

        returns (X_train, X_test, Y_train, Y_test) where all

        classes in Y are represented at least `min_count` times.

    """

    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])



    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)

    train_set_idxs = np.setdiff1d(index, test_set_idxs)



    test_set_mask = index.isin(test_set_idxs)

    train_set_mask = ~test_set_mask



    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])

import warnings

warnings.filterwarnings("ignore")

# data

text_columns = ["duties"]

text_data = combine_text_columns(df, to_keep=text_columns)

text_data.head()



# Tfidf for y labels - with smaller ngram_range to reduce computation time

keywords = resume_keywords

vec_tf = TfidfVectorizer(vocabulary=keywords, ngram_range=[1,3])

freq = vec_tf.fit_transform(text_data)

freq_df = pd.DataFrame(freq.toarray(), columns=keywords)



# List most frequent keywords

top_keywords = freq_df.sum().sort_values(ascending=False)[0:50]

not_occurring = freq_df.sum()[freq_df.sum() == 0]
# model

# binary version of most frequent outcome (y) values = keywords

use_keywords = list(top_keywords[0:24].index)

freq_df_bin = freq_df.loc[:, freq_df.columns.isin(use_keywords)]

freq_df_bin[freq_df_bin>0]=1

freq_df_bin.shape

X_train, X_test, y_train, y_test = multilabel_train_test_split(text_data, freq_df_bin, size=0.25, min_count=3, seed=467)

pl = Pipeline([

        ('vectorizer', TfidfVectorizer(ngram_range=[1,3])),  # no vocab here, want to use all the text

        ('feature_sel', SelectKBest(chi2, 100)),

        ('int', SparseInteractions(degree=2)),

        ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100)))

    ])

pl.fit(X_train, y_train)

print('')
# predicted probabilities for all labels

y_pred = pl.predict_proba(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=freq_df_bin.columns, index=y_test.index)

y_pred_df.head()

y_pred_df.sum()



# binary version

cutoff = 0.3

y_pred_bin = copy.deepcopy(y_pred_df)

y_pred_bin[y_pred_bin>=cutoff]=1

y_pred_bin[y_pred_bin<cutoff]=0

# compare y_test and y_pred_bin

sign = np.sign(y_test - y_pred_bin.values)

sign[sign==0] = 1

y_diff = (y_test + y_pred_bin.values)*sign

plt.figure(figsize=(16,16))

sns.heatmap(y_diff, cmap=sns.diverging_palette(20,240, n=75), center=0)

print("True positive: label predicted and present in test set labels")

print("True negative: label not predicted and not present in test set labels")

print("False positive: label predicted but not present in test set")

print("False negative: label not predicted but present in test set labels")
# quantify amount of true pos/neg (red), false pos (black), false neg (white)

flat = np.array(y_diff).flatten()

true_pos = np.mean(flat==2)

true_neg = np.mean(flat==0)

false_pos = np.mean(flat==-1)

false_neg = np.mean(flat==1)

samples_90_correct = np.mean(y_diff.apply(lambda row: np.mean(row==0), axis=1) >= 0.9)



print("Fraction 90% correct: " + str(samples_90_correct))

print("True positives: " + str(true_pos))

print("True negatives: " + str(true_neg))

print("False positives: " + str(false_pos))

print("False negatives: " + str(false_neg))
df.to_csv('Keyword_approach_output.csv')