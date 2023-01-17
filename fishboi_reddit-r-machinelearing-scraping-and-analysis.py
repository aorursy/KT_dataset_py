use_dataset = True
if not use_dataset:
    !pip install psaw
    #import praw # Official Reddit API
    from psaw import PushshiftAPI # Unofficial Reddit API (for getting content by date)
from timeit import default_timer as timer
from time import sleep
from datetime import datetime, timezone, timedelta
import os
import json
import pandas as pd
import time
import pickle
import nltk
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn
from tensorflow import keras

subreddits = ["MachineLearning"]
date_begin = datetime(2009, 7, 29)
date_final = datetime(2020, 3, 1)
request_delay = 2
months_per_interval = 6
if use_dataset:
    submissions = pickle.load(open('/kaggle/input/redditmachinelearning/submissions_fe.pickle', 'rb'))
    comments    = pickle.load(open('/kaggle/input/redditmachinelearning/comments_fe.pickle'   , 'rb'))
class terms_list:
    terms_models = {
        "SVM": ["support vector machine", "svm", "support vector"],
        "RNN": ["rnn", "recurrent neural network", "recurrent neural net", "recurrent nn"],
        "LSTM": ["lstm", "long short term memory"],
        "CNN": ["convolutional neural network", "convolutional neural net", "convolutional nn", "cnn", "convolution", "pool", "pooling"],
        "Neural Network": ["ann", "nn", "neural network", "neural net"],
        "Deep Learning": ["deep learning"],

        # Regression Models
        "Linear Regression": ["linear regression"],
        "Logistic Regression": ["logistic regression"],
        "LASSO Regression": ["lasso regression", "lasso"],
        "Ridge Regression": ["ridge regression", "ridge"],
        "Regression": ["regression"],

        "Random Forest": ["random forest"],
        "Decision Tree": ["decision tree"],
        "Naive Bayes": ["naive bayes", "naïve bayes"],
        "K-Means": ["k means"],
        "KNN": ["knn", "k nearest neighbors", "k nearest neighbor"],
        "PCA": ["pca", "principal component analysis"],
    }

    terms_ensembles = {
        "Ensemble": ["ensemble learning", "ensemble"],
        "Bagging": ["bagging", "bagged"],
        "Boosting": ["boosting"],
        "Stacking": ["stacking", "stacked"],
        "Blending": ["blending", "blended"],
        "XGBoost": ["xgboost", "xg boost"],
        "ADABoost": ["adaboost", "ada boost"],
        "Random Forest": ["random forest", "randomforest"],
    }

    terms_activations = {
        "Sigmoid": ["sigmoid", "logistic activation"],
        "tanh": ["tanh", "hyperbolic tangent"],
        "Leaky ReLU": ["leaky relu", "leakyrelu"],
        "Parametric ReLU": ["parametric relu", "parametricrelu"],
        "ReLU": ["relu"],
        "ELU": ["elu"],
        "Softmax": ["softmax", "soft max"],
        "Swish": ["swish"], # What is this?
    }

    terms_neural_nets = {
        "RNN": ["rnn", "recurrent neural network", "recurrent neural net", "recurrent nn"],
        "LSTM": ["lstm", "long short term memory"],
        "CNN": ["convolutional neural network", "convolutional neural net", "convolutional nn", "cnn", "convolution", "pool", "pooling"],
        "Neural Network": ["ann", "nn", "neural network", "neural net"],
        "Deep Learning": ["deep learning"],
        "Network": ["network", "net"],
        "Neuron": ["neuron"],
        "Perceptron": ["perceptron"],
        "Layer": ["hidden layer", "layer"],
        "Dropout": ["dropout"],
        "Fully Connected": ["fully connected"],
        "Dense": ["dense"],
        "Tensorflow": ["tensorflow", "tensor flow"],
        "TensorBoard": ["tensorboard", "tensor board"],
        "Torch": ["pytorch", "torch"],
        "Keras": ["keras"],
        "Theano": ["theano"],
        "Caffe": ["caffe", "caffe2"]
    }
    terms_neural_nets.update(terms_activations)

    terms_misc = {
        "Dying ReLU": ["dying relu"],
        "Vanishing Gradient": ["vanishing gradient"],
        "Kernel": ["kernel"],
    }

    terms_loss = {
        # Regression
        "Huber Loss": ["huber", "smooth mean absolute error"],
        "Log-Cosh Loss": ["log cosh loss"],
        "MSE": ["mse", "mean squared error", "mean square error", "l2 loss", "quadratic loss"],
        "MAE": ["mae", "mean absolute error", "l1 loss"],
        # Classification
        "Log Loss": ["logarithmic loss", "log loss", "logistic loss", "binary cross entropy", "binary entropy"],
        "Categorical Cross-Entropy": ["categorical cross entropy", "categorical entropy"],
        "Hinge Loss": ["hinge", "hinge loss"],
    }

    terms_optimizers = {
        "Gradient Descent": ["gradient descent"],
        "Adagrad": ["adagrad", "ada grad"],
        "RMSProp": ["rmsprop", "rms prop"],
        "Adam": ["adam"],
    }

    terms_regularization = {
        "Regularization": ["regularization", "regularized", "regularize"],
        "LASSO Regression": ["lasso regression", "lasso"],
        "Ridge Regression": ["ridge regression", "ridge"],
        "Dropout": ["dropout"],
    }

    terms_areas = {
        "Regression": ["regression"],
        "Classification": ["classification", "classifier"],
        "Unsupervised Learning": ["unsupervised"],
        "Supervised Learning": ["supervised"],
        "Reinforcement Learning": ["reinforcement learning", "rl"],
        "Clustering": ["clustering", "clusters", "cluster"],
        "Dimensionality Reduction": ["dimensionality reduction", "dimensional reduction", "dimension reduction"],
        "NLP": ["nlp", "natural language processing"]
    }
    
    def __init__(self):
        self.terms_all = dict()
        self.terms_all.update(self.terms_models)
        self.terms_all.update(self.terms_ensembles)
        self.terms_all.update(self.terms_neural_nets)
        self.terms_all.update(self.terms_activations)
        self.terms_all.update(self.terms_misc)
        self.terms_all.update(self.terms_loss)
        self.terms_all.update(self.terms_optimizers)
        self.terms_all.update(self.terms_regularization)
        self.terms_all.update(self.terms_areas)
    
terms = terms_list()
if not use_dataset:
    pushapi = PushshiftAPI()

    # Create necessary directories
    if not os.path.exists("data"):
        os.mkdir("data")
    for subreddit in subreddits:
        if not os.path.exists("data/{}".format(subreddit)):
            os.mkdir("data/{}".format(subreddit))
        if not os.path.exists("data/{}/submission".format(subreddit)):
            os.mkdir("data/{}/submission".format(subreddit))
        if not os.path.exists("data/{}/comment".format(subreddit)):
            os.mkdir("data/{}/comment".format(subreddit))

    # Get submissions and comments from each subreddit
    for subreddit in subreddits:
        # Should we resume progress?
        if len(os.listdir("data/" + subreddit + "/comment")) > 0:
            # Set day to last recorded day + 1 (assume files haven't been deleted)
            files = os.listdir("data/{}/submission".format(subreddit))
            files.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d.json"))
            date_start = datetime.strptime(files[len(files) - 1], "%Y-%m-%d.json") + timedelta(days=1)
            print("Resuming {} at day {}".format(subreddit, date_start))
        else:
            date_start = date_begin

        date_end = date_start + timedelta(days=1)

        # Iterate over each day
        while True:
            submissions = list()
            if date_end > date_final: break

            # Get submissions
            subsearch = pushapi.search_submissions(
                subreddit = subreddit,
                after = date_start,
                before = date_end,
                filter = [
                    "title",        # Title of submission
                    "selftext",     # Submission body of text (unless it's media)
                    "score",        # Upvotes - Downvotes
                    "num_comments", # Number of comments
                    "id",           # Submission ID, to link the post
                    "author",       # Username
                    #"num_crossposts"
                ], # Where is upvote_ratio?
                limit = 500, # Max number of results
            )
            sleep(request_delay)

            # Get comments
            comsearch = pushapi.search_comments(
                subreddit = subreddit,
                after = date_start,
                before = date_end,
                filter = [
                    "body",                  # Comment body of text
                    "score",                 # Upvotes - Downvotes
                    "author",                # Username
                    "total_awards_received", # Number of rewards received
                    "id",                    # ID of comment
                    "parent_id",             # Comment being replied to (equal to link_id if replying to submission)
                    "link_id",               # Original submission link
                ],
                limit = 500
            )
            sleep(request_delay)

            # Extract list of dictionaries from results
            submissions = [submission.d_ for submission in subsearch]
            comments    = [comment.d_    for comment    in comsearch]

            if len(submissions) > 500: print("\tWARNING: Failed to capture all submission data with limit=500")
            if len(comments)    > 500: print("\tWARNING: Failed to capture all comment data with limit=500")

            # Save to JSON file
            path = "data/{}/{}/{}"
            fname = "{}-{}-{}.json".format(
                date_start.year, date_start.month, date_start.day
            )
            with open(path.format(subreddit, "submission", fname), 'w') as fp:
                json.dump(submissions, fp)
            with open(path.format(subreddit, "comment", fname), 'w') as fp:
                json.dump(comments, fp)

            print(fname)

            # Re-iterate
            date_start += timedelta(days=1)
            date_end   += timedelta(days=1)
        print("done")
if not use_dataset:
    submissions_list = list()
    comments_list    = list()
    for subreddit in os.listdir("data"):
        if not subreddit in subreddits: continue
        for day_json in os.listdir("data/{}/submission".format(subreddit)):
            if not day_json.endswith(".json"): continue
            # Assume submission dir is same size as comment dir
            with open("data/{}/submission/{}".format(subreddit, day_json), "r") as f_sub, \
                 open("data/{}/comment/{}"   .format(subreddit, day_json), "r") as f_com:
                day_subs = json.load(f_sub)
                day_coms = json.load(f_com)
                for sub in day_subs: sub["subreddit"] = subreddit
                for com in day_coms: com["subreddit"] = subreddit
                submissions_list.extend(day_subs)
                comments_list   .extend(day_coms)
if not use_dataset:
    submissions = pd.DataFrame(submissions_list)
    submissions["created_utc"] = submissions["created_utc"].apply(datetime.utcfromtimestamp)
    submissions.drop(["created"], axis=1, inplace=True)
    submissions = submissions.astype({"subreddit": "category"})
    submissions = submissions[submissions["selftext"] != "[removed]"]
    submissions = submissions[submissions["selftext"] != "[deleted]"]

    comments = pd.DataFrame(comments_list)
    comments["created_utc"] = comments["created_utc"].apply(datetime.utcfromtimestamp)
    comments.drop(["created"], axis=1, inplace=True)
    comments = comments.astype({"subreddit": "category"})
    comments = comments[comments["body"] != "[removed]"]
    comments = comments[comments["body"] != "[deleted]"]

    # Get direct replies to comments
    post_replies = comments["parent_id"].map(lambda x: x.split("_")[1]).value_counts()
    def get_replies(cid):
        if cid in post_replies: return post_replies[cid]
        return 0
    comments["direct_replies"] = comments["id"].apply(get_replies)

    # Get total replies to comments (includes replies to replies of comment)
    total_replies = {com_id: 0 for com_id in comments["id"].unique()}
    def increment_replies(row):
        if row["link_id"] == row["parent_id"]: return
        pid = row["parent_id"].split("_")[1]
        if pid not in total_replies: return # Skip unrecorded comments
        total_replies[pid] += 1
        new_row = comments[comments["id"] == pid]
        if len(new_row) > 1: print(len(new_row))
        increment_replies(new_row.iloc[0])
    for idx, row in comments.iterrows():
        if   idx == 0: time_start = time.time()
        if idx % 1000 == 0:
            print(str(idx) + " / " + str(len(comments)))
        increment_replies(row)
    comments["total_replies"] = comments["id"].apply(lambda x: total_replies[x])
    
    display(submissions.dtypes)
    display(comments.dtypes)
if not use_dataset:
    # Given a text, frequency dictionary, and row number, records each ML term found in the text in the frequencies
    stemmer = nltk.SnowballStemmer("english")
    def count_terms(
        tokens,      # Tokenized text
        frequencies, # Current frequencies to be modified
        row,         # Which row of frequencies to modify
    ):
        # Iterate over each term
        for term in terms.terms_all.keys():
            count = 0
            # Iterate over all representations of term (neural net vs ANN, etc.)
            for rep in terms.terms_all[term]:
                # Tokenize term words
                rep_words = rep.split()

                # Search for representation in text
                start = 0
                end = len(rep_words)
                while True:
                    if end > len(tokens): break
                    found = True

                    # Check if tokens match
                    for i in range(start, end):
                        if stemmer.stem(tokens[i]) != rep_words[i-start] and tokens[i] != rep_words[i-start] and stemmer.stem(tokens[i]) != stemmer.stem(rep_words[i-start]):
                            found = False
                            break

                    if found:
                        count += 1
                        # Remove term from text
                        for i in range(start, end):
                            del tokens[start]
                    else:
                        start += 1
                        end += 1

            # Output total count from each representation
            frequencies[term][row] += count
if not use_dataset:
    stopwords = nltk.corpus.stopwords.words('english')

    # Returns a tokenized representation of a submission title
    def tokenize_title(title):
        text = str(title)
        text = text.lower()
        text = text.replace("-", " ")
        text = text.replace("'", "")
        text = text.replace(",", " ")
        text = text.replace("[Discussion]", "")
        text = text.replace("[D]", "")
        text = text.replace("[News]", "")
        text = text.replace("[N]", "")
        text = text.replace("[Research]", "")
        text = text.replace("[R]", "")
        text = text.replace("[Project]", "")
        text = text.replace("[P]", "")
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Filter stopwords for efficiency
        tokens = list(filter(lambda w: not w in stopwords, tokens))
        return tokens

    # Returns a tokenized representation of a submission body or comment
    def tokenize_body(body):
        text = str(body)
        text = text.lower()
        text = text.replace("-", " ")
        text = text.replace("'", "")
        text = text.replace(",", " ")
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Filter stopwords for efficiency
        tokens = list(filter(lambda w: not w in stopwords, tokens))
        return tokens

    # Add columns to DF's for each term
    for term in terms.terms_all.keys():
        submissions[term] = 0
        comments[term] = 0

    num_subs = len(submissions)
    num_coms = len(comments)

    # Set up dictionary of frequencies of terms, with a list representing the rows
    frequencies_subs = terms.terms_all.copy()
    frequencies_coms = terms.terms_all.copy()
    for term in terms.terms_all.keys():
        frequencies_subs[term] = [0 for _ in range(num_subs)]
        frequencies_coms[term] = [0 for _ in range(num_coms)]

    # Iterate over submission titles
    print("Submission Titles:")
    for row, title in enumerate(submissions.title):
        if row % 2000 == 0: print(str(row) + " / " + str(num_subs))
        tokens = tokenize_title(title)             # Tokenize the text
        count_terms(tokens, frequencies_subs, row) # Count occurences of each term
    print("done\n")

    # Iterate over submission bodies
    print("Submission Bodies:")
    for row, selftext in enumerate(submissions.selftext):
        if row % 2000 == 0: print(str(row) + " / " + str(num_subs))
        tokens = tokenize_body(selftext)           # Tokenize the text
        count_terms(tokens, frequencies_subs, row) # Count occurences of each term
    print("done\n")

    # Add to DataFrame
    for term in terms.terms_all.keys():
        submissions[term] = frequencies_subs[term]

    pickle.dump(submissions, open("submissions_fe.pickle", "wb"))

    # Iterate over comment bodies
    print("Comment Bodies:")
    for row, body in enumerate(comments.body):
        if row % 500 == 0: print(str(row) + " / " + str(num_coms))
        tokens = tokenize_body(body)               # Tokenize the text
        count_terms(tokens, frequencies_coms, row) # Count occurences of each term
        
    print("done")

    # Add to DataFrame
    for term in terms.terms_all.keys():
        comments[term] = frequencies_coms[term]

    pickle.dump(comments, open("comments_fe.pickle", "wb"))
    
    # Display counts of each term
    for i in frequencies_subs.keys():
        print(str(i) + ": " + str(sum(frequencies_subs[i])))
# Group posts into date ranges
date_ranges = list()
i = 0
while True:
    date1 = date_begin + timedelta(days = i * months_per_interval * 365 / 12)
    date2 = date_begin + timedelta(days = (i+1) * months_per_interval * 365 / 12)
    # Skip last interval (not full)
    if date2 >= date_final: break
    date_ranges.append((date1, date2))
    i += 1

submissions_by_date = list()
comments_by_date    = list()
for date_range in date_ranges:
    submissions_by_date.append(submissions[(submissions["created_utc"] > date_range[0]) & (submissions["created_utc"] < date_range[1])])
    comments_by_date   .append(comments   [(comments   ["created_utc"] > date_range[0]) & (comments   ["created_utc"] < date_range[1])])
def show_barplot(term_list, frequencies, date_strs, ymax=100, colored_date="", use_frequencies=True):
    fig, ax = plt.subplots(1, len(term_list), figsize=(7.5 * len(term_list), 7.5/2))
    
    for plot_idx, term in enumerate(term_list):
        if len(term_list) > 1:
            if use_frequencies: ax[plot_idx].set_title("% usage of term '{}' over time".format(term))
            else:               ax[plot_idx].set_title("# of occurences of term '{}' over time".format(term))
    #         ax[plot_idx].tick_params(labelrotation=45)
            ax[plot_idx].set_xticklabels(date_strs, rotation=45, ha="right")
            ax[plot_idx].set_ylim([0, ymax])
    #         ax[plot_idx].set_xticks(rotation=45, ha="right")
            for idx in range(len(date_strs)):
                date = date_strs[idx]
                color = "blue"
                if date == colored_date: color = "red"
                ax[plot_idx].bar(date, frequencies[term][idx], color=color)
        else:
            if use_frequencies: ax.set_title("% usage of term '{}' over time".format(term))
            else:               ax.set_title("# of occurences of term '{}' over time".format(term))
    #         ax[plot_idx].tick_params(labelrotation=45)
            ax.set_xticklabels(date_strs, rotation=45, ha="right")
            ax.set_ylim([0, ymax])
    #         ax[plot_idx].set_xticks(rotation=45, ha="right")
            for idx in range(len(date_strs)):
                date = date_strs[idx]
                color = "blue"
                if date == colored_date: color = "red"
                ax.bar(date, frequencies[term][idx], color=color)
        
    plt.show()

def get_terms_by_date(terms_list, use_frequencies):
    global frequencies
    global date_strs
    
    frequencies = dict()
    date_strs = list()
    
    for term in terms_list: frequencies[term] = list()

    for dateidx in range(len(submissions_by_date)):
        subs = submissions_by_date[dateidx]
    #     date_start = date_ranges[dateidx][0]
        date_end   = date_ranges[dateidx][1]
        date_strs.append(date_end.strftime("%m/%d/%y"))

        total = 0 # Total count of all terms
        if use_frequencies:
            for term in terms_list: total += subs[term].sum()
        for term in terms_list:
            if use_frequencies: frequencies[term].append(subs[term].sum() / total * 100)
            else:               frequencies[term].append(subs[term].sum())
frequencies = None
date_strs = None

get_terms_by_date(terms.terms_models, True)
show_barplot(["Deep Learning", "Neural Network"], frequencies, date_strs, 60, "01/26/16")
show_barplot(["Regression", "Deep Learning", "SVM"], frequencies, date_strs, 30, "01/26/16")
show_barplot(["RNN", "LSTM", "CNN"], frequencies, date_strs, 30, "01/26/16")
get_terms_by_date(terms.terms_ensembles, True)
show_barplot(["Bagging", "Boosting", "Stacking"], frequencies, date_strs, 70, "07/28/15")
show_barplot(["XGBoost", "ADABoost", "Random Forest"], frequencies, date_strs, 50, "07/28/15")
my_terms = terms.terms_misc
get_terms_by_date(my_terms, False)
show_barplot(["Vanishing Gradient"], frequencies, date_strs, 12, "07/28/14", use_frequencies=False)

my_terms = terms.terms_activations
get_terms_by_date(my_terms, True)
show_barplot(["Sigmoid", "tanh", "ReLU", ], frequencies, date_strs, 100, "07/28/14")
show_barplot(["ReLU", "ELU"], frequencies, date_strs, 40, "07/28/14")
my_terms = terms.terms_areas
get_terms_by_date(my_terms, True)
show_barplot(["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"], frequencies, date_strs, 30)
# Get total sub votes for each date range
total_sub_votes = [0 for _ in range(len(submissions))]
for i in range(len(submissions_by_date)):
    sel = submissions_by_date[i].id.unique()
    sub_votes = submissions.id.apply(lambda id1: id1 in sel)
    sub_votes *= submissions_by_date[i].score.sum()
    total_sub_votes += sub_votes
submissions["total_sub_votes"] = total_sub_votes
submissions = submissions[submissions.total_sub_votes != 0]
subs = submissions
# subs = submissions_by_date[20]

X = subs.drop([
    "created_utc", "num_comments", "score", "selftext", "title", "subreddit", "id", "author"
], axis=1)
y = subs[["score"]]
# X = submissions_by_date[len(submissions_by_date)-1].drop([
#     "created_utc", "num_comments", "score", "selftext", "title", "subreddit", "id", "author"
# ], axis=1)
# y = submissions_by_date[len(submissions_by_date)-1][["score"]]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size = 0.20, random_state=42
)

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train, y_train, test_size = 0.25
)
model_lg = sklearn.linear_model.Lasso(alpha=.015)
model_lg = model_lg.fit(X_train, y_train)

print(sklearn.metrics.mean_absolute_error(
    y_test, model_lg.predict(X_test)
))

print(sklearn.metrics.mean_squared_error(
    y_test, model_lg.predict(X_test)
))

# print(model_lg.coef_.argmax())
# print(X.columns[model_lg.coef_.argmax()])
coefs = sorted(model_lg.coef_)
for i in range(len(coefs)):
    print(X.columns[i] + " : " + str(coefs[i]))