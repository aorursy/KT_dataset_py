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
DATA_BASE_PATH = '../input/tweet-sentiment-extraction/' 

# Read training & test data as pandas-dataframe (ending _df)
train_df = pd.read_csv(DATA_BASE_PATH + 'train.csv') 
test_df = pd.read_csv(DATA_BASE_PATH + 'test.csv')
print(f"Train-Data shape is: {train_df.shape}") 
print(f"Test-Data shape is: {test_df.shape}")
train_df.head()
test_df.head()
train_df.dropna(inplace = True, how = 'any')
train_df.sample(15)
train_df.loc[[9710]]
train_df.loc[[1305]]
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

train_absolute_count = train_df["sentiment"].value_counts()
train_relative_count = train_df["sentiment"].value_counts(normalize = True)
test_relative_count = test_df["sentiment"].value_counts(normalize = True)

# Create figure and add traces
fig = make_subplots(1,3, subplot_titles = ('TRAIN data absolute amount',
                                           'TRAIN data relative amount', 
                                           'TEST data relative amount'))
for i in fig['layout']['annotations']:
            i['font'] = dict(size = 13)
  
fig.add_trace(go.Bar(x = train_absolute_count.index, y = train_absolute_count.values, 
                     marker_color = ['blue','green','red'], name= ''), row = 1, col = 1)

fig.add_trace(go.Bar(x = train_relative_count.index, y = train_relative_count.values,                     
                     marker_color = ['blue','green','red'], name= ''), row = 1, col = 2)

fig.add_trace(go.Bar(x = test_relative_count.index, y = test_relative_count.values,
                     marker_color = ['blue','green','red'], name= ''), row = 1, col = 3)


title_text = "Absolut and relative distribution of sentiments in train and test data"
fig.update_layout(title={'text': title_text})

# Define default go-layout for future use
default_layout =  go.Layout(  
    title = {                    
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    titlefont = {
     'size' : 15, 
     'color': 'black'
    },
    font = {
      'size' : 10 
    })

fig.update_layout(default_layout)

fig.show()
train=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
class_count = train['sentiment'].value_counts()
class_count
cache = """
train_max_len = max(train_df["TEXT_number_of_words"])
bins = np.linspace(0, train_max_len, train_max_len) # np.linspace takes following arguments: startpoint, endpoint, number of steps

# Cache tbd removed?
cache = plt.hist(train_df["TEXT_number_of_words"], bins, alpha=0.5, label = 'Number of words in "TEXT"')
cache = plt.hist(train_df["SELECTED_TEXT_number_of_words"], bins, alpha=0.5, label = 'Number of words in "SELECTED_TEXT"')
cache = plt.legend(loc = 'upper right')
cache = plt.style.use('seaborn-deep')
plt.gcf().set_size_inches(15, 7)  # Wow, there's really no set_size_cm ...but yea, a conversion via a pre-defined tuple could help here.
cache = plt.show()
"""
import seaborn as sns

train_df["TEXT_number_of_words"] = train_df["text"].apply(lambda x: len(str(x).split()))  
train_df["SELECTED_TEXT_number_of_words"] = train_df["selected_text"].apply(lambda x: len(str(x).split()))
train_max_len = max(train_df["TEXT_number_of_words"])

fig = plt.figure(figsize=(18,8))
# Number of bins shall equal the max-length in train_df['text']
bins = np.linspace(0, train_max_len, train_max_len) 
# sns.distplot is a nice combination of sns.hist and sns.kdplot
plot1 = sns.distplot(train_df['TEXT_number_of_words'], 
                     bins = bins, 
                     label = 'TEXT_number_of_words')
plot1 = sns.distplot(train_df['SELECTED_TEXT_number_of_words'], 
                     bins =  bins,  
                     label = 'SELECTED_TEXT_number_of_words')  
cache = plt.legend() 

fig.suptitle('Distribution of number of words', fontsize = 20)

# Defining default parameter for plt.rc for later re-use.
params = {
    'figure.titlesize': 22, # fontsize of plot title / fig.suptitle
    'axes.titlesize': 14,   # fontsize of the axes title
    'axes.labelsize': 11,   # fontsize of the x and y labels    
    'xtick.labelsize': 11,  # fontsize of the tick labels
    'ytick.labelsize': 11,  # fontsize of the tick labels
    'legend.fontsize': 12,  # fontsize for legends (plt.legend(), fig.legend())
}

plt.rcParams.update(params)
train_df['Diff_len_text_selected_text'] = train_df['TEXT_number_of_words'] - train_df['SELECTED_TEXT_number_of_words']
import seaborn as sns

grid = sns.FacetGrid(train_df, col = 'sentiment', height = 8)
 
grid.map(sns.lineplot, 'TEXT_number_of_words', 'Diff_len_text_selected_text', ci = None)
grid.add_legend()
plt.subplots_adjust(top = 0.8)
cache = grid.fig.suptitle('Difference of len(TEXT) and len(SELECTED_TEXT) over number of words in "TEXT"', fontsize = 24)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
from tqdm import trange
import random
from spacy.util import compounding,minibatch
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_box = go.Figure()

for _, sentiment in enumerate(sentiments):
    # CHANGE HERE! You can change "Diff_len_text_selected_text" in below line to 
    # "TEXT_number_of_words" or "SELECTED_TEXT_number_of_words" to get the respective boxplots.  
    fig_box.add_trace(go.Box(y = train_df[train_df['sentiment'] == sentiment]['Diff_len_text_selected_text'], name = sentiment)) 

title_text = 'Boxplot diagram difference in len(text) and len(selected_text)'
fig_box.update_layout(title = {'text': title_text})
fig_box.update_layout(default_layout)
          
fig_box.show()

def startsOrEndsWithSpecialCharacter(rowString): 
    '''
    Returns a boolean for whether a given string starts OR ends with an unexpected characters
    '''
    # Check beginning of the string
    pattern = '^[\/_,.|;:#*~+-?!].*'
    value =  0 if (re.match(pattern, rowString) == None) else 1
    # Check ending: expected interpunctuation at the end of the selected text (! ? .) is allowed.  
    pattern = '.*[\/_,|;:#*~+-]$' 
    value = value | (0 if (re.match(pattern, rowString) == None) else 1)
    # Check ending for white spaces before ending on a spec. character. E.g. "hi ."
    value = value | endsWithAppendedSpecialCharactersAndWhitespace(rowString) 
    # print(re.match(pattern, rowString)) # for analysis, if needed
    return value
def startsWithPrependedSpecialCharactersAndWhitespace(rowString):   
    pattern = '(^[\/_|,;.:#*~+-!?]\s+.*)'
    # print(re.match(pattern, rowString)) 
    return 0 if (re.match(pattern, rowString) == None) else 1

# looking for stuff like ',so happy'
def startsWithPrependedSpecialCharactersNoWhitespace(rowString): 
    pattern = '(^[\/_|,;.:#*~+-](?!\s+).*)'
    # print(re.match(pattern, rowString))
    return 0 if (re.match(pattern, rowString) == None) else 1

# looking for stuff like 'so happy,') 
# expected interpunctuation at the end of the selected text (! ? .) is allowed
def endsWithAppendedSpecialCharactersNoWhitespace(rowString):  
    pattern = '.*(<\s)[\/_|,;:#*~+-]$'  
    # print(re.match(pattern, rowString))
    return 0 if (re.match(pattern, rowString) == None) else 1

# looking for stuff like 'so happy ,') 
def endsWithAppendedSpecialCharactersAndWhitespace(rowString):  
    pattern = '(.*\s+)?[\/_,.|;:#*~+-?!]$'
    # print(re.match(pattern, rowString))
    return 0 if (re.match(pattern, rowString) == None) else 1
import re

startsOrEndsWithSpecialCharacter_series = []
startsOrEndsWithSpecialCharacter_series = train_df["selected_text"].apply(lambda x: startsOrEndsWithSpecialCharacter(x))
startsOrEndsWithSpecialCharacter_series = startsOrEndsWithSpecialCharacter_series[startsOrEndsWithSpecialCharacter_series > 0]
startsOrEndsWithSpecialCharacter_len = len((startsOrEndsWithSpecialCharacter_series[startsOrEndsWithSpecialCharacter_series == 1]))
print(f"There are {startsOrEndsWithSpecialCharacter_len} tweets"
      f"(= {startsOrEndsWithSpecialCharacter_len/train_df.shape[0] * 100:.2}%) that start or end with a special character pattern. \n"
      "Some examples look like this:")
train_df.loc[startsOrEndsWithSpecialCharacter_series.index].sample(10)[["textID", "text", "selected_text", "sentiment"]]
import spacy

spaCy_model = 'en_core_web_lg' 
nlp = spacy.load(spaCy_model)
spaCy_vocab_list = (list(nlp.vocab.strings))
# set ensures that all values are unique. All words converted to lower case.
spaCy_vocab_set = set([word.lower() for word in spaCy_vocab_list])
print(f"The loaded spaCy vocab {spaCy_model} contains unique lower case words: {len(spaCy_vocab_set)}")
train_df['clean_text'] = train_df['text'].str.lower()
train_df['clean_selected_text'] = train_df['selected_text'].str.lower()
test_df['clean_text'] = test_df['text'].str.lower()

# Create sets with unique words and update them 
train_text_vocab_set = set()
train_selected_text_vocab_set  = set()
test_text_vocab_set  = set()

# Apply set.update to fill the sets
train_df['clean_text'].str.lower().str.split().apply(train_text_vocab_set.update)
cache = train_df['clean_selected_text'].str.lower().str.split().apply(train_selected_text_vocab_set.update)
cache = test_df['clean_text'].str.lower().str.split().apply(test_text_vocab_set.update)
print(
    f"The used spaCy model contains unique words: {len(spaCy_vocab_set)}  \n"
    f"TRAIN-datas 'text' column contains unique words: {(len(train_text_vocab_set))}  \n"
    f"TRAIN-datas 'selected_text' column contains unique words: {len(train_selected_text_vocab_set)} \n"
    f"TEST-datas 'selected_text' column contains unique words: {len(test_text_vocab_set)} \n"
)
fraction_shared_words_train_text_to_spaCy = len(train_text_vocab_set.intersection(spaCy_vocab_set)) / (len(train_text_vocab_set))
fraction_shared_words_train_selected_text_spaCy = len(train_selected_text_vocab_set.intersection(spaCy_vocab_set)) / (len(train_selected_text_vocab_set))
fraction_shared_words_test_text_spaCy = len(test_text_vocab_set.intersection(spaCy_vocab_set)) / (len(test_text_vocab_set))
fraction_shared_words_test_text_train_text = len(test_text_vocab_set.intersection(train_text_vocab_set)) / (len(test_text_vocab_set))
fig = go.Figure([go.Bar(x = ["train['text'] & spaCy", "train['selected_text'] & spaCy", "test['text'] & spaCy", "test['text'] & train['text']"], 
                        y = [fraction_shared_words_train_text_to_spaCy, fraction_shared_words_train_selected_text_spaCy, fraction_shared_words_test_text_spaCy, fraction_shared_words_test_text_train_text], 
                        marker_color = ['blue','green','red'])])

title_text = 'Fraction of words contained in both sets for relevant pairs'
fig.update_layout(title={'text': title_text})
fig.update_layout(default_layout)
 
fig.show()
print(f"Only {len(train_selected_text_vocab_set.intersection(train_text_vocab_set)) / (len(train_selected_text_vocab_set)):.4f}%"
      f" of 'selected_text' is fully available in 'text',.. giving us {len(train_selected_text_vocab_set - train_text_vocab_set)} words which are cut off: \n")
print(train_selected_text_vocab_set - train_text_vocab_set)
train_clean_text_word_list = ' '.join([i for i in train_df['clean_text']]).split()  
train_clean_selected_text_word_list = ' '.join([i for i in train_df['clean_selected_text']]).split()  
test_clean_text_word_list = ' '.join([i for i in test_df['clean_text']]).split()  

# Calculate differences 
not_shared_train_text_to_spaCy = [word for word in train_clean_text_word_list if ((word in train_text_vocab_set) and (word not in spaCy_vocab_set))]
not_shared_train_selected_text_to_spaCy = [word for word in train_clean_selected_text_word_list if ((word in train_selected_text_vocab_set) and (word not in spaCy_vocab_set))]
not_shared_test_text_to_spaCy = [word for word in test_clean_text_word_list if ((word in test_text_vocab_set) and (word not in spaCy_vocab_set))]
not_shared_test_text_to_train_text = [word for word in test_clean_text_word_list if ((word in test_text_vocab_set) and (word not in train_text_vocab_set))]
from wordcloud import WordCloud, STOPWORDS

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(not_shared_train_text_to_spaCy ))
titleFontsize = 20

ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Not shared words: train_text and spaCy vocab',fontsize = titleFontsize);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(not_shared_train_selected_text_to_spaCy))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Not shared words: train_selected_text and spaCy vocab',fontsize = titleFontsize)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(not_shared_train_text_to_spaCy ))
titleFontsize = 20

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(not_shared_test_text_to_spaCy))
ax1.imshow(wordcloud3)
ax1.axis('off')
ax1.set_title('Not shared words: test_text and spaCy vocab',fontsize=titleFontsize);

wordcloud4 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(not_shared_test_text_to_train_text))
ax2.imshow(wordcloud4)
ax2.axis('off')
ax2.set_title('Not shared words: Test_text to train_text',fontsize=titleFontsize)
def jaccard_score (str1, str2):
    '''
    Returns the Jaccard Score (intersection over union) of two strings.
    '''
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    
    return float(len(c) / (len(a) + len(b) - len(c)))  
train_df["Jaccard_Score_text_selected_text"] = train_df.apply(lambda x: jaccard_score(str(x["text"]), str(x["selected_text"])), axis = 1) 
[avg_Jaccard_neu_train, avg_Jaccard_neg_train, avg_Jaccard_pos_train] = [train_df[train_df["sentiment"] == sentiment]["Jaccard_Score_text_selected_text"].mean() for sentiment in ["neutral", "negative", "positive"]]
avg_Jaccard_train = pd.Series(data = {'Avg Jaccard neutral': avg_Jaccard_neu_train, 'Avg Jaccard negative': avg_Jaccard_neg_train, "Avg Jaccard positive": avg_Jaccard_pos_train}, name = 'Jaccard Score per sentiment' )

print(f"Overall avg. Jaccard Score: {(avg_Jaccard_neu_train + avg_Jaccard_neg_train + avg_Jaccard_pos_train) / 3}")
print(avg_Jaccard_train)
train_val_split = 0.9

train_df_copy = train_df.copy() # Create a copy of the train_df to work with and not change stuff in-place
# Setting a random state for reproducable splits
train_set_df = train_df_copy.sample(frac = train_val_split, random_state = 42) 
val_set_df = train_df_copy.drop(train_set_df.index)
val_set_df.drop(['clean_text', 'clean_selected_text'], axis='columns', inplace = True)

# Get val_set for each sentiment
val_set_pos_df = val_set_df[val_set_df['sentiment'] == 'positive'].copy()
val_set_neg_df = val_set_df[val_set_df['sentiment'] == 'negative'].copy()
val_set_neu_df = val_set_df[val_set_df['sentiment'] == 'neutral'].copy()
def get_training_data(sentiment, splitAtLength = 0):
    '''
    Returns the training data in spaCy-required format for the given sentiment.
    If 'splitAtLength' is > 0, two none-empty arrays are returned: 
    first array containing the train_data with a 'text' containing more words than 'splitAtLength' 
    and the second arraycontaining train_data with 'text' containing number of words up to 'splitAtLength'.
    If splitAtLength is 0 or None, only one none-empty array 
    containing all train_data is returned, the second array is empty.
    
            Parameters:
                    sentiment (str): sentiment for which train_data needs to be returned
                    splitAtLength (int, optional): determins if and where the train_data is split

            Returns:
                    train_data (array): Returns the train data in an array, or array of arrays, if splitAtLength > 0.
    '''
    train_data_long = []
    train_data_short = []    
    
    for idx in train_set_df.index:
        if train_set_df.at[idx, 'sentiment'] == sentiment:
            text = train_set_df.at[idx,'text']
            len_text = len(text.split())            
            selected_text = train_set_df.at[idx,'selected_text']
            start = text.find(selected_text)
            end = start + len(selected_text)
            # create the train data in spaCy-required format. We can choose any "dummy_label" here
            # as we are anyway training just ONE model per sentiment:
            # all labels would be identical (e.g. positive) anyway.
            if (splitAtLength == None) or (splitAtLength == 0):
                   train_data_long.append((text, {"entities": [[start, end, "dummy_label"]]}))
            elif len(len_text) >= splitAtLength:
                   train_data_long.append((text, {"entities": [[start, end, "dummy_label"]]}))
            elif len(len_text) < splitAtLength:
                   train_data_short.append((text, {"entities": [[start, end, "dummy_label"]]}))
            else: print("something is wrong in getting training data")  
            
    return [train_data_long, train_data_short]
def get_validation_data(sentiment, model_type, splitAtLength = None):
    '''
    Returns the validation data used in training function
           
           Parameters:
                    sentiment (str): sentiment for which validation data needs to be returned
                    model_type (str): determins if data for short or long model is needed
                    splitAtLength (int, optional): determins if and where the data is split

           Returns:
                    train_data (array): Returns the train data in an array, or array of arrays, if splitAtLength > 0.
    '''
    if ((splitAtLength is None) or (splitAtLength == 0)):
        val_set_new_df = val_set_df[val_set_df['sentiment'] == sentiment].copy()  
        return val_set_new_df
    
    if (splitAtLength > 0) and (model_type == "short"): # tweets with length up to splitAtLength
        val_set_new_df = val_set_df[val_set_df["text"].str.split().str.len() < splitAtLength].copy()
        val_set_new_df = val_set_new_df[val_set_new_df['sentiment'] == sentiment].copy()
        return val_set_new_df
    
    if (splitAtLength > 0) and (model_type == "long"): # tweets with length > splitAtLength
        val_set_new_df = val_set_df[val_set_df["text"].str.split().str.len() >= splitAtLength].copy()
        val_set_new_df = val_set_new_df[val_set_new_df['sentiment'] == sentiment].copy()
        return val_set_new_df
def get_model_output_path(sentiment, splitAtLength = None, train_val_split = 0.90):
    '''
    Creates an easy to understand path for saving the model based on  the models parameters
    '''
    model_out_paths = []
    if splitAtLength == None or splitAtLength == 0:
        model_out_paths.append('models/model_'
                               + str(sentiment)
                               + '_splitAtLength_Longer_than'
                               + str(splitAtLength)
                               + '_train_val_split_'
                               + str(train_val_split))
        model_out_paths.append(None)
    elif splitAtLength != None:
        model_out_paths.append('models/model_' 
                               + str(sentiment)
                               + '_splitAtLength_Longer_than'
                               + str(splitAtLength)
                               + '_train_val_split_'
                               + str(train_val_split))
        model_out_paths.append('models/model_'
                               + str(sentiment)
                               + '_splitAtLength_Up_To'
                               + str(splitAtLength) + '_train_val_split_' 
                               + str(train_val_split))
    return model_out_paths
def save_model(output_dir, model , new_model_name, rank):
    '''
    Saves the model to an easy to understand path
    
        Parameters:
            output_dir (str): easy to understand path derived from models parameters
            model (model): the trained NLP model from spaCy            
            new_model_name (str): name for the model, NOT used for loading later
            rank (str): determins if the model is the best, 2nd best or 3rd best and adds it to save-path.
    '''
    output_dir = f'../working/{output_dir}_{rank}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.meta["name"] = new_model_name
        model.to_disk(output_dir)
        print("Saved model to", output_dir)
import numpy as np # algebra
import pandas as pd # data frames & processing, CSV file I/O 

# Visualization, graphs & plots
import matplotlib.pyplot as plt
import plotly.graph_objects as go # plots & graphics 
from plotly.subplots import make_subplots 
import seaborn as sns # plots & graphics

# SpaCy
import spacy
from spacy.util import compounding
from spacy.util import minibatch
from thinc.neural.optimizers import Adam # for hyperparameter tuning
from thinc.neural import Model # for hyperparameter tuning

# Utilities & helpers
from tqdm import tqdm_notebook as tqdm # progress bar
import time
import os
import re # Regex library
import random
import timeit # measuring execution time
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.


def train(sentiment, output_dir, epochs = 1, model = None, optimizer = None, dropout = 0.5, splitAtLength = 0, fill_no_predictions_with_text = False):
    '''
    Load the model, set up the pipeline and train the entity recognizer
    '''
    # define aborting conditions
    if epochs == 0 or epochs == None:
        return
    if output_dir == [] or output_dir == None:
        return

    # extract info if this model shall be for short or for long tweets from "output_dir"
    if "Longer_than" in output_dir:
        model_type = "long"
        # for long model, get 1st entry in array from get_training_data (used below)
        short_data = 0 
    elif "Up_To" in output_dir:
        model_type = "short" 
         # for short model, get 2nd entry in array from get_training_data (used below)
        short_data = 1
        
    # get train data relevant for this training session
    train_data = get_training_data(sentiment, splitAtLength = 0)[short_data]
    
    # get validation data relevant for this training session
    val_data = get_validation_data(sentiment, model_type, splitAtLength = splitAtLength)
   
    if output_dir == None:
        return 
   
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model to continue training
        print(f'Loaded model {model}')
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last = True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
        print("Getting NER-pipe in spaCy model.")
    
    # add all labels available in train_data
    # we could adjust train_data to get more/different labels
    # this will be explained in detail later.
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER       
        if model is None:
            nlp.begin_training()
            optimizer = nlp.begin_training() if optimizer is None else optimizer
        else:
            nlp.resume_training() 
            optimizer = nlp.resume_training() if optimizer is None else optimizer

        # initialize values to avoid "referenced before assignment" issues.
        best_Jaccard_score, second_best_Jaccard_score, third_best_Jaccard_score = [0,0,0]
        last_update_best_model, last_update_2nd_best_model, last_update_3rd_best_model = [0,0,0]
        improvement_best, improvement_2ndbest, improvement_3rdbest = [0,0,0]
        
       
        # actual train-step
        for itn in tqdm(range(epochs)):
            random.shuffle(train_data)
            # batch up the examples using spaCy's minibatch
            # compounding(start batch size, end batch size,  compounding factor).
            batches = minibatch(train_data, size = compounding(1.0, 100.0, 1.15) )    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                                    annotations,  # batch of annotations
                                    drop = dropout,   # makes it harder to memorize data
                                    sgd = optimizer,
                                    losses=losses, 
                                    )
            
       # test model on validattion_df and measure time
            start = time.time()
            [avg_pred_jaccard, number_no_predictions] = val_predictions_and_calc_Jaccard(
                sentiment = sentiment,
                model = nlp,
                val_df = val_data,               
                fill_no_predictions_with_text = fill_no_predictions_with_text)
            ende = time.time()
            print("Losses", losses)
            print(f'Avg. Jaccard Score for sentiment "{sentiment}" is: {avg_pred_jaccard:.4f},'                  
                  f' Number of empty predictions: {number_no_predictions}')
            
            # keep track of top 3 models & save them                      
            if avg_pred_jaccard > best_Jaccard_score:
                        improvement_best = avg_pred_jaccard - best_Jaccard_score
                        best_Jaccard_score = avg_pred_jaccard
                        save_model(output_dir,
                                   model = nlp, 
                                   new_model_name = output_dir.split('/')[-1],
                                   rank = 'best')
                        last_update_best_model = itn + 1  
            elif avg_pred_jaccard > second_best_Jaccard_score:
                        improvement_2ndbest = avg_pred_jaccard - second_best_Jaccard_score
                        second_best_Jaccard_score = avg_pred_jaccard
                        save_model(output_dir,
                                   model = nlp,
                                   new_model_name = output_dir.split('/')[-1],
                                   rank = 'second_best')
                        last_update_2nd_best_model = itn + 1     
            elif avg_pred_jaccard > third_best_Jaccard_score:
                        improvement_3rdbest = avg_pred_jaccard - third_best_Jaccard_score
                        third_best_Jaccard_score = avg_pred_jaccard
                        save_model(output_dir,
                                   model = nlp,
                                   new_model_name = output_dir.split('/')[-1],
                                   rank = 'third_best')  
                        last_update_3rd_best_model = itn + 1   
            else: print("Model didn't perform better, therefore not saved.")    
                    
    
    if best_Jaccard_score > 0: # only if some progress was made:
        print(f'\n \n \nBest model reached {best_Jaccard_score:.4f} and was updated' 
              f' (+{improvement_best:.4f}) last in {last_update_best_model}th epoch \n'
              f'2nd best model reached {second_best_Jaccard_score:.4f} and was updated'
              f' (+{improvement_2ndbest:.4f}) last in {last_update_2nd_best_model}th epoch\n'
              f'3rd best model reached {third_best_Jaccard_score:.4f} and was updated' 
              f' (+{improvement_3rdbest:.4f})last in {last_update_3rd_best_model}th epoch')    

    log = str(splitAtLength) + '_' + str(model_type)
    results[log] = {'best_Jaccard_score': best_Jaccard_score,
                                  'sentiment': sentiment,
                                  'splitAtLength': splitAtLength,                              
                                  'model_type': model_type}
def make_predictions(text, model, fill_no_predictions_with_text):
    '''
    Predicts entities based on the given model.
    Set fill_no_predictions_with_text to TRUE for creating a valid (high scoring) submission.
    If fill_no_predictions_with_text is false; all no-predictions will be marked with
    "NO-PREDICTION" for fruther evaluation.
    '''
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        ent_array.append([start, end, ent.label_])
    if fill_no_predictions_with_text:
        selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    else:
        selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else "NO-PREDICTION"
    return selected_text
def val_predictions_and_calc_Jaccard (sentiment, model, val_df, fill_no_predictions_with_text):    
    val_df['prediction']  = val_df["text"].apply(str).apply(lambda x: make_predictions(x, model, fill_no_predictions_with_text))
    val_df['pred_jaccard'] = val_df.apply(lambda x: jaccard_score (str(x['selected_text']), str(x['prediction'])), axis = 1)
    return [val_df['pred_jaccard'].mean(), np.sum(val_df["prediction"] == "NO-PREDICTION")]           
'''
splitAtLength: Allows to create a model per sentiment for tweets longer than splitAtLength words
and another model for shorter tweets --> so two models per sentiment.
splitAtLength = 0 (default): means NO split at all --> 1 model per sentiment.
splitAtLength_list: Can contain one or multiple values which will be iterated over.
Use this for grid search for optimal length!
Median for positive and negative tweets is 12, so half of the tweets contain less than 
(or equal to) 12 words and the other half contains more.
'''
import numpy as np # algebra
import pandas as pd # data frames & processing, CSV file I/O 

# Visualization, graphs & plots
import matplotlib.pyplot as plt
import plotly.graph_objects as go # plots & graphics 
from plotly.subplots import make_subplots 
import seaborn as sns # plots & graphics

# SpaCy
import spacy
from spacy.util import compounding
from spacy.util import minibatch
from thinc.neural.optimizers import Adam # for hyperparameter tuning
from thinc.neural import Model # for hyperparameter tuning

# Utilities & helpers
from tqdm import tqdm_notebook as tqdm # progress bar
import time
import os
import re # Regex library
import random
import timeit # measuring execution time
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.
splitAtLength_list = [0] 

# Epochs for long and short model
epochs_long = 30
epochs_short = 30

dropout_rate = 0.2

'''
If fill_no_predictions_with_text is FALSE: all rows without a prediction will be marked with
"NO-PREDICTION" for further evaluation.
'''
fill_no_predictions_with_text = True 

# Optimizer options
ops = Model.ops
learn_rate = 0.0015 # default: 0.001
L2 = 1e-5 # L2 regularisation penatly. Default: 1e-6
max_grad_norm = 1.0 # avoiding exploding gradients. Default: 1
optimizer = Adam(ops, learn_rate, L2 = L2) 
optimizer.max_grad_norm = max_grad_norm

# Determin for which sentiments to train a model for.
# Neutral sentiment is skipped here, as we can't get a better score
# for neutral sentiment than just using text als selected_text.

train_for_sentiments = ['negative', 'positive']

## Start the training
for splitAtLength in splitAtLength_list:
    for sentiment in train_for_sentiments:
        # Gget model path and determin whether to train two or one model per sentiment based on 
        # the value of  'splitAtLength'. returns [model_path_long, []] if splitAtLength = 0. 
        # Returns two paths if splitAtLength > 0
        [model_path_long, model_paths_short]  = get_model_output_path(
            sentiment,
            splitAtLength = splitAtLength,
            train_val_split = train_val_split) 

        train(sentiment, 
              output_dir = model_path_long, 
              epochs = epochs_long, 
              model = None,
              optimizer = optimizer,
              dropout = dropout_rate, 
              splitAtLength = splitAtLength, 
              fill_no_predictions_with_text = fill_no_predictions_with_text)
        
        # Train for short model will only be executed if model_paths_short != [] 
        train(sentiment, 
              output_dir = model_paths_short, 
              epochs = epochs_short, 
              model = None,
              optimizer = optimizer, 
              dropout = dropout_rate,
              splitAtLength = splitAtLength, 
              fill_no_predictions_with_text = fill_no_predictions_with_text)
def load_best_models(splitAtLength):
    '''
    Loads the best models, predicting for each row in "target_df" and putting result in "target_column"
    '''
    BASE_PATH = f'../working/'
    BASE_PATH_BEST_LONG_POS = BASE_PATH + get_model_output_path("positive", splitAtLength, train_val_split)[0] + "_best"
    BASE_PATH_BEST_SHORT_POS = BASE_PATH + get_model_output_path("positive", splitAtLength, train_val_split)[1] + "_best" if get_model_output_path("positive", splitAtLength, train_val_split)[1] else "no-model"
    BASE_PATH_BEST_LONG_NEG = BASE_PATH + get_model_output_path("negative", splitAtLength, train_val_split)[0] + "_best"
    BASE_PATH_BEST_SHORT_NEG = BASE_PATH + get_model_output_path("negative", splitAtLength, train_val_split)[1] + "_best" if get_model_output_path("positive", splitAtLength, train_val_split)[1] else "no-model"
        
    model_long_best_pos = spacy.load(BASE_PATH_BEST_LONG_POS) if os.path.isdir(BASE_PATH_BEST_LONG_POS) else None
    model_short_best_pos = spacy.load(BASE_PATH_BEST_SHORT_POS) if os.path.isdir(BASE_PATH_BEST_SHORT_POS) else None
    model_long_best_neg = spacy.load(BASE_PATH_BEST_LONG_NEG) if os.path.isdir(BASE_PATH_BEST_LONG_NEG) else None
    model_short_best_neg = spacy.load(BASE_PATH_BEST_SHORT_NEG) if os.path.isdir(BASE_PATH_BEST_SHORT_NEG) else None
    print(f'Models loaded:\nModel_pos_long_best: {model_long_best_pos != None} \n'
          f'Model_pos_short_best: {model_short_best_pos != None} \n'
          f'Model_neg_long_best: {model_long_best_neg != None}\n'
          f'Model_neg_short_best: {model_short_best_neg != None}')
    return [model_long_best_pos, model_short_best_pos, model_long_best_neg, model_short_best_neg]
def fill_dataframe_with_predictions(splitAtLength, target_df, target_column, fill_no_predictions_with_text):
    '''
    Loads the best models, predicting for each row in "target_df" and putting result in "target_column"
    '''
    
    [model_long_best_pos, model_short_best_pos, model_long_best_neg, model_short_best_neg] = load_best_models(splitAtLength)
                 
    # making it easier to deal with splitAtLength = None
    if splitAtLength is None:
        splitAtLength = 0
            
    target_df[target_column] = "EMPTY"
        
    for idx in target_df.index:
        text = target_df.at[idx,'text']
        sentiment = target_df.at[idx,'sentiment']
        if sentiment == 'neutral':  
            target_df.at[idx, target_column] = target_df.at[idx, 'text']
            # positive sentiment    
        elif sentiment == 'positive' and len(text.split()) > splitAtLength:
            target_df.at[idx, target_column] = make_predictions(
                                                                text,
                                                                model_long_best_pos,
                                                                fill_no_predictions_with_text) if model_long_best_pos != None else text  
        elif sentiment == 'positive' and len(text.split()) <= splitAtLength:
            target_df.at[idx, target_column] = make_predictions(text,
                                                                model_short_best_pos,
                                                                fill_no_predictions_with_text) if model_short_best_pos != None else text
            # negative sentiment      
        elif sentiment == 'negative' and len(text.split()) > splitAtLength:
            target_df.at[idx, target_column] = make_predictions(text,
                                                                model_long_best_neg,
                                                                fill_no_predictions_with_text) if model_long_best_neg != None else text
        elif sentiment == 'negative' and len(text.split()) <= splitAtLength:
            target_df.at[idx, target_column] = make_predictions(text,
                                                                model_short_best_neg,
                                                                fill_no_predictions_with_text) if model_short_best_neg != None else text
        else:
            print('something is wrong with fillSubmissionDf()')
    return target_df
val_set_with_preds_df = fill_dataframe_with_predictions(splitAtLength,
                                                        val_set_df.copy(),
                                                        target_column = 'pred_selected_text',
                                                        fill_no_predictions_with_text = False)