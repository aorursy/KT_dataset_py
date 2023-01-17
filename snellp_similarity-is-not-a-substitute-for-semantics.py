import pandas as pd

pd.set_option('max_colwidth', 250) #so that the full column of tagged sentences can be displayed

import numpy as np

import nltk

from nltk.corpus import stopwords

from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning) #to stop the annoying deprecation warnings from sklearn



#Some simple functions

def remove_stopwords(tokenized_sent):

    unique_stopwords = set(stopwords.words('english'))

    return [word for word in tokenized_sent if word.lower() not in unique_stopwords]



def concatenate_tokens(token_list):

    return str(' '.join(token_list))



def find_similarity(sent1, sent2):

	return SequenceMatcher(lambda x: x in (' ', '?', '.', '""', '!'), sent1, sent2).ratio()



def return_common_tokens(sent1, sent2):

    return " ".join([word.lower() for word in sent1 if word in sent2])



def convert_tokens_lower(tokens):

    return [token.lower() for token in tokens]



#Reading the train file

train_sample = pd.read_csv('../input/train.csv', encoding = 'utf-8', index_col = 0, header = 0, iterator = True).get_chunk(100000)

#train_sample =train_sample[:100]
train_sample.head(2)

train_sample.size




transformed_sentences_train = pd.DataFrame(index = train_sample.index)

naive_similarity = pd.DataFrame()

temp_features = pd.DataFrame()

dictionary = pd.DataFrame()

temp_isDuplicate = pd.DataFrame()



#Deriving the naive features

for i in (1, 2):

        transformed_sentences_train['question%s_tokens' % i] = train_sample['question%s' % i].apply(nltk.word_tokenize)

        transformed_sentences_train['question%s_lowercase_tokens' % i] = transformed_sentences_train['question%s_tokens' % i].apply(convert_tokens_lower)

        transformed_sentences_train['question%s_lowercase' % i] = transformed_sentences_train['question%s_lowercase_tokens' % i].apply(concatenate_tokens)

        transformed_sentences_train['question%s_words' % i] = transformed_sentences_train['question%s_tokens' % i].apply(remove_stopwords)

        transformed_sentences_train['question%s_pruned' % i] = transformed_sentences_train['question%s_words' % i].apply(concatenate_tokens)

naive_similarity['similarity'] = np.vectorize(find_similarity)(train_sample['question1'], train_sample['question2'])

naive_similarity['pruned_similarity'] = np.vectorize(find_similarity)(transformed_sentences_train['question1_pruned'], transformed_sentences_train['question2_pruned'])

temp_features['common_tokens'] = np.vectorize(return_common_tokens)(transformed_sentences_train['question1_tokens'], transformed_sentences_train['question2_tokens'])



naive_similarity['is_duplicate'] = train_sample['is_duplicate']

naive_similarity['question1'] = train_sample['question1']

naive_similarity['question2'] = train_sample['question2']

#temp_isDuplicate = train_sample['is_duplicate']



#print (naive_similarity[:20])

#naive_similarity.head(18)
naive_similarity['Both'] = ((naive_similarity['similarity']>0.66 ) & (naive_similarity['is_duplicate']))
naive_similarity.head(20)
type(tempdf)
tempdf.size
naive_similarity.size
naive_similarity.head(3)
Í#tempdf.groupby('col1').count().size

counts = naive_similarity.groupby('Both').size(); counts

#tempdf.value_counts()
#test = np.vectorize(find_similarity)('good geologist ?', 'great geologist ?')

test = np.vectorize(find_similarity)("how can i be a good geologist ", "what should i do to be a great geologist ?")   

#test = np.vectorize(find_similarity)("howcanibeagoodgeologist?", "whatshouldidotobeagreatgeologist?")    

print (test)
transformed_sentences_train.head(8)
dictionary = pd.DataFrame()



#Deriving the TF-IDF

dictionary['concatenated_questions'] = transformed_sentences_train['question1_lowercase'] + transformed_sentences_train['question2_lowercase']



vectorizer = CountVectorizer()

terms_matrix = vectorizer.fit_transform(dictionary['concatenated_questions'])

terms_matrix_1 = vectorizer.transform(transformed_sentences_train['question1_lowercase'])

terms_matrix_2 = vectorizer.transform(transformed_sentences_train['question2_lowercase'])

common_terms_matrx = vectorizer.transform(temp_features['common_tokens'])



transformer = TfidfTransformer(smooth_idf = False)

weights_matrix = transformer.fit_transform(terms_matrix)

weights_matrix_1 = transformer.transform(terms_matrix_1)

weights_matrix_2 = transformer.transform(terms_matrix_2)

common_weights_matrix = transformer.transform(common_terms_matrx)



#Converting the sparse matrices into dataframes

transformed_matrix_1 = weights_matrix_1.tocoo(copy = False)

transformed_matrix_2 = weights_matrix_2.tocoo(copy = False)

transformed_common_weights_matrix = common_weights_matrix.tocoo(copy = False)



weights_dataframe_1 = pd.DataFrame({'index': transformed_matrix_1.row, 'term_id': transformed_matrix_1.col, 'weight_q1': transformed_matrix_1.data})[['index', 'term_id', 'weight_q1']].sort_values(['index', 'term_id']).reset_index(drop = True)

weights_dataframe_2 = pd.DataFrame({'index': transformed_matrix_2.row, 'term_id': transformed_matrix_2.col, 'weight_q2': transformed_matrix_2.data})[['index', 'term_id', 'weight_q2']].sort_values(['index', 'term_id']).reset_index(drop = True)

weights_dataframe_3 = pd.DataFrame({'index': transformed_common_weights_matrix.row, 'term_id': transformed_common_weights_matrix.col, 'common_weight': transformed_common_weights_matrix.data})[['index', 'term_id', 'common_weight']].sort_values(['index', 'term_id']).reset_index(drop = True)



#Summing the weights of each token in each question to get the summed weight of the question

sum_weights_1, sum_weights_2, sum_weights_3 = weights_dataframe_1.groupby('index').sum(), weights_dataframe_2.groupby('index').sum(), weights_dataframe_3.groupby('index').sum()



weights = sum_weights_1.join(sum_weights_2, how = 'outer', lsuffix = '_q1', rsuffix = '_q2').join(sum_weights_3, how = 'outer', lsuffix = '_cw', rsuffix = '_cw').join(train_sample['is_duplicate'])

weights = weights.fillna(0)

del weights['term_id_q1'], weights['term_id_q2'], weights['term_id']

#weights[is_dup] = train_sample['is_duplicate']





print (weights[:20])
X = naive_similarity.join(weights, how = 'inner')



#Creating a random train-test split

y = train_sample['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)



#Scaling the features

sc = StandardScaler()

for frame in (X_train, X_test):

    sc.fit(frame)

    frame = pd.DataFrame(sc.transform(frame), index = frame.index, columns = frame.columns)



print (X_train[:20])
#Training the algorithm and making a prediction

gbc = GradientBoostingClassifier(n_estimators = 8000, learning_rate = 0.3, max_depth = 3)

gbc.fit(X_train, y_train.values.ravel())

prediction = pd.DataFrame(gbc.predict(X_test), columns = ['is_duplicate'], index = X_test.index)



#Inspecting our mistakes

prediction_actual = prediction.join(y_test, how = 'inner', lsuffix = '_predicted', rsuffix = '_actual').join(train_sample[['question1', 'question2']], how = 'inner').join(X_test, how = 'inner')



print ('The log loss is %s' % log_loss(y_test, prediction))
print (prediction_actual[prediction_actual['is_duplicate_predicted'] != prediction_actual['is_duplicate_actual']][:10])