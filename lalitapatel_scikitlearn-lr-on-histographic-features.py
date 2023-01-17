import pandas as pd
from itertools import chain
from collections import Counter
from sklearn.linear_model import LogisticRegression
DATA_DIR = '../input/'
OUT_DIR = ''

train_df = pd.read_csv(DATA_DIR + 'train.csv')
test_df = pd.read_csv(DATA_DIR + 'test.csv')
def print_block(sometitle, someblock):
    '''
    Print something in between its title  and separator.
    '''
    print(sometitle)
    print("\n")
    print(someblock)
    print("\n" + "="*80 + "\n")
def text_to_words(orig_text):
    '''
    Trim a text, and split it into one-words and two-words.
    '''
    word_vect = orig_text.lower()
    word_vect = word_vect.replace("  "," ")
    remove_char = ['.', '?', '!', ',', ';', '(', ')', '"', """''"""]
    for i in range(9):
        word_vect = word_vect.replace(remove_char[i], "")
        
    # Make a list of one words.
    word_vect = word_vect.split()
    word_count = len(word_vect)
    
    # Add two consecutive words.
    if word_count > 2:
        for i in range(word_count - 2):
            word_vect.append(word_vect[i] + " " + word_vect[i+1])
    
    # Add two alternate words.
    if word_count > 3:
        for i in range(word_count - 3):
            word_vect.append(word_vect[i] + " " + word_vect[i+2])
    
    return word_vect
def count_match(df1, df2):
    """
    Count matching elements in two dataframes.
    """
    qty = 0
    for ele in df1:
        if ele in df2:
            qty = qty + 1
    return qty
def df_common_words(df):
    '''
    Prepare a dataframe of most common insincere and sincere words,
    based on a given dataframe of texts and lables.
    '''
    insincere_words = df[df['target']==1]['question_text'].apply(
                   lambda x: text_to_words(x)).reset_index(drop=True)
    insincere_words = Counter(chain.from_iterable(insincere_words[i] for
                   i in range(len(insincere_words))))
    insincere_words = pd.DataFrame(Counter.most_common(insincere_words),
                      columns=["word","freq"])
    print_block("10 most common insincere words", insincere_words[:10])
    
    sincere_words = df[df['target']==0]['question_text'].apply(
                   lambda x: text_to_words(x)).reset_index(drop=True)
    sincere_words = Counter(chain.from_iterable(sincere_words[i] for
                   i in range(len(sincere_words))))
    sincere_words = pd.DataFrame(Counter.most_common(sincere_words),
                    columns=["word","freq"])
    print_block("10 most common sincere words", sincere_words[:10])
    
    insincere_words['both'] = insincere_words['word'].isin(sincere_words['word'])
    sincere_words['both'] = sincere_words['word'].isin(insincere_words['word'])
    
    insincere_words = insincere_words[insincere_words.both == False].reset_index(
                      drop=True).drop(['freq','both'], axis=1)
    print_block("10 most insincere not-most-sincere words", insincere_words[:10])
    
    sincere_words = sincere_words[sincere_words.both == False].reset_index(
                    drop=True).drop(['freq','both'], axis=1)
    print_block("10 most sincere not-most-insincere words", sincere_words[:10])
    
    insincere_00 = insincere_words[:1]
    insincere_01 = insincere_words[1:2]
    insincere_02 = insincere_words[2:4]
    insincere_03 = insincere_words[4:8]
    insincere_04 = insincere_words[8:16]
    insincere_05 = insincere_words[16:32]
    insincere_06 = insincere_words[32:64]
    insincere_07 = insincere_words[64:128]
    insincere_08 = insincere_words[128:256]
    insincere_09 = insincere_words[256:512]
    insincere_10 = insincere_words[512:1024]
    insincere_99 = insincere_words[1024:]
    
    sincere_00 = sincere_words[:1]
    sincere_01 = sincere_words[1:2]
    sincere_02 = sincere_words[2:4]
    sincere_03 = sincere_words[4:8]
    sincere_04 = sincere_words[8:16]
    sincere_05 = sincere_words[16:32]
    sincere_06 = sincere_words[32:64]
    sincere_07 = sincere_words[64:128]
    sincere_08 = sincere_words[128:256]
    sincere_09 = sincere_words[256:512]
    sincere_10 = sincere_words[512:1024]
    sincere_11 = sincere_words[1024:2048]
    sincere_99 = sincere_words[2048:]
    
    df_common = (insincere_00, insincere_01, insincere_02, insincere_03, insincere_04,
                 insincere_05, insincere_06, insincere_07, insincere_08, insincere_09,
                 insincere_10, insincere_99,
                 sincere_00, sincere_01, sincere_02, sincere_03, sincere_04, sincere_05,
                 sincere_06, sincere_07, sincere_08, sincere_09, sincere_10, sincere_11,
                 sincere_99)
    
    return df_common
df_common = df_common_words(train_df)
def add_measure(df, typ):
    """
    Add various measures to a dataset.
    """
    df['chars'] = df['question_text'].apply(lambda x: len(x))
    df['words'] = df['question_text'].apply(lambda x: len(x.split()))
    
    for i in range(25):
        df['c' + str(100+i)] = df['question_text'].apply(lambda x:
                         count_match(x.split(" "), df_common[i]))
    
    dfx = df.drop(['question_text'], axis=1)
    print("First 5 elements of " + typ + " dataset:")
    print(dfx[:5])
    
    return dfx
train_df = add_measure(train_df, "training")
test_df = add_measure(test_df, "test")
columos = ["chars","words","c100","c101","c102","c103","c104","c105","c106",
           "c107","c108","c109","c110","c111","c112","c113","c114","c115",
           "c116","c117","c118","c119","c120","c121","c122","c123","c124"]

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model = clf.fit(train_df[columos], train_df['target'])
test_df['prediction'] = model.predict(test_df[columos])

print(test_df['prediction'][:20])
test_df[['qid','prediction']].to_csv(OUT_DIR + 'submission.csv', index = False)
