import pandas as pd
from itertools import chain
from collections import Counter
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
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
def df_insincerity(df):
    '''
    Prepare a dataframe of words and label-indices,
    based on a given dataframe of texts and lables.
    '''
    any_count = len(df)
    df_any = df['question_text'].apply(lambda x: text_to_words(x))
    print_block("Words in first five questions", df_any[:5])
    df_any = Counter(chain.from_iterable(df_any[i] for i in range(any_count)))
    print_block("5 most common words", Counter.most_common(df_any)[:5])
    
    insincere_count = len(df[df['target']==1])    
    df_insincere = df[df['target']==1]['question_text'].apply(
                   lambda x: text_to_words(x)).reset_index(drop=True)
    print_block("Words in first five insincere questions", df_insincere[:5])    
    df_insincere = Counter(chain.from_iterable(df_insincere[i] for
                   i in range(insincere_count)))
    print_block("5 most common words from insincere questions",
                Counter.most_common(df_insincere)[:5])
    
    sincere_count = len(df[df['target']==0])    
    df_sincere = df[df['target']==0]['question_text'].apply(
                 lambda x: text_to_words(x)).reset_index(drop=True)
    print_block("Words in first five sincere questions", df_sincere[:5])    
    df_sincere = Counter(chain.from_iterable(df_sincere[i] for
                 i in range(sincere_count)))
    print_block("5 most common words from sincere questions",
                Counter.most_common(df_sincere)[:5])
    
    df_insincerity_a = {k: round(df_insincere[k]/insincere_count -
                                 df_sincere[k]/sincere_count,
                                 2) for k in df_any.keys()}    
    df_insincerity_a = Counter({k: float(round(v, 2)) for k, v in
                       df_insincerity_a.items() if abs(v) > 0.0001})
    df_insincerity_a = pd.DataFrame(Counter.most_common(df_insincerity_a),
                       columns=["word","insincerity"])
    print_block("5 most insincere words", df_insincerity_a[:5])
    print_block("5 least insincere words", df_insincerity_a[-5:])
    
    return df_insincerity_a
df_insincerity = df_insincerity(train_df)
def question_insincerity(quest):
    '''
    Compute insincerity index of a question,
    based on the prepared list of words with insincerity indices.
    '''
    df_questword = pd.DataFrame(text_to_words(quest), columns=["word"])
    df_questword = pd.merge(df_questword, df_insincerity,
                   on='word', how='inner')
    
    if sum(df_questword['insincerity']) > 0:
        return 1
    else:
        return 0
test_df['prediction'] = test_df['question_text'].apply(
                        lambda x: question_insincerity(x))
print(test_df['prediction'][:50])
test_df[['qid','prediction']].to_csv('submission.csv', index = False)
