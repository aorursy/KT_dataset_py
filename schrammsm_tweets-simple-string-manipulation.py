import pandas as pd
sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train['text_lower'] = train.text.apply(lambda x: str(x).lower().strip())

train['selected_text_lower'] = train.selected_text.apply(lambda x: str(x).lower().strip())

train['in_out_is_same'] = train.text_lower == train.selected_text_lower
train['text_len'] = train.text_lower.apply(lambda x: len(x))

train['selected_text_len'] = train.selected_text_lower.apply(lambda x: len(x))



train['len_diff'] = train.text_len - train.selected_text_len
train.groupby(by=['sentiment', 'in_out_is_same']).textID.count()
positive_phrases = list(set(train[train.sentiment == 'positive'].selected_text.values))

negative_phrases = list(set(train[train.sentiment == 'negative'].selected_text.values))



len(positive_phrases), len(negative_phrases)
def simple_model(text, sentiment):

    if sentiment == 'neutral':

        selected_text = text

    elif sentiment == 'positive':

        selected_text = ' '.join([w for w in text.lower().split() if w in positive_phrases])

        if len(selected_text.strip()) < 1:

            selected_text = text

    elif sentiment == 'negative':

        selected_text = ' '.join([w for w in text.lower().split() if w in negative_phrases])

        if len(selected_text.strip()) < 1:

            selected_text = text

    return selected_text
sub.selected_text = test.apply(lambda x: simple_model(x.text, x.sentiment), axis=1)

sub.head()
sub.to_csv('submission.csv', index=False)