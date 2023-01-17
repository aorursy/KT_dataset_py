%%capture



!apt update && apt install -y openjdk-8-jdk

!update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

!python -m pip install --upgrade pip

!pip install --upgrade language-check pycontractions commonregex ekphrasis
%%capture



import warnings

from typing import List



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    from pycontractions import Contractions



    from ekphrasis.classes.tokenizer import SocialTokenizer

    from ekphrasis.classes.segmenter import Segmenter

    from ekphrasis.classes.preprocessor import TextPreProcessor

    from ekphrasis.dicts.emoticons import emoticons



    from sklearn.pipeline import Pipeline

    from sklearn.svm import LinearSVC

    from sklearn.model_selection import train_test_split as tts

    from sklearn.metrics import classification_report

    from sklearn.calibration import CalibratedClassifierCV

    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.pipeline import FeatureUnion

    from sklearn.linear_model import SGDClassifier

    from sklearn.ensemble import VotingClassifier

    from sklearn.naive_bayes import ComplementNB



    import pandas as pd

    import numpy as np
%%capture



EMBEDDINGS = "glove-twitter-200"



cont = Contractions(api_key=EMBEDDINGS)

cont.load_models()



text_processor = TextPreProcessor(

    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],

    annotate={}, # No annotation around special tokens

    fix_html=True,

    segmenter="twitter",

    corrector="twitter", 

    unpack_hashtags=True,

    unpack_contractions=True,

    spell_correct_elong=True, # Increase the preprocessing time

    tokenizer=SocialTokenizer(lowercase=False).tokenize,

    dicts=[emoticons],

    fix_text=True,

)
def decontraction(corpus: List[str]) -> List[str]:

    """

    Expand all contractions.

    """

    return list(cont.expand_texts(corpus))





def clean(corpus: List[str]) -> List[str]:

    """

    Clean the corpus.

    """

    return list(map(lambda text: " ".join(text_processor.pre_process_doc(text)), corpus))
%%capture



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train["clean_text"] = decontraction(train["text"].tolist())

train["clean_text"] = clean(train["clean_text"].tolist())



test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test["clean_text"] = decontraction(test["text"].tolist())

test["clean_text"] = clean(test["clean_text"].tolist())
%%capture



model = Pipeline(

    [

        ('embedder', FeatureUnion(

            [

                ('char',  TfidfVectorizer(sublinear_tf=True, lowercase=True, ngram_range=(1, 10), analyzer='char_wb')),

                ('word', TfidfVectorizer(sublinear_tf=True, lowercase=True, ngram_range=(1, 4), stop_words='english'))

            ])

        ),

        ('classifier', VotingClassifier(

            [

                ('clf1', CalibratedClassifierCV(LinearSVC(), cv=5)),

                ('clf2', ComplementNB()),

                ('clf3', SGDClassifier(alpha=1e-4, max_iter=50, penalty="elasticnet"))

            ])

        )

    ]

)
macro_f1s = []



for seed in [0, 1, 2, 42, 56]:

    train_, eval_ = tts(train, test_size=0.05, stratify=train["target"], random_state=seed)

    model.fit(train_["clean_text"], train_["target"])

    macro_f1s.append(classification_report(eval_["target"], model.predict(eval_["clean_text"]), output_dict=True)["macro avg"]["f1-score"])



print(f"Max: {np.max(macro_f1s)} Min: {np.min(macro_f1s)} Avg: {np.mean(macro_f1s)}")
model.fit(train["clean_text"], train["target"])
model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

model_submission['target'] = model.predict(test["clean_text"])

model_submission.to_csv('model_submission.csv', index=False)
train.to_csv("clean_train.csv")

test.to_csv("clean_test.csv")