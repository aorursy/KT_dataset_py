import json
import time
import warnings
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.pipeline import make_pipeline, make_union
warnings.filterwarnings('ignore')
def apply_word2vec(sentences):
    vectorizer = Word2Vec(
        sentences,
        size=500,
        window=20,
        min_count=3,
        sg=1,
        iter=20
    )

    def to_vector(sentence):
        words = [word for word in sentence if word in vectorizer.wv.vocab]
        if words:
            return np.mean(vectorizer[words], axis=0)
        else:
            return np.zeros(500)

    return np.array([to_vector(sentence) for sentence in sentences])

def apply_fasttext(sentences):
    vectorizer = FastText(
        size=500,
        window=20,
        min_count=3,
        sg=1,
        iter=20
    )
    vectorizer.build_vocab(sentences)
    vectorizer.train(sentences, total_examples=vectorizer.corpus_count, epochs=vectorizer.iter)

    def to_vector(sentence):
        words = [word for word in sentence if word in vectorizer.wv.vocab]
        if words:
            return np.mean(vectorizer.wv[words], axis=0)
        else:
            return np.zeros(500)

    return np.array([to_vector(sentence) for sentence in sentences])

def train_model(x, y, n_splits=3):
    model = LogisticRegression(C=10, solver='sag', multi_class='multinomial', max_iter=300, n_jobs=-1)
    i = 0
    accuracies = []
    kfold = KFold(n_splits)
    for train_index, test_index in kfold.split(x):
        classifier = LogisticRegression(C=10, solver='sag', multi_class='multinomial', max_iter=300, n_jobs=-1)
        classifier.fit(x[train_index], y[train_index])

        predictions = classifier.predict(x[test_index])
        accuracies.append(accuracy_score(predictions, y[test_index]))
        i += 1
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

def run_experiment(preprocessor):
    train = json.load(open('../input/train.json'))

    target = [doc['cuisine'] for doc in train]
    lb = LabelEncoder()
    y = lb.fit_transform(target)

    x = preprocessor.fit_transform(train)

    return train_model(x, y)
results = []
for (name, preprocessor) in [
    ('CountVectorizer', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        CountVectorizer(),
    )),
    ('TfidfVectorizer', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(),
    )),
    ('TfidfVectorizer + TruncatedSVD', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(),
        TruncatedSVD(n_components=500, algorithm='arpack')
    )),
    ('Word2Vec', make_pipeline(
        FunctionTransformer(lambda x: [doc['ingredients'] for doc in x], validate=False),
        FunctionTransformer(lambda x: apply_word2vec(x), validate=False),
    )),
    ('fastText', make_pipeline(
        FunctionTransformer(lambda x: [doc['ingredients'] for doc in x], validate=False),
        FunctionTransformer(lambda x: apply_fasttext(x), validate=False),
    )),
    ('TfidfVectorizer + fastText', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        make_union(
            make_pipeline(
                TfidfVectorizer()
            ),
            make_pipeline(
                FunctionTransformer(lambda x: apply_fasttext(x), validate=False)
            )
        )
    ))
]:
    start = time.time()
    accuracy = run_experiment(preprocessor)
    execution_time = time.time() - start
    results.append({
        'name': name,
        'accuracy': accuracy,
        'execution time': f'{round(execution_time, 2)}s'
    })
pd.DataFrame(results, columns=['name', 'accuracy', 'execution time']).sort_values(by='accuracy', ascending=False)
results = []
for (name, preprocessor) in [
    ('TfidfVectorizer()', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(),
    )),
    ('TfidfVectorizer(binary=True)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(binary=True),
    )),
    ('TfidfVectorizer(min_df=3)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(min_df=3),
    )),
    ('TfidfVectorizer(min_df=5)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(min_df=5),
    )),
    ('TfidfVectorizer(max_df=0.95)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(max_df=0.95),
    )),
    ('TfidfVectorizer(max_df=0.9)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(max_df=0.9),
    )),
    ('TfidfVectorizer(sublinear_tf=True)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
    )),
    ('TfidfVectorizer(strip_accents=unicode)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(strip_accents='unicode'),
    )),
]:
    start = time.time()
    accuracy = run_experiment(preprocessor)
    execution_time = time.time() - start
    results.append({
        'name': name,
        'accuracy': accuracy,
        'execution time': f'{round(execution_time, 2)}s'
    })
pd.DataFrame(results, columns=['name', 'accuracy', 'execution time']).sort_values(by='accuracy', ascending=False)
results = []
for (name, preprocessor) in [
    ('TruncatedSVD(n_components=100, algorithm=randomized)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=100, algorithm='randomized')
    )),
    ('TruncatedSVD(n_components=100, algorithm=arpack)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=100, algorithm='arpack')
    )),
    ('TruncatedSVD(n_components=200, algorithm=randomized)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=200, algorithm='randomized')
    )),
    ('TruncatedSVD(n_components=200, algorithm=arpack)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=200, algorithm='arpack')
    )),
    ('TruncatedSVD(n_components=500, algorithm=randomized)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=500, algorithm='randomized')
    )),
    ('TruncatedSVD(n_components=500, algorithm=arpack)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=500, algorithm='arpack')
    )),
    ('TruncatedSVD(n_components=1000, algorithm=randomized)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=1000, algorithm='randomized')
    )),
    ('TruncatedSVD(n_components=1000, algorithm=arpack)', make_pipeline(
        FunctionTransformer(lambda x: [" ".join(doc['ingredients']).lower() for doc in x], validate=False),
        TfidfVectorizer(sublinear_tf=True),
        TruncatedSVD(n_components=1000, algorithm='arpack')
    )),
]:
    start = time.time()
    accuracy = run_experiment(preprocessor)
    execution_time = time.time() - start
    results.append({
        'name': name,
        'accuracy': accuracy,
        'execution time': f'{round(execution_time, 2)}s'
    })
pd.DataFrame(results, columns=['name', 'accuracy', 'execution time']).sort_values(by='accuracy', ascending=False)
import tensorflow as tf
import tensorflow_hub as hub

train = pd.read_json('../input/train.json').set_index('id')
train['ingredients'] = train['ingredients'].apply(lambda x: ' '.join(x))

train = train.reset_index()[['ingredients', 'cuisine']]
label_encoder = LabelEncoder()
train['cuisine'] = label_encoder.fit_transform(train['cuisine'])

train_input_fn = tf.estimator.inputs.pandas_input_fn(train, train['cuisine'], num_epochs=None, shuffle=True)
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train, train['cuisine'], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="ingredients", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1"
)

classifier = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=len(train['cuisine'].unique()),
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

classifier.train(input_fn=train_input_fn, steps=1000)

train_eval_result = classifier.evaluate(input_fn=predict_train_input_fn)

"Training set accuracy: {accuracy}".format(**train_eval_result)