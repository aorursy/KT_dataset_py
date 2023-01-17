from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import word_tokenize

import pandas as pd

import sys

sys.path.insert(0, "../")
def replace_none(X):

    if X == '':

        X = np.nan

    return X
def build_model(max_epochs, vec_size, alpha, tagged_data):

    

    model = Doc2Vec(vector_size=vec_size,

               alpha=alpha,

               min_alpha=0.00025,

               min_count=1,

               dm=1)

    

    model.build_vocab(tag_data)

    

    # With the model built we simply train on the data.

    

    for epoch in range(max_epochs):

        print(f"Iteration {epoch}")

        model.train(tag_data,

                   total_examples=model.corpus_count,

                   epochs=model.epochs)



        # Here I decrease the learning rate. 



        model.alpha -= 0.0002



        model.min_alpha = model.alpha

    

    # Now simply save the model to avoid training again. 

    

    model.save("COVID_MEDICAL_DOCS_w2v_MODEL.model")

    print("Model Saved")

    return model
corona_df = pd.read_csv("../input/covid19-medical-paperscsv/kaggle_covid-19_open_csv_format.csv")
corona_df.isnull().sum()
corona_df['title'] = corona_df['title'].apply(replace_none)

corona_df['text_body'] = corona_df['text_body'].apply(replace_none)

corona_df = corona_df.dropna()



w2v_data_body = list(corona_df['text_body'])

w2v_data_title = list(corona_df['title'])



w2v_total_data = w2v_data_body + w2v_data_title
tag_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(w2v_total_data)]
model = build_model(max_epochs=5, vec_size=10, alpha=0.025, tagged_data=tag_data)
model.wv.similar_by_word("risk")
model.wv.similar_by_word("symptoms")
model.wv.similar_by_word("pregnant")
model.wv.similar_by_word("economy")
model.wv.similar_by_word("isolation")