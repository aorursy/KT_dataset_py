import pandas as pd

from gensim.models.word2vec import Word2Vec
train_df = pd.read_pickle("../input/hw2pkl/train.pkl")

test_df = pd.read_pickle("../input/hw2pkl/test.pkl")

train_df.head()
corpus = pd.concat([train_df.text, test_df.text]).sample(frac=1)

corpus.head()
model = Word2Vec(corpus)
def most_similar(w2v_model, words, topn=10):

    similar_df = pd.DataFrame()

    for word in words:

        try:

            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])

            similar_df = pd.concat([similar_df, similar_words], axis=1)

        except:

            print(word, "not found in Word2Vec model!")

    return similar_df
most_similar(model, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])
model.save('word2vec.model')

model = Word2Vec.load('word2vec.model')
model_d250 = Word2Vec(corpus, size=250, iter=10)

most_similar(model_d250, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])
model_d250_sg = Word2Vec(corpus, size=250, iter=10, sg=1)

most_similar(model_d250_sg, ['懷孕', '網拍', '補習', '東京', 'XDD', '金宇彬','化妝品', '奧斯卡', '主管', '女孩'])