!pip install gensim
# Word2Vecは、単語同士の繋がりに基づいて単語同士の関係性をベクトル化します。
# ベクトルにすることで、単語同士の似た特徴を持つ単語の推測や、次にくる単語を予測することができます。
# 必要なモジュールのインポート
import gensim
from gensim import corpora
from pprint import pprint
from collections import defaultdict

# 解析対象の文章を準備
docs = []

# 文を単語に分割した配列にする
"""
こちら変数documentsに解析したい文章を代入してください。
"""
documents = ["今日 の 天気 は 良い 天気 だ",
             "明日 は 雨 が 降る",
             "今日 の お弁当 は 唐揚げ 弁当 だ",
             "彼 は 田中 さん が 好き だ"
]

# ストップワードとは、役に立たない単語を取り除く方法の1つで、あまりに頻出する役に立たない単語を捨てる方法
# ストップワードを定義
stop_words = set('だ'.split()) # 例えば、「だ」をストップワードにする

# 文を単語に分割し、ストップワードを除去した配列を作成する
texts = [[word for word in document.lower().split() if word not in stop_words] for document in documents]

pprint(texts)

dictionary = corpora.Dictionary(texts)
# ファイルに保存
dictionary.save('./sample.dict')

# テキストファイルに保存
dictionary.save_as_text('./sample.dict.txt')
corpus = [dictionary.doc2bow(text) for text in texts]
# ファイルに保存
corpora.MmCorpus.serialize('./sample.mm', corpus)
# num_topics=3とは、3個のトピックを持つLDAモデルを作成という意味
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=3, id2word=dictionary)
# テストしたい文章を定義
test_documents = ["今日 の 天気 は 曇り だ"]

# 単語を分割
test_texts = [[word for word in document.lower().split()] for document in test_documents]

# 既存の辞書を使用して、コーパスを作成
test_corpus = [dictionary.doc2bow(text) for text in test_texts]
pprint(test_corpus)
for topics_per_document in lda[test_corpus]:
    pprint(topics_per_document)