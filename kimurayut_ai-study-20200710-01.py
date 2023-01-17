# クリーニング処理
import urllib.request
from bs4 import BeautifulSoup

url = "https://news.livedoor.com/topics/category/main/"
html = urllib.request.urlopen(url)

soup = BeautifulSoup(html, "lxml")

titles = soup.find_all("h3", class_="articleListTtl")
print(titles)
for title in titles:
    print(title.string)
!pip install gensim
# 形態素解析
from janome.tokenizer import Tokenizer
t = Tokenizer()
for token in t.tokenize('今日の天気は雨でした'):
    print(token)
# 正規化
!pip install neologdn
import neologdn

# 半角を全角に統一
full_letter = neologdn.normalize("ｸﾙﾏ")
print(full_letter) # クルマ

# 不要なスペース削除
cut_space = neologdn.normalize("こ　ん　に　ち　は　")
print(cut_space) # こんにちは

# 似た文字の統一
unification = neologdn.normalize("-⁃֊⁻₋−‑˗‒–")
print(unification) # -

# 伸ばし棒を一つにする
shorter = neologdn.normalize("とてもきれいだーーーーーー")
print(shorter) # とてもきれいだー

# 繰り返しの制限
cut_repeat = neologdn.normalize("あああいたたたいなぁ", repeat=1)
print(cut_repeat) # あいたいなぁ
import re
t = "12月のクリスマスに100万円のダイヤモンドをプレゼントする"
print(re.sub("[0-9]+","0", t)) # 0月のクリスマスに0万円のダイヤモンドをプレゼントする

#補足
#レアな単語は１つの文書中に1回または2回しかでてこない単語は、
#その文書の意味を捉える特徴量にはなりにくいです。そのため、
#このような単語はリスト化して取り除くか、まとめて1つの特徴量として扱うべきです。
#こうすることによって精度向上と、学習時間の削減を実現できます。
# ストップワード除去
import os
def download(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        urllib.request.urlretrieve(url, path)
        print("Finish!")

download("stopwords")
with open("./stopwords", "r", encoding="utf-8") as f:
    lines = f.readlines()

stopword_list = [line.split("\n")[0] for line in lines if line.split("\n")[0]]

# 形態素解析した表層形のリスト
text = ["あそこ","に","は","、", "多く", "の", "りんご", "が", "あり", "ます", "。"]

result = [word for word in text if word not in stopword_list]
print(result)
# 単語のベクトル表現
#　単語をベクトル化することで単語の類似度を測ることが可能になるなど扱いやすくなります。
#　最終的には「King」-「Man」＋「Woman」＝「Queen」という出力ができるようにしたい。
#　そのために単語ごとにベクトル化（特徴の数値化）を行い、上記のような出力ができるような形式にする。

from gensim.models import word2vec

data = word2vec.Text8Corpus('/kaggle/input/result.txt') # 形態素解析して表層形のみ書いてあるファイル

# size:圧縮したい次元数
# min_count:最低出現数
# window:ある単語の前後でみる単語数
# iter:反復数
model = word2vec.Word2Vec(data, size=100, min_count=5, window=5, iter=100)
print(model["猫"])