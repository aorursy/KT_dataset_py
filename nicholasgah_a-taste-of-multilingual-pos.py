!pip install malaya

!pip install spacy

!pip install pythainlp

!pip install jieba

!pip install underthesea

!pip install nagisa

!pip install sklearn_crfsuite
!python -m spacy download en_core_web_sm
import spacy

from pythainlp.tag.named_entity import ThaiNameTagger

import malaya

import jieba.posseg as pseg

from underthesea import pos_tag as vi_pos_tag

import nagisa
en_model = spacy.load("en_core_web_sm")

en_docs = ["Apple is looking at buying U.K. startup for $1 billion", "Autonomous cars shift insurance liability toward manufacturers",

          "And now for something completely different"]
for doc in en_docs:

    nlp = en_model(doc)

    pos_tags = [(token.text, token.pos_, token.tag_) for token in nlp]

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))
th_model = ThaiNameTagger()

th_docs = ["วันที่ 15 ก.ย. 61 ทดสอบระบบเวลา 14:49 น", "รถยนต์อิสระเปลี่ยนภาระการประกันภัยต่อผู้ผลิต", "และตอนนี้สำหรับสิ่งที่แตกต่างอย่างสิ้นเชิง"]
for doc in th_docs:

    pos_tags = [token[:-1] for token in th_model.get_ner(doc)]

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))
ms_model = malaya.pos.transformer(model = 'albert')

ms_docs = ["KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.",

          "Benda yg SALAH ni, jgn lah didebatkan. Yg SALAH xkan jadi betul. Ingat tu. Mcm mana kesat sekalipun org sampaikan mesej, dan memang benda tu salah, diam je. Xyah nk tunjuk kau open sangat nk tegur cara org lain berdakwah",

          "melayu bodoh, dah la gay, sokong lgbt lagi, memang tak guna"]
for doc in ms_docs:

    pos_tags = [{"text": token["text"], "type": token["type"]} for token in ms_model.analyze(doc)["tags"]]

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))
zh_model = pseg

zh_docs = ["小明硕士毕业于中国科学院计算所，后在日本京都大学深造", "我来到北京清华大学",

           "当被问及媒体各种揣测时，当地警方指出，“没有证据证明”这起案件与刘冰在匹兹堡大学的研究、以及新冠病毒疫情有任何关联。"]
for doc in zh_docs:

    pos_tags = [[token.word, token.flag] for token in zh_model.cut(doc)]

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))
vi_model = vi_pos_tag

vi_docs = ["Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò", "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét",

          "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?"]
for doc in vi_docs:

    pos_tags = vi_model(doc)

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))
ja_model = nagisa

ja_docs = ["Pythonで簡単に使えるツールです", "3月に見た「3月のライオン」",

           "新あたらしい記事きじを書かこうという気持きもちになるまで長ながい時間じかんがかかった。書かきたいことはたくさんあったけれど、息子むすこを産うんだ後あとは書かく時間じかんがあまりなかった"]
for doc in ja_docs:

    tagged = nagisa.tagging(doc)

    pos_tags = list(zip(tagged.words, tagged.postags))

    print("Document:\n{}\nPOS Tags:\n{}\n\n".format(doc, pos_tags))