from fastai.datasets import untar_data, URLs, download_data



imdb_path = untar_data(URLs.IMDB)

imdb_path.ls()
from fastai.text import TextList



imdb_data_lm = (TextList.from_folder(imdb_path)

                # 过滤我们需要的文件夹

                .filter_by_folder(include=['unsup'])

                # 随机划分一定百分比数据到验证集

                .split_by_rand_pct(0.1)

                # 指定标签类型为语言模型

                .label_for_lm()

                # 处理成 DataBunch

                .databunch())

imdb_data_lm
from fastai.text.models import AWD_LSTM

from fastai.text import language_model_learner



learner_lm = language_model_learner(imdb_data_lm, AWD_LSTM)

learner_lm.fit(1)

learner_lm.save_encoder('fine_tuned_encoder')
TEXT = "I loved that movie because"  # 指定句子的开头

N_WORDS = 40  # 指定句子长度

N_SENTENCES = 5  # 指定预测句子数量



print("\n".join(learner_lm.predict(TEXT, N_WORDS, temperature=0.75)

                for _ in range(N_SENTENCES)))
imdb_data = (TextList.from_folder(imdb_path, vocab=imdb_data_lm.vocab)

             # 过滤我们需要的两个文件夹

             .filter_by_folder(include=['train', 'test'])

             # 0.1 划为验证集

             .split_by_rand_pct(0.1)

             # 指定标签类型

             .label_from_folder(classes=['neg', 'pos'])

             # 处理成 DataBunch

             .databunch())

imdb_data
from fastai.text import text_classifier_learner



# 定义 Learner

learner_ft = text_classifier_learner(imdb_data, AWD_LSTM)

learner_ft.load_encoder('fine_tuned_encoder')

learner_ft.fit(3)