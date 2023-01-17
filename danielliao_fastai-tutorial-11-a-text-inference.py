import fastai.text as ft
imdb = ft.untar_data(ft.URLs.IMDB_SAMPLE); imdb
imdb.ls()
data_lm = (ft.TextList.from_csv(imdb, 'texts.csv', cols='text')

                   .random_split_by_pct()

                   .label_for_lm()

                   .databunch())

data_lm.path
data_lm.save(); data_lm.path
data_lm.show_batch()
learn = ft.language_model_learner(data_lm, ft.AWD_LSTM)

learn.fit_one_cycle(2, 1e-2)

learn.save('mini_train_lm')

learn.save_encoder('mini_train_encoder')
learn.show_results()
learn = ft.language_model_learner(data_lm, ft.AWD_LSTM, pretrained=False).load('mini_train_lm', with_opt=False);
learn.export(fname = 'export_lm.pkl')
learn = ft.load_learner(imdb, fname = 'export_lm.pkl')
learn.predict('This is a simple test of', n_words=20)
learn.beam_search('This is a simple test of', n_words=20, beam_sz=200)
## Classification
data_clas = (ft.TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=data_lm.vocab)

                   .split_from_df(col='is_valid')

                   .label_from_df(cols='label')

                   .databunch(bs=42))
data_clas.show_batch()
learn = ft.text_classifier_learner(data_clas, ft.AWD_LSTM)

learn.load_encoder('mini_train_encoder')

learn.fit_one_cycle(2, slice(1e-3,1e-2))

learn.save('mini_train_clas')
learn = ft.text_classifier_learner(data_clas, ft.AWD_LSTM, pretrained=False).load('mini_train_clas', with_opt=False);

learn.export(fname = 'export_clas.pkl')
learn = ft.load_learner(imdb, fname = 'export_clas.pkl')
learn.predict('I really loved that movie!')