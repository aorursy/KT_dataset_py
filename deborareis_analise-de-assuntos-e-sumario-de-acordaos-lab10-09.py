#from google.colab import drive

#drive.mount('/content/drive',)
#from pandas import read_csv

#df = read_csv('drive/My Drive/ia/datasets/tcu-assunto-sumario.csv')
#df2 = read_csv('drive/My Drive/ia/datasets/tcu-assunto-sumario.csv')
# Pegando o dataset do output do kaggle

import pandas as pd

df = pd.read_csv('/kaggle/input/eda-acordaos-para-csv/tcu-assunto-sumario.csv')

df2 = pd.read_csv('/kaggle/input/eda-acordaos-para-csv/tcu-assunto-sumario.csv')
df2[['assunto','sumario']].groupby(['assunto'])['sumario'].count().nlargest(30)
df.head(5)
df.sample().values[0]
#quantidade de registros

df.shape
#dropa as linhas em que o sumário estão em branco

df.dropna(inplace=True)
df.shape
!pip install unidecode



from unidecode import unidecode

import re
def normaliza_assunto(x):

    return re.sub("[^a-zA-Z]"," ",unidecode(x).lower()).strip()
df['assunto'] = df.assunto.apply(normaliza_assunto)
#quantidades de labels

df['assunto'].unique().size
#listando os top 30 assuntos com mais ocorrência

df[['assunto','sumario']].groupby(['assunto'])['sumario'].count().nlargest(30)
df.head(5)
from fastai.text import *
#tokenizador

tokenizer = Tokenizer(lang='pt', n_cpus=8)

#vocabulario

with open('drive/My Drive/ia/models/itos.pkl', 'rb') as f:

    itos = pickle.load(f)

vocab = Vocab(itos)
from pathlib import Path

path = Path('drive/My Drive/ia/models')
path
bs = 24
train_bool = np.random.rand(len(df)) < 0.8
data_lm = TextLMDataBunch.from_df(path,

                                  train_df= df[train_bool],

                                  valid_df= df[~train_bool],

                                  tokenizer=tokenizer,

                                  vocab=vocab,

                                  text_cols=0,

                                  bs=bs,

                                  max_vocab=35000)
data_lm.save('data_lm')
data_lm = load_data(path, 'data_lm', bs=bs)
#learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)



config = awd_lstm_lm_config.copy()

config['n_hid'] = 1150

learn = language_model_learner(data_lm, AWD_LSTM, config=config,

                               pretrained_fnames=['model30k', 'itos'], 

                               drop_mult=0.3)



#learn_lm = language_model_learner(data_lm, arch=AWD_LSTM, pretrained_fnames=('model30k','itos'))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=4e-2)
learn.save_encoder('pretrained_encoder')
data_clas = TextClasDataBunch.from_df(path,

                                      train_df= df[train_bool],

                                      valid_df= df[~train_bool],

                                      tokenizer=tokenizer,

                                      text_cols=1,

                                      bs=24,

                                      vocab=vocab,

                                      max_vocab=35000,

                                      label_cols=0)
data_clas = (TextList.from_df(df, path, vocab=data_lm.vocab, cols='sumario')

    .split_by_rand_pct(0.1, seed=42)

    .label_from_df(cols='assunto')

    .databunch(bs=bs, num_workers=1, backwards=True))



#data_clas.save(f'{lang}_textlist_class_bwd')
print(type(data_clas.valid_ds.x))
len(data_clas.valid_ds.x)
clf = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
clf.fit_one_cycle(5, 1e-2)
clf.data.classes
pred = clf.predict('PEDIDO DE REEXAME. APOSENTADORIA. TEMPO DE SERVIÇO EXERCIDO EM ATIVIDADE RURAL E TEMPO DE ESTÁGIO PRESTADO JUNTO AO PROJETO RONDON, AMBOS COMPUTADOS PARA APOSENTADORIA, EM DESCONFORMIDADE COM A LEGISLAÇÃO PERTINENTE E COM O ENTENDIMENTO JURISPRUDENCIAL DO TCU. NÃO PROVIMENTO DO RECURSO. REVISÃO DE OFÍCIO DE OUTRO ATO. ABERTURA DE CONTRADITÓRIO.    1. O tempo de atividade rural somente pode ser computado para efeito de aposentadoria no serviço público se comprovados os recolhimentos ¿ em época própria, ou em momento posterior, de forma indenizada ¿ das respectivas contribuições previdenciárias.    2. É indevida a contagem, para efeitos de aposentadoria, de tempo referente a estágio prestado no Projeto Rondon, uma vez que não há vínculo empregatício de qualquer natureza nem contribuição para qualquer regime previdenciário.    3. Cabe revisão de ofício pelo Tribunal, mediante a oitiva do Ministério Público junto ao TCU, de ato de concessão julgado legal, no prazo de cinco anos do julgamento, se verificada violação da ordem jurídica'); pred
valid_df2= df[~train_bool]
valid_df2.sample().values[0]