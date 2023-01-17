%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.text import *
# bs=48

# bs=24

bs=128
import torch

torch.cuda.device_count()

torch.cuda.set_device(0)
# Utilizei para copiar o dataset da folha de SP da pasta input (read-only) para a pasta working

#import os

#!cp -r '../input/news-of-the-site-folhauol/' '../working'

#print(os.listdir("../working/news-of-the-site-folhauol"))
lang = 'pt'

name = f'{lang}News'

data_path = "../input/news-of-the-site-folhauol"

path = data_path

lm_fns = [f'{lang}_wt', f'{lang}_wt_vocab']
#Monta um databunch a partir do dataset da folha

#data = (TextList.from_csv(path,'articles.csv')

#            .split_by_rand_pct(0.1, seed=42)

#            .label_for_lm()           

#            .databunch(bs=bs, num_workers=1))

#len(data.vocab.itos),len(data.train_ds),len(data.valid_ds)
# Demonstra como salvar e carregar o databunch

#data.save(f'{lang}_databunch')

#data = load_data(path, f'{lang}_databunch', bs=bs)
#Cria um learn model a partir do databunch 

# Destaque para o parâmetro pretrained=False demonstrando que não existe um pré-treinamento

#learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False).to_fp16()
#Ajusta a learning rate, descongela as camadas e roda 10 vezes o primeiro ciclo 

#lr = 1e-2

#lr *= bs/48  # Scale learning rate by batch size



#learn.unfreeze()

#learn.fit_one_cycle(10, lr, moms=(0.8,0.7))
#Criamos a pasta models

#mdl_path = path+'/models'

#os.mkdir(mdl_path)

#print(os.listdir(path))
# Salvamos pt_wt na pasta models

#fileName = lm_fns[0]

#learn.to_fp32().save(fileName, with_opt=False)

#print(os.listdir(path+'/models'))
# Salvamos pt_wt_vocab.pkl na pasta models

#fileName = path +'/models/'+ lm_fns[1] + '.pkl'

#print (fileName)

#learn.data.vocab.save(fileName)

#print(os.listdir(path+'/models'))
#Carrega o dataset do olist da pasta input

from pandas import read_csv

df = read_csv('../input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
#transforma nas colunas que representam tempo string para timestamp 

df['review_creation_date'] =  pd.to_datetime(df['review_creation_date'])

df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'])
#transforma o review score em categorias

df['sentimento'] = pd.cut(df['review_score'],[0,2,3,5],labels=['negativo','neutro','positivo'])
#Cria um nova coluna 'text' a partir do título e da mensagem dos comentários

x = []



for i in zip(df['review_comment_title'],df['review_comment_message']):

  x.append(str(i[0]) + ' ' + str(i[1]))



df['text'] = x
#Retira alguns colunas que não serão utilizadas

df.drop(['review_id','order_id','review_score','review_answer_timestamp','review_comment_title','review_comment_message'],axis=1,inplace=True)
#Balanceamento dos comentários positivos, neutros e negativos

df_nn = df[(df.sentimento == 'negativo')|(df.sentimento == 'neutro')]

df_pos = df[(df.sentimento == 'positivo')&(df.text != 'nan nan')].reset_index(drop=True).iloc[np.random.permutation(np.arange(11500)),:]

df_final = pd.concat([df_nn,df_pos])

df_final = df_final[df_final.text != 'nan nan'].copy().reset_index(drop=True)
#Cria uma coluna text_tratado aplicando as substitucioes definadas no metodo pre_process

def pre_process(texto):

  texto = texto.replace(' nan ','')

  texto = texto.replace('nan ','')

  texto = texto.replace(' nan','')

  texto = re.sub('\r',' ',texto)

  texto = re.sub('\n',' ',texto)

  texto = re.sub(',',' ',texto)

  texto = re.sub('\.',' ',texto)

  texto = re.sub('\?',' ',texto)

  texto = re.sub('\!',' ',texto)

  texto = re.sub('\;',' ',texto)

  return texto



df_final['text_tratado'] = df_final.text.apply(pre_process)
# Cria a pasta models e transfere os arquivos que representam o modelo pré-treinado da folha de SP 

import os

print(os.listdir('../input/portuguesenewsulmfit/'))

os.mkdir('../working/models/')

os.system("mv ../input/portuguesenewsulmfit/* ../working/models/")

print(os.listdir('../working/models/'))
#Cria um databunch do dataset do Olist

path = '../working'

data_lm = (TextList.from_df(df_final, path, cols='text_tratado')

    .split_by_rand_pct(0.1, seed=42)

    .label_for_lm()           

    .databunch(bs=bs, num_workers=1))
#Cria um learner model a partir do databunch do Olist e do learner model anterior da Folha ( pretrained_fnames=lm_fns )

lm_fns = ['pt_wt', 'pt_wt_vocab']

learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=1.0)
#Ajusta a learning rate e roda 2 vezes o primeiro ciclo do modelo

lr = 1e-3

lr *= bs/48

learn_lm.fit_one_cycle(2, lr*10, moms=(0.8,0.7))
#Descongela o modelo e roda 8 vezes o primeiro ciclo 

# Destaque para a learning rate que não está multiplicada por 10 dessa vez

learn_lm.unfreeze()

learn_lm.fit_one_cycle(8, lr, moms=(0.8,0.7))
#Salva os arquivos que representam o learner model

learn_lm.save(f'{lang}fine_tuned')

learn_lm.save_encoder(f'{lang}fine_tuned_enc')
#Salva o vocabulario

fileName = '../working/models/'+ f'{lang}fine_tuned_vocab' + '.pkl'

learn_lm.data.vocab.save(fileName)

print(os.listdir('../working/models'))
print(path)
#Para criar o classificar passa como parâmetros: 

# dataframe do Olist

# caminho para a pasta onde se encontra a pasta models

# vocabulario ja treinado com o Olist

# Coluna que sera classificada no caso 'text_tratado'

# Coluna que representa a classificacao no caso coluna 'sentimento'

data_clas = (TextList.from_df(df_final, path, vocab=data_lm.vocab, cols='text_tratado')

    .split_by_rand_pct(0.1, seed=42)

    .label_from_df(cols='sentimento')

    .databunch(bs=bs, num_workers=1))
#Demonstra como salvar o classificador e depois carregá-lo novamente

data_clas.save(f'{lang}_textlist_class')

data_clas = load_data(path, f'{lang}_textlist_class', bs=bs, num_workers=1)
#Define o metodo que sera utilizado para comparar a precision and recall

from sklearn.metrics import f1_score

@np_func

# No notebook nn-vietnamese.ipynb não foi passado valor para o parametro average 

# Portanto estava sendo utilizado o valor default "binary", 

# porém como a label é multiclass 'positivo', 'neutro' e 'negativo' nao podia ser "binary"

# As opções disponíveis para esse parametro estao descritas aqui 

#https://en.wikipedia.org/wiki/F1_score



def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='micro')
#Cria um modelo a partir do classificador

#Nao entendi pq tem que se conectar na internet para baixar um arquivo wt103

learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()

learn_c.load_encoder(f'{lang}fine_tuned_enc')

learn_c.freeze()
#Redefine a learning rate e roda duas vezes o primeiro ciclo 

lr=2e-2

lr *= bs/48

learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))
learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7))
learn_c.freeze_to(-3)

learn_c.fit_one_cycle(2, slice(lr/2/(2.6**4),lr/2), moms=(0.8,0.7))
learn_c.unfreeze()

learn_c.fit_one_cycle(1, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))
learn_c.save(f'{lang}clas')
bs=128

lang = 'pt'

lm_fns = [f'{lang}_wt_bwd', f'{lang}_wt_vocab_bwd']

path = "../working/"
!cp '../input/news-of-the-site-folhauol/articles.csv' '../working'

print(os.listdir("../working/"))
#Monta um databunch a partir do dataset da folha

# Só que agora de trás para frente

data = (TextList.from_csv(path,'articles.csv')

            .split_by_rand_pct(0.1, seed=42)

            .label_for_lm()           

            .databunch(bs=bs, num_workers=1, backwards=True))

len(data.vocab.itos),len(data.train_ds),len(data.valid_ds)
data.save(f'{lang}_databunch_bwd')
data = load_data(path, f'{lang}_databunch_bwd', bs=bs, backwards=True)
#Cria um learn model a partir do databunch 

# Destaque para o parâmetro pretrained=False demonstrando que não existe um pré-treinamento

learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False).to_fp16()
#Ajusta a learning rate, interessante que na primeira vez

# foi utilizado lr = 1e-2

lr = 3e-3

lr *= bs/48  # Scale learning rate by batch size
learn.unfreeze()

learn.fit_one_cycle(10, lr, moms=(0.8,0.7))
# Cria a pasta models, se ela já não existir

mdl_path = path+'models'

os.mkdir(mdl_path)

print(os.listdir(path))
# Salvamos pt_wt na pasta models

fileName = lm_fns[0]

learn.to_fp32().save(fileName, with_opt=False)

print(os.listdir(path+'/models'))
# Salvamos pt_wt_vocab_bwd.pkl na pasta models

fileName = path +'/models/'+ lm_fns[1] + '.pkl'

#print (fileName)

learn.data.vocab.save(fileName)

print(os.listdir(path+'/models'))
#Cria um databunch do dataset do Olist

#Destaque para o backwards=True para dizer que agora é de trás para frente 

path = '../working'

data_lm = (TextList.from_df(df_final, path, cols='text_tratado')

    .split_by_rand_pct(0.1, seed=42)

    .label_for_lm()           

    .databunch(bs=bs, num_workers=1,backwards=True))
#Cria um learner model a partir do databunch do Olist e do learner model BWD da Folha ( pretrained_fnames=lm_fns )

lang = 'pt'

lm_fns = [f'{lang}_wt_bwd', f'{lang}_wt_vocab_bwd']

learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=1.0)
#Ajusta a learning rate e roda 2 vezes o primeiro ciclo do modelo

lr = 1e-3

lr *= bs/48

learn_lm.fit_one_cycle(2, lr*10, moms=(0.8,0.7))
#Descongela o modelo e roda 8 vezes o primeiro ciclo 

# Destaque para a learning rate que não está multiplicada por 10 dessa vez

learn_lm.unfreeze()

learn_lm.fit_one_cycle(8, lr, moms=(0.8,0.7))
#Salva os arquivos que representam o learner model BWD

learn_lm.save(f'{lang}fine_tuned_bwd')

learn_lm.save_encoder(f'{lang}fine_tuned_enc_bwd')
#Salva o vocabulario

fileName = '../working/models/'+ f'{lang}fine_tuned_vocab_bwd' + '.pkl'

learn_lm.data.vocab.save(fileName)

print(os.listdir('../working/models'))
#Para criar o classificar passa como parâmetros: 

# dataframe do Olist

# caminho para a pasta onde se encontra a pasta models

# vocabulario ja treinado com o Olist

# Coluna que sera classificada no caso 'text_tratado'

# Coluna que representa a classificacao no caso coluna 'sentimento'

data_clas = (TextList.from_df(df_final, path, vocab=data_lm.vocab, cols='text_tratado')

    .split_by_rand_pct(0.1, seed=42)

    .label_from_df(cols='sentimento')

    .databunch(bs=bs, num_workers=1,backwards=True))
#Demonstra como salvar o classificador e depois carregá-lo novamente

data_clas.save(f'{lang}_textlist_class_bwd')

data_clas = load_data(path, f'{lang}_textlist_class', bs=bs, num_workers=1,backwards=True)
#Cria um modelo a partir do classificador BWD

#Nao entendi pq tem que se conectar na internet para baixar um arquivo wt103

learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()

learn_c.load_encoder(f'{lang}fine_tuned_enc')

learn_c.freeze()
#Redefine a learning rate e roda duas vezes o primeiro ciclo 

lr=2e-2

lr *= bs/48

learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))
learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7))
learn_c.freeze_to(-3)

learn_c.fit_one_cycle(2, slice(lr/2/(2.6**4),lr/2), moms=(0.8,0.7))
learn_c.unfreeze()

learn_c.fit_one_cycle(1, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))
learn_c.save(f'{lang}clas_bwd')
print(os.listdir('../working'))
data_clas = load_data(path, f'{lang}_textlist_class', bs=bs, num_workers=1)

learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()

learn_c.load(f'{lang}clas', purge=False);
preds,targs = learn_c.get_preds(ordered=True)

accuracy(preds,targs),f1(preds,targs)
#utiliza a técnica de ir de trás para frente backwards (bwd)

data_clas_bwd = load_data(path, f'{lang}_textlist_class_bwd', bs=bs, num_workers=1, backwards=True)

learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()

learn_c_bwd.load(f'{lang}clas_bwd', purge=False);
preds_b,targs_b = learn_c_bwd.get_preds(ordered=True)

accuracy(preds_b,targs_b),f1(preds_b,targs_b)
preds_avg = (preds+preds_b)/2
accuracy(preds_avg,targs_b),f1(preds_avg,targs_b)