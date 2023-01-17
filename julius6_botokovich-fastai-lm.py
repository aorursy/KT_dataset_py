from fastai.text import *



import numpy as np



import pandas as pd





config = awd_lstm_lm_config.copy()

config['n_hid'] = 300

#config['emb_sz'] = 500



epochs = 40

bs = 10

lr = .001





def collect_sequences(filename):

    sequences = np.load(filename, allow_pickle=True)

    return sequences





def truncate_sequences(sequences):

    seq_length = len(min(sequences, key=len))

    sequences = [sequence[:seq_length] for sequence in sequences]

    return sequences





def convert_lists_to_strings(sequences):

    sequences = [list(map(str, sequence)) for sequence in sequences]

    str_seqs = list()

    for sequence in sequences:

        str_seqs.append(' '.join(sequence))

    return str_seqs





def create_train_val_sets(str_seqs):

    train_val_ind = int(len(str_seqs) * .8)

    train, val = pd.DataFrame(), pd.DataFrame()

    train['seq'] = str_seqs[:train_val_ind]

    val['seq'] = str_seqs[train_val_ind:]

    return train, val





def create_databunch(train, val, bs):

    tok = Tokenizer(pre_rules=list(), post_rules=list(), special_cases=[UNK])

    data_lm = TextLMDataBunch.from_df("data", train_df=train, valid_df=val, text_cols='seq', tokenizer=tok, bs=bs)

    return data_lm





def create_model_and_train(data_lm, config, epochs):

    learner = language_model_learner(data=data_lm, arch=AWD_LSTM, pretrained=False, config=config)

    learner.fit(epochs=epochs, lr=lr)

    learner.save("awd_lstm")

    return learner





def predict(learner, start, length):

    start = "xxbos " + start

    return learner.predict(start, n_words=length)





sequences = np.load("../input/note_sequences.npy")

train, val = create_train_val_sets(sequences)

data_lm = create_databunch(train, val, bs)

learner = create_model_and_train(data_lm, config, epochs)
seq_start = "xxbos 12 24 28 31 step step step step step step step step step step step step"

model = "../working/data/models/awd_lstm"

num_seq = 10

len_measures = 100





def predict(learner, num_seq, seq_start, len_measures):

    seq_len = len_measures * 12

    sequences = list()

    

    for i in range(num_seq):

        np.random.seed(i)

        seq = learner.predict(seq_start, seq_len)

        sequences.append(seq)

    

    np.save("sequences", sequences)

    

    

predict(learner, num_seq, seq_start, len_measures)

        
import base64



from IPython.display import HTML





def create_download_link(file, filename="sequences"):

    seq = np.load(file)

    csv = pd.DataFrame(seq).to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title="Download file",filename=filename)

    return HTML(html)





create_download_link("../working/sequences.npy")