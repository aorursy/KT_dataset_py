import os, sys, time, random
import h5py
import numpy as np
def FreqDict2List(dt):
	return sorted(dt.items(), key=lambda d:d[-1], reverse=True)

def LoadList(fn):
	with open(fn, encoding = "utf-8") as fin:
		st = list(ll for ll in fin.read().split('\n') if ll != "")
	return st

def LoadCSV(fn):
	ret = []
	with open(fn, encoding='utf-8') as fin:
		for line in fin:
			lln = line.rstrip('\r\n').split('\t')
			ret.append(lln)
	return ret

def LoadCSVg(fn):
	with open(fn, encoding='utf-8') as fin:
		for line in fin:
			lln = line.rstrip('\r\n').split('\t')
			yield lln

def SaveList(st, ofn):
	with open(ofn, "w", encoding = "utf-8") as fout:
		for k in st:
			fout.write(str(k) + "\n")
class TokenList:
	def __init__(self, token_list):
		self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
		self.t2id = {v:k for k,v in enumerate(self.id2t)}
	def id(self, x):	return self.t2id.get(x, 1)
	def token(self, x):	return self.id2t[x]
	def num(self):		return len(self.id2t)
	def startid(self):  return 2
	def endid(self):    return 3
	
def pad_to_longest(xs, tokens, max_len=999):
	longest = min(len(max(xs, key=len))+2, max_len)
	X = np.zeros((len(xs), longest), dtype='int32')
	X[:,0] = tokens.startid()
	for i, x in enumerate(xs):
		x = x[:max_len-2]
		for j, z in enumerate(x):
			X[i,1+j] = tokens.id(z)
		X[i,1+len(x)] = tokens.endid()
	return X

def MakeS2SDict(fn=None, min_freq=5, delimiter=' ', dict_file=None):
	if dict_file is not None and os.path.exists(dict_file):
		print('loading', dict_file)
		lst = LoadList(dict_file)
		midpos = lst.index('<@@@>')
		itokens = TokenList(lst[:midpos])
		otokens = TokenList(lst[midpos+1:])
		return itokens, otokens
	data = LoadCSV(fn)
	wdicts = [{}, {}]
	for ss in data:
		for seq, wd in zip(ss, wdicts):
			for w in seq.split(delimiter): 
				wd[w] = wd.get(w, 0) + 1
	wlists = []
	for wd in wdicts:	
		wd = FreqDict2List(wd)
		wlist = [x for x,y in wd if y >= min_freq]
		wlists.append(wlist)
	print('seq 1 words:', len(wlists[0]))
	print('seq 2 words:', len(wlists[1]))
	itokens = TokenList(wlists[0])
	otokens = TokenList(wlists[1])
	if dict_file is not None:
		SaveList(wlists[0]+['<@@@>']+wlists[1], dict_file)
	return itokens, otokens

def MakeS2SData(fn=None, itokens=None, otokens=None, delimiter=' ', h5_file=None, max_len=200):
	if h5_file is not None and os.path.exists(h5_file):
		print('loading', h5_file)
		with h5py.File(h5_file) as dfile:
			X, Y = dfile['X'][:], dfile['Y'][:]
		return X, Y
	data = LoadCSVg(fn)
	Xs = [[], []]
	for ss in data:
		for seq, xs in zip(ss, Xs):
			xs.append(list(seq.split(delimiter)))
	X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
	if h5_file is not None:
		with h5py.File(h5_file, 'w') as dfile:
			dfile.create_dataset('X', data=X)
			dfile.create_dataset('Y', data=Y)
	return X, Y

def S2SDataGenerator(fn, itokens, otokens, batch_size=64, delimiter=' ', max_len=999):
	Xs = [[], []]
	while True:
		for ss in LoadCSVg(fn):
			for seq, xs in zip(ss, Xs):
				xs.append(list(seq.split(delimiter)))
			if len(Xs[0]) >= batch_size:
				X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
				yield [X, Y], None
				Xs = [[], []]

itokens, otokens = MakeS2SDict('../input/en2de.s2s.txt', 6, dict_file='en2de_word.txt')
X, Y = MakeS2SData('../input/en2de.s2s.txt', itokens, otokens, h5_file='en2de.h5')
print(X.shape, Y.shape)
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf
itokens, otokens = MakeS2SDict('../input/en2de.s2s.txt', dict_file='./en2de_word.txt')
Xtrain, Ytrain = MakeS2SData('../input/en2de.s2s.txt', itokens, otokens, h5_file='./en2de.h5')
Xvalid, Yvalid = MakeS2SData('../input/en2de.s2s.valid.txt', itokens, otokens, h5_file='//en2de.valid.h5')
print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)
class Encoder():
    def __init__(self, i_token_num, latent_dim, layers=3):
        self.emb_layer = Embedding(i_token_num, latent_dim, mask_zero=True)
        cells = [GRUCell(latent_dim) for _ in range(layers)]
        self.rnn_layer = RNN(cells, return_state=True)
    def __call__(self, x):
        x = self.emb_layer(x)
        xh = self.rnn_layer(x)
        x, h = xh[0], xh[1:]
        return x, h

class Decoder():
    def __init__(self, o_token_num, latent_dim, layers=3):
        self.emb_layer = Embedding(o_token_num, latent_dim, mask_zero=True)
        cells = [GRUCell(latent_dim) for _ in range(layers)]
        self.rnn_layer = RNN(cells, return_sequences=True, return_state=True)
        self.out_layer = Dense(o_token_num)
    def __call__(self, x, state):
        x = self.emb_layer(x)
        xh = self.rnn_layer(x, initial_state=state)
        x, h = xh[0], xh[1:]
        x = TimeDistributed(self.out_layer)(x)
        return x, h
class RNNSeq2Seq():
    def __init__(self, i_tokens, o_tokens, latent_dim, layers=3):
        self.i_tokens = i_tokens
        self.o_tokens = o_tokens

        encoder_inputs = Input(shape=(None,), dtype='int32')
        decoder_inputs = Input(shape=(None,), dtype='int32')

        encoder = Encoder(i_tokens.num(), latent_dim, layers)
        decoder = Decoder(o_tokens.num(), latent_dim, layers)

        encoder_outputs, encoder_states = encoder(encoder_inputs)

        dinputs = Lambda(lambda x:x[:,:-1])(decoder_inputs)
        dtargets = Lambda(lambda x:x[:,1:])(decoder_inputs)

        decoder_outputs, decoder_state_h = decoder(dinputs, encoder_states)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([decoder_outputs, dtargets])
        self.ppl = Lambda(K.exp)(loss)
        self.accu = Lambda(get_accu)([decoder_outputs, dtargets])

        self.model = Model([encoder_inputs, decoder_inputs], loss)
        self.model.add_loss([K.mean(loss)])

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_states_inputs = [Input(shape=(latent_dim,)) for _ in range(3)]

        decoder_outputs, decoder_states = decoder(decoder_inputs, decoder_states_inputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def compile(self, optimizer):
        self.model.compile(optimizer, None)
        self.model.metrics_names.append('ppl')
        self.model.metrics_tensors.append(self.ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def decode_sequence(self, input_seq, delimiter=''):
        input_mat = np.zeros((1, len(input_seq)+3))
        input_mat[0,0] = self.i_tokens.id('<S>')
        for i, z in enumerate(input_seq): input_mat[0,1+i] = self.i_tokens.id(z)
        input_mat[0,len(input_seq)+1] = self.i_tokens.id('</S>')

        state_value = self.encoder_model.predict_on_batch(input_mat)
        target_seq = np.zeros((1, 1))
        target_seq[0,0] = self.o_tokens.id('<S>')

        decoded_tokens = []
        while True:
            output_tokens_and_h = self.decoder_model.predict_on_batch([target_seq] + state_value)
            output_tokens, h = output_tokens_and_h[0], output_tokens_and_h[1:]
            sampled_token_index = np.argmax(output_tokens[0,-1,:])
            sampled_token = self.o_tokens.token(sampled_token_index)
            decoded_tokens.append(sampled_token)
            if sampled_token == '</S>' or len(decoded_tokens) > 50: break
            target_seq = np.zeros((1, 1))
            target_seq[0,0] = sampled_token_index
            state_value = h
        return delimiter.join(decoded_tokens[:-1])
s2s_model = RNNSeq2Seq(itokens,otokens, 256)
s2s_model.compile(Adam(0.01, 0.9, 0.98, epsilon=1e-9))
s2s_model.model.summary() 
model_saver = ModelCheckpoint("./s2s_model.h5", save_best_only=True, save_weights_only=True,monitor='val_accu',mode='max',verbose=1)

s2s_model.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None),callbacks=[model_saver],verbose=1)
s2s_model.model.load_weights("./s2s_model.h5")
score = s2s_model.model.evaluate([Xvalid, Yvalid], None,batch_size = 32)
print(score)
from keras.initializers import *
class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
									 initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
									initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

class ScaledDotProductAttention():
	def __init__(self, d_model, attn_dropout=0.1):
		self.temper = np.sqrt(d_model)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
		self.mode = mode
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = LayerNormalization() if use_norm else None
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, d_k])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		if not self.layer_norm: return outputs, attn
		outputs = Add()([outputs, q])
		return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.enc_att_layer  = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
		output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
		output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

def GetPadMask(q, k):
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask

class Encoder():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
				layers=6, dropout=0.1, word_emb=None, pos_emb=None):
		self.emb_layer = word_emb
		self.pos_layer = pos_emb
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
	def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
		x = self.emb_layer(src_seq)
		if src_pos is not None:
			pos = self.pos_layer(src_pos)
			x = Add()([x, pos])
		if return_att: atts = []
		mask = Lambda(lambda x:GetPadMask(x, x))(src_seq)
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x

class Decoder():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
			  layers=6, dropout=0.1, word_emb=None, pos_emb=None):
		self.emb_layer = word_emb
		self.pos_layer = pos_emb
		self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
	def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
		dec = self.emb_layer(tgt_seq)
		pos = self.pos_layer(tgt_pos)
		x = Add()([dec, pos])

		self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
		self_sub_mask = Lambda(GetSubMask)(tgt_seq)
		self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		
		enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])

		if return_att: self_atts, enc_atts = [], []
		for dec_layer in self.layers[:active_layers]:
			x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
			if return_att: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		return (x, self_atts, enc_atts) if return_att else x

class Transformer:
	def __init__(self, i_tokens, o_tokens, len_limit, d_model=256, \
			  d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, \
			  share_word_emb=False):
		self.i_tokens = i_tokens
		self.o_tokens = o_tokens
		self.len_limit = len_limit
		self.src_loc_info = True
		self.d_model = d_model
		self.decode_model = None
		d_emb = d_model

		pos_emb = Embedding(len_limit, d_emb, trainable=False, \
						   weights=[GetPosEncodingMatrix(len_limit, d_emb)])

		i_word_emb = Embedding(i_tokens.num(), d_emb)
		if share_word_emb: 
			assert i_tokens.num() == o_tokens.num()
			o_word_emb = i_word_emb
		else: o_word_emb = Embedding(o_tokens.num(), d_emb)

		self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
							word_emb=i_word_emb, pos_emb=pos_emb)
		self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
							word_emb=o_word_emb, pos_emb=pos_emb)
		self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask

	def compile(self, optimizer='adam', active_layers=999):
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_seq_input = Input(shape=(None,), dtype='int32')

		src_seq = src_seq_input
		tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input)
		tgt_true = Lambda(lambda x:x[:,1:])(tgt_seq_input)

		src_pos = Lambda(self.get_pos_seq)(src_seq)
		tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
		if not self.src_loc_info: src_pos = None

		enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
		dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)	
		final_output = self.target_layer(dec_output)

		def get_loss(args):
			y_pred, y_true = args
			y_true = tf.cast(y_true, 'int32')
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss

		def get_accu(args):
			y_pred, y_true = args
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
			return K.mean(corr)
				
		loss = Lambda(get_loss)([final_output, tgt_true])
		self.ppl = Lambda(K.exp)(loss)
		self.accu = Lambda(get_accu)([final_output, tgt_true])

		self.model = Model([src_seq_input, tgt_seq_input], loss)
		self.model.add_loss([loss])
		self.output_model = Model([src_seq_input, tgt_seq_input], final_output)
		
		self.model.compile(optimizer, None)
		self.model.metrics_names.append('ppl')
		self.model.metrics_tensors.append(self.ppl)
		self.model.metrics_names.append('accu')
		self.model.metrics_tensors.append(self.accu)

	def make_src_seq_matrix(self, input_seq):
		src_seq = np.zeros((1, len(input_seq)+3), dtype='int32')
		src_seq[0,0] = self.i_tokens.startid()
		for i, z in enumerate(input_seq): src_seq[0,1+i] = self.i_tokens.id(z)
		src_seq[0,len(input_seq)+1] = self.i_tokens.endid()
		return src_seq

	def decode_sequence(self, input_seq, delimiter=''):
		src_seq = self.make_src_seq_matrix(input_seq)
		decoded_tokens = []
		target_seq = np.zeros((1, self.len_limit), dtype='int32')
		target_seq[0,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			output = self.output_model.predict_on_batch([src_seq, target_seq])
			sampled_index = np.argmax(output[0,i,:])
			sampled_token = self.o_tokens.token(sampled_index)
			decoded_tokens.append(sampled_token)
			if sampled_index == self.o_tokens.endid(): break
			target_seq[0,i+1] = sampled_index
		return delimiter.join(decoded_tokens[:-1])

	def make_fast_decode_model(self):
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_seq_input = Input(shape=(None,), dtype='int32')
		src_seq = src_seq_input
		tgt_seq = tgt_seq_input

		src_pos = Lambda(self.get_pos_seq)(src_seq)
		tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
		if not self.src_loc_info: src_pos = None
		enc_output = self.encoder(src_seq, src_pos)
		self.encode_model = Model(src_seq_input, enc_output)

		enc_ret_input = Input(shape=(None, self.d_model))
		dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_ret_input)	
		final_output = self.target_layer(dec_output)
		self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)
		
		self.encode_model.compile('adam', 'mse')
		self.decode_model.compile('adam', 'mse')

	def decode_sequence_fast(self, input_seq, delimiter=''):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seq)
		enc_ret = self.encode_model.predict_on_batch(src_seq)

		decoded_tokens = []
		target_seq = np.zeros((1, self.len_limit), dtype='int32')
		target_seq[0,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			output = self.decode_model.predict_on_batch([src_seq,enc_ret,target_seq])
			sampled_index = np.argmax(output[0,i,:])
			sampled_token = self.o_tokens.token(sampled_index)
			decoded_tokens.append(sampled_token)
			if sampled_index == self.o_tokens.endid(): break
			target_seq[0,i+1] = sampled_index
		return delimiter.join(decoded_tokens[:-1])

	def beam_search(self, input_seq, topk=5, delimiter=''):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seq)
		src_seq = src_seq.repeat(topk, 0)
		enc_ret = self.encode_model.predict_on_batch(src_seq)

		final_results = []
		decoded_tokens = [[] for _ in range(topk)]
		decoded_logps = [0] * topk
		lastk = 1
		target_seq = np.zeros((topk, self.len_limit), dtype='int32')
		target_seq[:,0] = self.o_tokens.startid()
		for i in range(self.len_limit-1):
			if lastk == 0 or len(final_results) > topk * 3: break
			output = self.decode_model.predict_on_batch([src_seq,enc_ret,target_seq])
			output = np.exp(output[:,i,:])
			output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)
			cands = []
			for k, wprobs in zip(range(lastk), output):
				if target_seq[k,i] == self.o_tokens.endid(): continue
				wsorted = sorted(list(enumerate(wprobs)), key=lambda x:x[-1], reverse=True)
				for wid, wp in wsorted[:topk]: 
					cands.append( (k, wid, decoded_logps[k]+wp) )
			cands.sort(key=lambda x:x[-1], reverse=True)
			cands = cands[:topk]
			backup_seq = target_seq.copy()
			for kk, zz in enumerate(cands):
				k, wid, wprob = zz
				target_seq[kk,] = backup_seq[k]
				target_seq[kk,i+1] = wid
				decoded_logps[kk] = wprob
				decoded_tokens.append(decoded_tokens[k] + [self.o_tokens.token(wid)]) 
				if wid == self.o_tokens.endid(): final_results.append( (decoded_tokens[k], wprob) )
			decoded_tokens = decoded_tokens[topk:]
			lastk = len(cands)
		final_results = [(x,y/(len(x)+1)) for x,y in final_results]
		final_results.sort(key=lambda x:x[-1], reverse=True)
		final_results = [(delimiter.join(x),y) for x,y in final_results]
		return final_results

class LRSchedulerPerStep(Callback):
	def __init__(self, d_model, warmup=4000):
		self.basic = d_model**-0.5
		self.warm = warmup**-1.5
		self.step_num = 0
	def on_batch_begin(self, batch, logs = None):
		self.step_num += 1
		lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
		K.set_value(self.model.optimizer.lr, lr)

class LRSchedulerPerEpoch(Callback):
	def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
		self.basic = d_model**-0.5
		self.warm = warmup**-1.5
		self.num_per_epoch = num_per_epoch
		self.step_num = 1
	def on_epoch_begin(self, epoch, logs = None):
		self.step_num += self.num_per_epoch
		lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
		K.set_value(self.model.optimizer.lr, lr)

class AddPosEncoding:
	def __call__(self, x):
		_, max_len, d_emb = K.int_shape(x)
		pos = GetPosEncodingMatrix(max_len, d_emb)
		x = Lambda(lambda x:x+pos)(x)
		return x


add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])

class QANet_ConvBlock:
	def __init__(self, dim, n_conv=2, kernel_size=7, dropout=0.1):
		self.convs = [SeparableConv1D(dim, kernel_size, activation='relu', padding='same') for _ in range(n_conv)]
		self.norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		for i in range(len(self.convs)):
			z = self.norm(x)
			if i % 2 == 0: z = self.dropout(z)
			z = self.convs[i](z)
			x = add_layer([x, z])
		return x

class QANet_Block:
	def __init__(self, dim, n_head, n_conv, kernel_size, dropout=0.1, add_pos=True):
		self.conv = QANet_ConvBlock(dim, n_conv=n_conv, kernel_size=kernel_size, dropout=dropout)
		self.self_att = MultiHeadAttention(n_head=n_head, d_model=dim, 
									 d_k=dim//n_head, d_v=dim//n_head, 
									 dropout=dropout, use_norm=False)
		self.feed_forward = PositionwiseFeedForward(dim, dim, dropout=dropout)
		self.norm = LayerNormalization()
		self.add_pos = add_pos
	def __call__(self, x, mask):
		if self.add_pos: x = AddPosEncoding()(x)
		x = self.conv(x)
		z = self.norm(x)
		z, _ = self.self_att(z, z, z, mask)
		x = add_layer([x, z])
		z = self.norm(x)
		z = self.feed_forward(z)
		x = add_layer([x, z])
		return x

class QANet_Encoder:
	def __init__(self, dim=128, n_head=8, n_conv=2, n_block=1, kernel_size=7, dropout=0.1, add_pos=True):
		self.dim = dim
		self.n_block = n_block
		self.conv_first = SeparableConv1D(dim, 1, padding='same')
		self.enc_block = QANet_Block(dim, n_head=n_head, n_conv=n_conv, kernel_size=kernel_size, 
								dropout=dropout, add_pos=add_pos)
	def __call__(self, x, mask):
		if K.int_shape(x)[-1] != self.dim:
			x = self.conv_first(x)
		for i in range(self.n_block):
			x = self.enc_block(x, mask)
		return x
d_model = 512
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
				   n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)

mfile = './en2de.model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.01, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()

s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None), callbacks=[lr_scheduler, model_saver])
score = s2s.model.evaluate([Xvalid, Yvalid], None,batch_size = 32)
print(score)
 