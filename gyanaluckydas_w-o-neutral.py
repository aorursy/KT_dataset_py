import pandas as pd, numpy as np

import tensorflow as tf

from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

import math

print('TF version',tf.__version__)

MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

EPOCHS = 3

BATCH_SIZE = 32

PAD_ID = 1

SEED = 777

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/rcleaningv3/r-clean-train.csv').fillna('')

train.head()
train.loc[4842,'selected_text'] = "musical theatre actor' i wish"

train.loc[4891,'selected_text'] = "is an important date"
train = train[train['sentiment']!='neutral']

train = train.reset_index(drop=True)
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

count=0

for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = train.loc[k,'text']

    text2 = train.loc[k,'selected_text']

    text1 = " "+" ".join(text1.split())

    text2 = " ".join(text2.split())

    text1 = text1.lower()

    text2 = text2.lower()

    idx = text1.find(text2)

    if idx == -1:

        count=count+1

        print("idx   ::::::--->>", train.loc[k,'textID'])

        print("test1 ::::::--->>", text1)

        print("text2 ::::::--->>", text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask_t[k,:len(enc.ids)+3] = 1
import pickle



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean(loss)

    return loss
import tensorflow as tf

import re



class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    """Applies a warmup schedule on a given learning rate decay schedule."""



    def __init__(

        self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None,

    ):

        super().__init__()

        self.initial_learning_rate = initial_learning_rate

        self.warmup_steps = warmup_steps

        self.power = power

        self.decay_schedule_fn = decay_schedule_fn

        self.name = name



    def __call__(self, step):

        with tf.name_scope(self.name or "WarmUp") as name:

            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the

            # learning rate will be `global_step/num_warmup_steps * init_lr`.

            global_step_float = tf.cast(step, tf.float32)

            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)

            warmup_percent_done = global_step_float / warmup_steps_float

            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)

            return tf.cond(

                global_step_float < warmup_steps_float,

                lambda: warmup_learning_rate,

                lambda: self.decay_schedule_fn(step),

                name=name,

            )



    def get_config(self):

        return {

            "initial_learning_rate": self.initial_learning_rate,

            "decay_schedule_fn": self.decay_schedule_fn,

            "warmup_steps": self.warmup_steps,

            "power": self.power,

            "name": self.name,

        }







def create_optimizer(init_lr, num_train_steps, num_warmup_steps, end_lr=0.0, optimizer_type="adamw"):

    """Creates an optimizer with learning rate schedule."""

    # Implements linear decay of the learning rate.

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(

        initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=end_lr,

    )

    if num_warmup_steps:

        lr_schedule = WarmUp(

            initial_learning_rate=init_lr, decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps,

        )



    optimizer = AdamWeightDecay(

        learning_rate=lr_schedule,

        weight_decay_rate=0.01,

        beta_1=0.9,

        beta_2=0.999,

        epsilon=1e-6,

        exclude_from_weight_decay=["layer_norm", "bias"],

    )



    return optimizer







class AdamWeightDecay(tf.keras.optimizers.Adam):

    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the

  correct way of using L2 regularization/weight decay with Adam, since that will

  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with

  the m/v parameters. This is equivalent to adding the square of the weights to

  the loss with plain (non-momentum) SGD.

  """



    def __init__(

        self,

        learning_rate=0.001,

        beta_1=0.9,

        beta_2=0.999,

        epsilon=1e-7,

        amsgrad=False,

        weight_decay_rate=0.0,

        include_in_weight_decay=None,

        exclude_from_weight_decay=None,

        name="AdamWeightDecay",

        **kwargs

    ):

        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)

        self.weight_decay_rate = weight_decay_rate

        self._include_in_weight_decay = include_in_weight_decay

        self._exclude_from_weight_decay = exclude_from_weight_decay



    @classmethod

    def from_config(cls, config):

        """Creates an optimizer from its config with WarmUp custom object."""

        custom_objects = {"WarmUp": WarmUp}

        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)





    def _prepare_local(self, var_device, var_dtype, apply_state):

        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(

            self.weight_decay_rate, name="adam_weight_decay_rate"

        )



    def _decay_weights_op(self, var, learning_rate, apply_state):

        do_decay = self._do_use_weight_decay(var.name)

        if do_decay:

            return var.assign_sub(

                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],

                use_locking=self._use_locking,

            )

        return tf.no_op()



    def apply_gradients(self, grads_and_vars, name=None):

        grads, tvars = list(zip(*grads_and_vars))

        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name,)





    def _get_lr(self, var_device, var_dtype, apply_state):

        """Retrieves the learning rate with the given state."""

        if apply_state is None:

            return self._decayed_lr_t[var_dtype], {}



        apply_state = apply_state or {}

        coefficients = apply_state.get((var_device, var_dtype))

        if coefficients is None:

            coefficients = self._fallback_apply_state(var_device, var_dtype)

            apply_state[(var_device, var_dtype)] = coefficients



        return coefficients["lr_t"], dict(apply_state=apply_state)



    def _resource_apply_dense(self, grad, var, apply_state=None):

        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)

        decay = self._decay_weights_op(var, lr_t, apply_state)

        with tf.control_dependencies([decay]):

            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)



    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):

        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)

        decay = self._decay_weights_op(var, lr_t, apply_state)

        with tf.control_dependencies([decay]):

            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)



    def get_config(self):

        config = super().get_config()

        config.update({"weight_decay_rate": self.weight_decay_rate})

        return config





    def _do_use_weight_decay(self, param_name):

        """Whether to use L2 weight decay for `param_name`."""

        if self.weight_decay_rate == 0:

            return False



        if self._include_in_weight_decay:

            for r in self._include_in_weight_decay:

                if re.search(r, param_name) is not None:

                    return True



        if self._exclude_from_weight_decay:

            for r in self._exclude_from_weight_decay:

                if re.search(r, param_name) is not None:

                    return False

        return True







# Extracted from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py

class GradientAccumulator(object):

    """Gradient accumulation utility.

  When used with a distribution strategy, the accumulator should be called in a

  replica context. Gradients will be accumulated locally on each replica and

  without synchronization. Users should then call ``.gradients``, scale the

  gradients if required, and pass the result to ``apply_gradients``.

  """



    # We use the ON_READ synchronization policy so that no synchronization is

    # performed on assignment. To get the value, we call .value() which returns the

    # value on the current replica without synchronization.



    def __init__(self):

        """Initializes the accumulator."""

        self._gradients = []

        self._accum_steps = None



    @property

    def step(self):

        """Number of accumulated steps."""

        if self._accum_steps is None:

            self._accum_steps = tf.Variable(

                tf.constant(0, dtype=tf.int64),

                trainable=False,

                synchronization=tf.VariableSynchronization.ON_READ,

                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,

            )



        return self._accum_steps.value()



    @property

    def gradients(self):

        """The accumulated gradients on the current replica."""

        if not self._gradients:

            raise ValueError("The accumulator should be called first to initialize the gradients")

        return list(gradient.value() for gradient in self._gradients)



    def __call__(self, gradients):

        """Accumulates :obj:`gradients` on the current replica."""

        if not self._gradients:

            _ = self.step  # Create the step variable.

            self._gradients.extend(

                [

                    tf.Variable(

                        tf.zeros_like(gradient),

                        trainable=False,

                        synchronization=tf.VariableSynchronization.ON_READ,

                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,

                    )

                    for gradient in gradients

                ]

            )

        if len(gradients) != len(self._gradients):

            raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))



        for accum_gradient, gradient in zip(self._gradients, gradients):

            accum_gradient.assign_add(gradient)



        self._accum_steps.assign_add(1)



    def reset(self):

        """Resets the accumulated gradients on the current replica."""

        if not self._gradients:

            return

        self._accum_steps.assign(0)

        for gradient in self._gradients:

            gradient.assign(tf.zeros_like(gradient))
def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    x1 = CuDNNLSTM(150, return_sequences=True,name='lstm_layer')(x[0])

    x1 = CuDNNLSTM(96, return_sequences=True,name='lstm_layer2')(x1)

    x1 = tf.keras.layers.Dropout(0.1)(x1) 

    x1 = tf.keras.layers.Conv1D(1,1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = CuDNNLSTM(150, return_sequences=True,name='lstm_layer3')(x[0])

    x2 = CuDNNLSTM(96, return_sequences=True,name='lstm_layer4')(x2)

    x2 = tf.keras.layers.Dropout(0.1)(x2) 

    x2 = tf.keras.layers.Conv1D(1,1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    num_train_steps = math.ceil(13090/BATCH_SIZE)*3

    optimizer = create_optimizer(3e-5, num_train_steps, 0, end_lr=0.0, optimizer_type="adamw")

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
model,_ = build_model()
model.summary()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
jac = []; VER='v3'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()

        

#     sv = tf.keras.callbacks.ModelCheckpoint(

#        '%s-roberta-%i-checkpoint.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

#        save_weights_only=True, mode='auto', save_freq='epoch')

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    less_val= np.inf

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)



    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-2:b-1])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)