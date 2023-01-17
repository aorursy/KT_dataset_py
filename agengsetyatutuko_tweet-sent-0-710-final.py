# !pip install transformers
# !python -m pip install --upgrade pandas
# # !pip install fairseq torchviz
import pandas as pd
import os
import re
data_path = '/kaggle/input/tweet-sentiment-extraction'
config_path = '/kaggle/input/submission'
train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_dataframe = pd.read_csv(os.path.join(data_path, 'test.csv'))
train_dataframe.head(5)
test_dataframe.head(5)
import torch
from transformers.tokenization_roberta import RobertaTokenizerFast
import re
import numpy as np

rob_distil_tozer = RobertaTokenizerFast(vocab_file=os.path.join(config_path, 'roberta-base-vocab.json'),
                                        merges_file=os.path.join(config_path, 'roberta-base-merges.txt'),
                                        add_prefix_space=True)
def robdist_tokenize(tweet, tozer = None, offset_map=True):
  if not isinstance(tweet, str):
    raise TypeError
  if tozer is not None and not isinstance(tozer, RobertaTokenizerFast):
    raise TypeError

  temp = tozer.encode_plus(tweet, return_offsets_mapping=offset_map, add_special_tokens=True) if tozer is not None else rob_distil_tozer.encode_plus(tweet, return_offsets_mapping=offset_map, add_special_tokens=True)

  return {'input_ids': torch.tensor(temp['input_ids']),
          'att_mask': torch.tensor(temp['attention_mask']),
          'offsets': temp['offset_mapping']}


def robdist_detokenize(token_ids, tozer = None, skip_special_tok=True):
  if not isinstance(token_ids, torch.LongTensor) and not isinstance(token_ids, torch.cuda.LongTensor):
    raise TypeError('First argument is not type tensor.long')
  
  if tozer is not None and isinstance(tozer, RobertaTokenizerFast):
    raise TypeError('Second argument is not type TokenizerFast')

  return tozer.decode(token_ids.tolist(), skip_special_tokens=skip_special_tok)[1:] if tozer is not None else rob_distil_tozer.decode(token_ids.tolist(), skip_special_tokens=skip_special_tok)[1:]
def offset_paddings(offset: list, maxtweetlen, prefix_space=3):
  try:
    assert offset[0] == offset[-1]
  except AssertionError:
    raise AssertionError("Offsets already padded or None type at tails!")

  offset[0] = (offset[1][0], offset[1][0])

  while offset[-1] is None or offset[-1] == (0, 0):
    del offset[-1]

  offset = offset + [(offset[-1][-1],offset[-1][-1]) for i in range(maxtweetlen - len(offset) - prefix_space)]
  offset = [(0,0) for i in range(prefix_space)] + offset

  return torch.tensor(offset)
def sentiment_padding(tweet, sent_st, max_tweet_len):
    tweet_dump = " " + str(' '.join(tweet.split()))
    sent_st_dump = " " + str(' '.join(sent_st.split()))
    st_tensor = torch.zeros(max_tweet_len, dtype=torch.long)
    st_len = len(sent_st_dump)
    idx_start = None
    idx_end = None
    
    
    for idx in (idx for idx, letter in enumerate(tweet_dump[1:]) if letter==sent_st_dump[1]):
        if tweet_dump[idx + 1:idx + st_len] == sent_st_dump[1:]:
            idx_start = idx + 1
            idx_end = idx + st_len ### INCLUSIVE GATING!!!
            st_tensor[idx_start: idx_end + 1] = st_tensor[idx_start: idx_end + 1] + 1
            break
        
    # sanity check
    try:
        assert (idx_start != None and idx_end != None)
        assert (st_len != 0)
    except AssertionError:
        print('sentiment_padding : Either improper selected sentiment support or sentiment support doesnt have length!')
        return -1
    
    return [idx_start], [idx_end], st_tensor
def data_preprocess(df, max_tweet_len, datasets='test', token_gap=0, debug_mode=False, offset_return=True):
    
    try:
        assert (type(df) == pd.DataFrame)
    except AssertionError:
        raise AssertionError("Input dataframe is not type pandas.DataFrame!")


    # Sentiment map
    dict_sentiment = {'positive': robdist_tokenize('positive')['input_ids'],
                      'negative': robdist_tokenize('negative')['input_ids'],
                      'neutral': robdist_tokenize('neutral')['input_ids']}
    enc_sentiment_len = len(dict_sentiment['neutral'])
    
    # Init data tensor, generic
    zero_temp = robdist_tokenize(" " + str(' '.join(str(df.iloc[0]['text']).split())))
    df_BPE = torch.cat((dict_sentiment[df.iloc[0]['sentiment']], zero_temp['input_ids']), dim=-1)
    df_att = torch.from_numpy(np.asarray([1]*enc_sentiment_len + [0]*token_gap + [1]*len(zero_temp['input_ids']) + [0]*(max_tweet_len - enc_sentiment_len - token_gap - len(zero_temp['input_ids']))))
    df_tok = torch.from_numpy(np.asarray([0]*(enc_sentiment_len + token_gap) + [1]*len(zero_temp['input_ids']) + [0]*(max_tweet_len - enc_sentiment_len - token_gap - len(zero_temp['input_ids']))))
    df_textid = robdist_tokenize(str(df.iloc[0]['textID']))['input_ids']
    df_offsets = offset_paddings(zero_temp['offsets'], max_tweet_len) # max_tweet_len x 2 ; start and end - 1

    if df_BPE.shape[0] < max_tweet_len:
        df_BPE = torch.cat((df_BPE, torch.zeros(max_tweet_len - df_BPE.shape[0], dtype=torch.long)), -1)
    else:
        df_BPE = df_BPE[:max_tweet_len]
    
    if df_textid.shape[0] < max_tweet_len:
        df_textid = torch.cat((df_textid, torch.zeros(max_tweet_len - df_textid.shape[0], dtype=torch.long)), -1)
    else:
        df_textid = df_textid[:max_tweet_len]

    ##### Additional init for training set ####
    if datasets=='train':
        init_search = True
        targetidx_1, targetidx_2, _ = sentiment_padding(' '.join(df.iloc[0]['text'].split()), ' '.join(df.iloc[0]['selected_text'].split()), max_tweet_len)
        for i in range(df_offsets.shape[0]):
            if init_search and df_offsets[i, 0] <= targetidx_1[0] < df_offsets[i, 1]:
                init_search = False
                targetidx_1 = [i]
            if df_offsets[i, 0] <= targetidx_2[0] <= df_offsets[i, 1]:
                targetidx_2 = [i]
                break

    #### Converting to higher Dim Tensor for concatenating ####
    df_att = df_att.unsqueeze(dim=-1)
    df_tok = df_tok.unsqueeze(dim=-1)
    df_BPE = df_BPE.unsqueeze(dim=-1)
    df_textid = df_textid.unsqueeze(dim=-1)
    df_offsets = df_offsets.unsqueeze(dim=0)
      
    
    # Tokenize data text
    for idx in range(1, len(df)):
        tempstring = str(' '.join(str(df.iloc[idx]['text']).split()))
        if len(tempstring) < 1:
            continue
        ins_ids = robdist_tokenize(" " + str(' '.join(str(df.iloc[idx]['text']).split())))
        text_unique_ids = robdist_tokenize(str(df.iloc[idx]['textID']))['input_ids']
        temp_token = torch.cat((dict_sentiment[df.iloc[idx]['sentiment']], ins_ids['input_ids']), dim=-1)
        temp_att = torch.from_numpy(np.asarray([1]*enc_sentiment_len + [0]*token_gap + [1]*len(ins_ids['input_ids'])+ [0]*(max_tweet_len - enc_sentiment_len - token_gap - len(ins_ids['input_ids']))))
        temp_tok = torch.from_numpy(np.asarray([0]*(enc_sentiment_len + token_gap) + [1]*len(ins_ids['input_ids'])+ [0]*(max_tweet_len - enc_sentiment_len - token_gap - len(ins_ids['input_ids']))))
        
        if temp_token.shape[0] < max_tweet_len:
            temp_token = torch.cat((temp_token, torch.zeros(max_tweet_len - temp_token.shape[0], dtype=torch.long)), dim=0)
        else:
            temp_token = temp_token[:max_tweet_len]

        if text_unique_ids.shape[0] < max_tweet_len:
            text_unique_ids = torch.cat((text_unique_ids, torch.zeros(max_tweet_len - text_unique_ids.shape[0], dtype=torch.long)), dim=0)
        else:
            text_unique_ids = text_unique_ids[:max_tweet_len]
        
        df_BPE = torch.cat((df_BPE, temp_token.unsqueeze(dim=-1)), dim=-1)
        df_textid = torch.cat((df_textid, text_unique_ids.unsqueeze(dim=-1)), dim=-1)
        df_att = torch.cat((df_att, temp_att.unsqueeze(dim=-1)), dim=-1)
        df_tok = torch.cat((df_tok, temp_tok.unsqueeze(dim=-1)), dim=-1)
        df_offsets = torch.cat((df_offsets, offset_paddings(ins_ids['offsets'], max_tweet_len).unsqueeze(dim=0)), dim=0)
        
        ##### Separator for training set ####
        if datasets=='train':
            init_search = True
            temp_idx1, temp_idx2, _ = sentiment_padding(' '.join(str(df.iloc[idx]['text']).split()), ' '.join(str(df.iloc[idx]['selected_text']).split()), max_tweet_len)
            for i in range(df_offsets.shape[1]):
                if init_search and df_offsets[idx, i, 0] <= temp_idx1[0] < df_offsets[idx, i, 1]:
                    init_search = False
                    temp_idx1 = [i]
                if df_offsets[idx, i, 0] <= temp_idx2[0] <= df_offsets[idx, i, 1]:
                    temp_idx2 = [i]
                    break
            targetidx_1 = targetidx_1 + temp_idx1
            targetidx_2 = targetidx_2 + temp_idx2
            

    if datasets=='test':
        return {
            'BPEencoded': df_BPE.T,
            'textID': df_textid.T,
            'att_mask': df_att.T,
            'tok_typeids': df_tok.T,
            'offsets': df_offsets
        }
    elif datasets=='train':
        return {
            'BPEencoded': df_BPE.T,
            'textID': df_textid.T,
            'target_idx': torch.from_numpy(np.stack((np.asarray(targetidx_1), np.asarray(targetidx_2)), axis=0)).T,
            'att_mask': df_att.T,
            'tok_typeids': df_tok.T,
            'offsets': df_offsets
        }
    else:
        return -1
import pandas as pd
import os
import torch
import re
import numpy as np
import transformers

data_path = '/kaggle/input/tweet-sentiment-extraction'
config_path = '/kaggle/input/submission'
prep_out_path = '/kaggle/working'
temporary_tweet = '/kaggle/input/roberta-openai'
def stackup_ids(input_ids, mask_ids, tokentype_ids, textID_ids, offsets, bertmodel='roberta'):
    try:
        assert (input_ids.shape == mask_ids.shape == tokentype_ids.shape)
        assert (input_ids.shape[0] ==  textID_ids.shape[0])
    except AssertionError:
        raise AssertionError('stackup_ids: arguments do not have matched shape!')
    
    token_type_modifier = tokentype_ids*int(0) if bertmodel=='roberta' else tokentype_ids
    
    cat_out = torch.cat((input_ids.unsqueeze(dim=-1),
                         mask_ids.long().unsqueeze(dim=-1),
                         token_type_modifier.long().unsqueeze(dim=-1),
                         textID_ids.unsqueeze(dim=-1),
                         offsets.long()), dim=-1) # casting manual nparray to long for compatibility purpose

    return cat_out
offnum = np.random.randint(0,27450,(1,))[0]
getnum = 10 # len(train_dataframe) - offnum - 1
testdatacheck = data_preprocess(train_dataframe.loc[offnum:offnum + getnum], 192, datasets='train')
print(str(train_dataframe.loc[offnum:offnum + getnum]['text']))
print()
print(testdatacheck['textID'].shape)
print()
print(testdatacheck['target_idx'])
print()
print(testdatacheck['target_idx'].shape)
print()
print(train_dataframe.loc[offnum:offnum + getnum]['textID'])
print()
print(train_dataframe.loc[offnum:offnum + getnum]['text'])
print()
print(train_dataframe.loc[offnum:offnum + getnum]['selected_text'])
print()
print('\n'.join([robdist_detokenize(testdatacheck['textID'][i,:]) for i in range(len(testdatacheck['textID']))]))
print()
print(testdatacheck['offsets'].shape)
testdatastack = stackup_ids(testdatacheck['BPEencoded'],
                            testdatacheck['att_mask'],
                            testdatacheck['tok_typeids'],
                            testdatacheck['textID'],
                            testdatacheck['offsets'])
print()
print(testdatastack.shape)
print()
randidx = np.random.randint(0,getnum + 1,(1,))[0]
textIDdecode = testdatastack[randidx,:,-3].squeeze(dim=0)
print(testdatastack[randidx, :40, :])
print()
print(testdatacheck['target_idx'][randidx, :])
print()
# sentiment support
print(str(" " + robdist_detokenize(testdatastack[randidx, 3:, 0]))[testdatastack[randidx, testdatacheck['target_idx'][randidx, 0], -2]:testdatastack[randidx, testdatacheck['target_idx'][randidx, 1], -1]])
print()
print(robdist_detokenize(textIDdecode))
print()
print(train_dataframe.loc[offnum + randidx:offnum + randidx]['textID'].item())
print()
print('textID matched!' if train_dataframe.loc[offnum + randidx:offnum + randidx]['textID'].item() == robdist_detokenize(textIDdecode) else 'textID do not match')

del testdatacheck, testdatastack, randidx, textIDdecode, offnum, getnum
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch.optim as optim
import torch.nn as nn

# GLOBAL HYPERPARAMETER #1 for data
MAX_ORIG_TWEET_LEN = 192 # Changing this requires deleting previous pre-processed dataset and ensure it is value is more than max len of entire datasets 'text', it will throw error in padding and data preprocessing
BATCH_SIZE = 32
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
fold_run = 1/0.05


try:
    train_preprocess, test_preprocess = pickle.load(open(os.path.join(temporary_tweet, 'tweet_preprocess.p'), mode='rb'))
    print("Pre-processed tweet dataset loaded!")
except FileNotFoundError:
    print("Initializing/Repeating preprocess of data!")
    train_preprocess = data_preprocess(train_dataframe, MAX_ORIG_TWEET_LEN, datasets='train')
    test_preprocess = data_preprocess(test_dataframe, MAX_ORIG_TWEET_LEN, datasets='test')
    pickle.dump((train_preprocess, test_preprocess), open(os.path.join(prep_out_path, 'tweet_preprocess.p'), 'wb'))
    print('Preprocessing completed and saved the data!')

# Validation set slicing   
val_ratio = float(1/fold_run)
val_slice_idx = int((1-val_ratio)*train_preprocess['BPEencoded'].shape[0])

train_x_ids = stackup_ids(train_preprocess['BPEencoded'][0:val_slice_idx, :],
                          train_preprocess['att_mask'][0:val_slice_idx, :].long(),
                          train_preprocess['tok_typeids'][0:val_slice_idx, :].long(),
                          train_preprocess['textID'][0:val_slice_idx, :],
                          train_preprocess['offsets'][0:val_slice_idx, :].long())

train_y = train_preprocess['target_idx'][0:val_slice_idx, :]

val_x_ids = stackup_ids(train_preprocess['BPEencoded'][val_slice_idx:, :],
                        train_preprocess['att_mask'][val_slice_idx:, :].long(),
                        train_preprocess['tok_typeids'][val_slice_idx:, :].long(),
                        train_preprocess['textID'][val_slice_idx:, :],
                        train_preprocess['offsets'][val_slice_idx:, :].long())
val_y = train_preprocess['target_idx'][val_slice_idx:, :]

# Train data loader
train_x_ids_dataset = TensorDataset(train_x_ids, train_y)
train_loader = DataLoader(train_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# Validation data loader
val_x_ids_dataset = TensorDataset(val_x_ids, val_y)
val_loader = DataLoader(val_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
print(len(train_x_ids_dataset))
print(len(val_x_ids_dataset))
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def char_jaccard_seq(str1, str2):
    jaccscore = 0.0
    a = list(''.join(str1.lower().split()))
    b = list(''.join(str2.lower().split()))
    minlen = len(a) if len(a) < len(b) else len(b)
    
    for idx in range(minlen):
        if a[idx] == b[idx]:
            jaccscore += 1
    
    return (jaccscore*2)/(len(a) + len(b))
stopper= 1

devv = 'cuda' if torch.cuda.is_available() else 'cpu'

for i, batch in enumerate(train_loader):
  if stopper < 1:
    break
  batchx, batchy = batch
  batchx, batchy = batchx.to(devv), batchy.to(devv)
  stopper -= 1

textidd = robdist_detokenize(batchx[0,:,3])
print(train_dataframe.loc[train_dataframe['textID'] == textidd]['text'].item())
print(robdist_detokenize(batchx[0,:,0]))
print(batchy[0,:])
print('---------------------------------------------------------------------------------------------------------------')
train1 = train_dataframe.loc[train_dataframe['textID'] == textidd]['selected_text'].item()
train2 = str(" " + ' '.join(train_dataframe.loc[train_dataframe['textID'] == textidd]['text'].item().split()))[batchx[0,:,-2:][batchy[0,0], 0]:batchx[0,:,-2:][batchy[0,1], 1]]
print(train1)
print(train2)
print(f'Jaccard {jaccard(train1,train2)}')
print('---------------------------------------------------------------------------------------------------------------')
print(batchx[0,:40,-2:].squeeze(dim=0).T)
print(batchx[0,:50,:])

del devv, textidd, train1, train2
# Packages and path
from transformers.modeling_roberta import RobertaModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_bert import BertPreTrainedModel
import torch.nn as nn

roberta_path = config_path
model_in = '/kaggle/input/submission/model_colab_24.pt'
# see_state = torch.load(model_in) if dev == 'cuda' else torch.load(model_in, map_location='cpu')
# for i,e in see_state.items():
#     if i == 'model':
#         pass
#     else:
#         print(f'{i} : {e}')
# del see_state
def weights_init(mods):
  if type(mods) == nn.Linear:
    torch.nn.init.normal_(mods.weight, mean= 0.0, std=0.1)


class roberta_mlp_net(BertPreTrainedModel):

    def __init__(self, robpath, net_config, maxtweetlen):
      super(roberta_mlp_net, self).__init__(net_config)
      net_config.output_hidden_states = True
      self.roberta = RobertaModel.from_pretrained(pretrained_model_name_or_path=robpath, config=net_config)
      self.drop_out1 = nn.Dropout(p=0.1)
      self.laynorm = nn.LayerNorm((768*2,))
      self.densenet = nn.Linear(768*2, 2)
      self.densenet.apply(weights_init)

    def todevice(self):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.to(device)
      print(f'Processing device: {device}')
    
    def forward(self, in_ids, att_ids, tok_ids):
      _ , _, out = self.roberta(input_ids=in_ids, attention_mask=att_ids, token_type_ids=tok_ids) # last hidden, pooled bert cls, (embed output + all hidden states) #12 bertlayers for roberta openai
      out = torch.cat((out[-1], out[-2]), dim=-1) #last and second last hidden output
      out = self.drop_out1(out)
      out = self.laynorm(out)
      out = self.densenet(out)

      return out
# GLOBAL HYPERPARAMETER #2 for initial training
learn_rate = 80e-7
criterion_mod = nn.CrossEntropyLoss()
global_threshold_valid = np.Inf
n_epoch = 30
print_every = 100

model_config = PretrainedConfig.from_json_file(os.path.join(roberta_path, 'roberta-base-openai-detector-config.json'))
testmodel = roberta_mlp_net(robpath=roberta_path, net_config=model_config, maxtweetlen=MAX_ORIG_TWEET_LEN)
optimizer_mod = optim.AdamW(testmodel.parameters(), lr=learn_rate)
optimizer_mod.param_groups[0]['amsgrad'] = False

if dev == 'cuda':
    testmodel.load_state_dict(torch.load(model_in)['model'])
else:
    testmodel.load_state_dict(torch.load(model_in, map_location='cpu')['model'])
print(testmodel)
print(model_config)
def len_mismatch_penalty(pred_start, pred_end, tar_start, tar_end, coeff_mult=0.1):
    _, start_cand = pred_start.squeeze(dim=-1).topk(1, dim=-1)
    _, end_cand = pred_end.squeeze(dim=-1).topk(1, dim=-1)
    start_cand = start_cand.squeeze(dim=-1)
    end_cand = end_cand.squeeze(dim=-1)
    diff_cand = abs(end_cand - start_cand) - (tar_end - tar_start)
    result = torch.matmul(diff_cand.float(), diff_cand.float().T)
    return result*coeff_mult
def train_func(fold, model, dev, optimizer, trainloader, validloader, tweetlen, criterion, epochs, glob_th, verbose_every=10, first_train=True):
  valid_loss_min = np.Inf
  init_train = True
  early_stop = 2
  prev_val_loss = np.Inf

  model.todevice()
  for epoch in range(1, epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    if 3 < epoch < 5:
      optimizer.param_groups[0]['lr'] = 0.8*optimizer.param_groups[0]['lr']
    elif 5 < epoch < 10:
      optimizer.param_groups[0]['lr'] = 0.8*optimizer.param_groups[0]['lr']
    elif epoch > 10:
      optimizer.param_groups[0]['lr'] = 0.8*optimizer.param_groups[0]['lr']

    ##### Training #####
    model.train()
    print(f'Fold {fold} : Training of epoch #{epoch}...')
    for batch_idx, (trainx, trainy) in enumerate(trainloader):
      model.zero_grad()
      trainx, trainy = trainx.to(dev), trainy.to(dev)
      output = model(trainx[:, :, 0], trainx[:, :, 1], trainx[:, :, 2])
      loss = (criterion(output[:, :, 0], trainy[:, 0]) + criterion(output[:, :, 1], trainy[:, 1]))
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

      if batch_idx % verbose_every == 0 and batch_idx != 0:
        print(f'Training loss at epoch # {epoch}: {train_loss/(verbose_every)}')
        train_loss = 0.0
    
    if init_train and not first_train:
      valid_loss_min = glob_th
      init_train = False

    #### Validation ####
    model.eval()
    print(f'Validation of epoch #{epoch}...')
    for batch_idx, (valx, valy) in enumerate(validloader):
      valx, valy = valx.to(dev), valy.to(dev)
      with torch.no_grad():
        val_output = model(valx[:, :, 0], valx[:, :, 1], valx[:, :, 2])
        vloss = criterion(val_output[:, :, 0], valy[:, 0]) + criterion(val_output[:, :, 1], valy[:, 1])
        valid_loss += vloss.item()
    
    valid_loss = valid_loss/len(validloader)
    print('Validation loss : {:.6f}'.format(valid_loss))
    print('Validation loss min : {:.6f}'.format(valid_loss_min))
        
    if valid_loss < valid_loss_min:
      print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
      valid_loss_min = valid_loss
      glob_th = valid_loss
      checkpoint_model = {'architecture': save_architecture,
                          'learnrate': learn_rate,
                          'validmin': valid_loss_min,
                          'model': model.state_dict()}
      torch.save(checkpoint_model, os.path.join(model_out))
      early_stop = 2
    else:
      if prev_val_loss < valid_loss:
        early_stop -= 1
      else:
        early_stop = 2
    
    if early_stop < 0:
      break
    prev_val_loss = valid_loss
  
  checkpoint_model = {'architecture': save_architecture,
                      'learnrate': learn_rate,
                      'validmin': glob_th,
                      'model': model.state_dict()}
  torch.save(checkpoint_model, os.path.join(model_last_out))

  print('Training completed!')
  return model, glob_th
testmodel, global_threshold_valid = train_func(fold=0,
                                               model=testmodel,
                                               dev=dev,
                                               optimizer=optimizer_mod,
                                               trainloader=train_loader,
                                               validloader=val_loader,
                                               tweetlen=MAX_ORIG_TWEET_LEN,
                                               criterion=criterion_mod,
                                               epochs=n_epoch,
                                               glob_th=global_threshold_valid,
                                               verbose_every=print_every,
                                               first_train=False)
# # Continued run

# restartidx = 2

# global_threshold_valid = 1.743169

# val_ratio = float(1/fold_run)
# val_slice_idx = int((1-val_ratio)*train_preprocess['BPEencoded'].shape[0])
# init_slice = val_slice_idx - int((restartidx + 1)*val_ratio*train_preprocess['BPEencoded'].shape[0])
# end_slice =  val_slice_idx - int(restartidx*val_ratio*train_preprocess['BPEencoded'].shape[0])


# train_x_ids = stackup_ids(torch.cat((train_preprocess['BPEencoded'][0:init_slice, :],train_preprocess['BPEencoded'][end_slice:, :]), dim=0),
#                           torch.cat((train_preprocess['att_mask'][0:init_slice, :],train_preprocess['att_mask'][end_slice:, :]), dim=0),
#                           torch.cat((train_preprocess['tok_typeids'][0:init_slice, :],train_preprocess['tok_typeids'][end_slice:, :]), dim=0),
#                           torch.cat((train_preprocess['textID'][0:init_slice, :],train_preprocess['textID'][end_slice:, :]), dim=0),
#                           torch.cat((train_preprocess['offsets'][0:init_slice, :], train_preprocess['offsets'][end_slice:, :]), dim=0))

# train_y = torch.cat((train_preprocess['target_idx'][0:init_slice, :],train_preprocess['target_idx'][end_slice:, :]), dim=0)

# val_x_ids = stackup_ids(train_preprocess['BPEencoded'][init_slice:end_slice, :],
#                         train_preprocess['att_mask'][init_slice:end_slice, :],
#                         train_preprocess['tok_typeids'][init_slice:end_slice, :],
#                         train_preprocess['textID'][init_slice:end_slice, :],
#                         train_preprocess['offsets'][init_slice:end_slice, :])

# val_y = train_preprocess['target_idx'][init_slice:end_slice, :]

# # Train data loader
# train_x_ids_dataset = TensorDataset(train_x_ids, train_y)
# train_loader = DataLoader(train_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# # Validation data loader
# val_x_ids_dataset = TensorDataset(val_x_ids, val_y)
# val_loader = DataLoader(val_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
# for i in range(0, fold_run - 1):
#     # Model reinstantiation for next fold
#     del model_config, testmodel, criterion_mod, optimizer_mod
#     model_config = PretrainedConfig.from_json_file(os.path.join(roberta_path, 'roberta-base-openai-detector-config.json'))
#     testmodel = roberta_mlp_net(robpath=roberta_path, net_config=model_config, maxtweetlen=MAX_ORIG_TWEET_LEN)
#     # testmodel.load_state_dict(torch.load(model_out)['model'])
#     # testmodel.load_state_dict(torch.load(model_in, map_location='cpu')['model'])
#     criterion_mod = nn.CrossEntropyLoss()
#     optimizer_mod = optim.AdamW(testmodel.parameters(), lr=learn_rate)
#     testmodel, global_threshold_valid = train_func(i, testmodel, dev, optimizer_mod, train_loader, val_loader, MAX_ORIG_TWEET_LEN, criterion_mod, n_epoch, global_threshold_valid, print_every, first_train=False)
    
#     val_ratio = float(1/fold_run)
#     val_slice_idx = int((1-val_ratio)*train_preprocess['BPEencoded'].shape[0])
#     init_slice = val_slice_idx - int((i + 1)*val_ratio*train_preprocess['BPEencoded'].shape[0])
#     end_slice =  val_slice_idx - int(i*val_ratio*train_preprocess['BPEencoded'].shape[0])
    
#     if i < 3:
#         train_x_ids = stackup_ids(torch.cat((train_preprocess['BPEencoded'][0:init_slice, :],train_preprocess['BPEencoded'][end_slice:, :]), dim=0),
#                                   torch.cat((train_preprocess['att_mask'][0:init_slice, :],train_preprocess['att_mask'][end_slice:, :]), dim=0),
#                                   torch.cat((train_preprocess['tok_typeids'][0:init_slice, :],train_preprocess['tok_typeids'][end_slice:, :]), dim=0),
#                                   torch.cat((train_preprocess['textID'][0:init_slice, :],train_preprocess['textID'][end_slice:, :]), dim=0),
#                                   torch.cat((train_preprocess['offsets'][0:init_slice, :], train_preprocess['offsets'][end_slice:, :]), dim=0))

#         train_y = torch.cat((train_preprocess['target_idx'][0:init_slice, :],train_preprocess['target_idx'][end_slice:, :]), dim=0)

#         val_x_ids = stackup_ids(train_preprocess['BPEencoded'][init_slice:end_slice, :],
#                                 train_preprocess['att_mask'][init_slice:end_slice, :],
#                                 train_preprocess['tok_typeids'][init_slice:end_slice, :],
#                                 train_preprocess['textID'][init_slice:end_slice, :],
#                                 train_preprocess['offsets'][init_slice:end_slice, :])
        
#         val_y = train_preprocess['target_idx'][init_slice:end_slice, :]

#         # Train data loader
#         train_x_ids_dataset = TensorDataset(train_x_ids, train_y)
#         train_loader = DataLoader(train_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        
#         # Validation data loader
#         val_x_ids_dataset = TensorDataset(val_x_ids, val_y)
#         val_loader = DataLoader(val_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
#     else:
#         train_x_ids = stackup_ids(train_preprocess['BPEencoded'][end_slice:, :],
#                                   train_preprocess['att_mask'][end_slice:, :],
#                                   train_preprocess['tok_typeids'][end_slice:, :],
#                                   train_preprocess['textID'][end_slice:, :],
#                                   train_preprocess['offsets'][end_slice:, :])

#         train_y = torch.cat((train_preprocess['target_idx'][end_slice:, :],train_preprocess['target_idx'][end_slice:, :]), dim=0)

#         val_x_ids = stackup_ids(train_preprocess['BPEencoded'][0:end_slice, :],
#                                 train_preprocess['att_mask'][0:end_slice, :],
#                                 train_preprocess['tok_typeids'][0:end_slice, :],
#                                 train_preprocess['textID'][0:end_slice, :],
#                                 train_preprocess['offsets'][0:end_slice, :])
        
#         val_y = train_preprocess['target_idx'][0:end_slice, :]

#         # Train data loader
#         train_x_ids_dataset = TensorDataset(train_x_ids, train_y)
#         train_loader = DataLoader(train_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        
#         # Validation data loader
#         val_x_ids_dataset = TensorDataset(val_x_ids, val_y)
#         val_loader = DataLoader(val_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        
        
#         # Reinstantiate model for last fold
#         del model_config, testmodel, criterion_mod, optimizer_mod
#         model_config = PretrainedConfig.from_json_file(os.path.join(roberta_path, 'roberta-base-openai-detector-config.json'))
#         testmodel = roberta_mlp_net(robpath=roberta_path, net_config=model_config, maxtweetlen=MAX_ORIG_TWEET_LEN)
#         criterion_mod = nn.CrossEntropyLoss()
#         optimizer_mod = optim.AdamW(testmodel.parameters(), lr=learn_rate)
        
#         testmodel, global_threshold_valid = train_func(i + 1, testmodel, dev, optimizer_mod, train_loader, val_loader, MAX_ORIG_TWEET_LEN, criterion_mod, n_epoch, global_threshold_valid, print_every, first_train=False)
def pp(filtered_output, real_tweet):
    filtered_output = ' '.join(filtered_output.split())
    if len(real_tweet.split()) < 2:
        filtered_output = real_tweet
    else:
        if len(filtered_output.split()) == 1:
            if filtered_output.endswith(".."):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '..', filtered_output)
                return filtered_output
            if filtered_output.endswith('!!'):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!!', filtered_output)
                return filtered_output

        if real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = ' '.join(real_tweet.split())
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag < start:
                filtered_output = real_tweet[start:end]

        if "  " in real_tweet and not real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = re.sub(" {2,}", " ", real_tweet)
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag < start:
                filtered_output = real_tweet[start:end]
    return filtered_output
def eval_jacc(model, dev, decoder, testloader, criterion, dataframe, enable_log=True, is_test=False, every_batch=10):
  model.eval()
  model.to(dev)
  instance_count = 0
  test_loss = 0.0
  jacc_score_overall = 0.0
  pred_seltext_global = []
  batch_textid_global = []
  batch_text_global = []
  sentiment_global = []
    
  for batchidx, batch in enumerate(testloader):
    pred_seltext = []
    select_text = []
    if batchidx % every_batch == every_batch - 1:
      print(f'Dumping eval of {batchidx + 1}-th batch...')
    if not is_test:
        testx, testy = batch
        testx, testy = testx.to(dev), testy.to(dev)
    else:
        testx = batch[0]
        testx = testx.to(dev)
    
    test_offsets = testx[:,:, -2:] # Getting last two columns of x tensors, offset positions of each words

    
    batch_textid_tensor = testx[:, :, 3].to('cpu').long() # Roberta-encoded text ID, column 3 of x tensors
    batch_textid = [decoder(batch_textid_tensor[i,:]) for i in range(batch_textid_tensor.shape[0])]
    batch_textid_global = batch_textid_global + batch_textid
    sentiment_text = [decoder(testx[i, :3, 0].to('cpu').long()) for i in range(batch_textid_tensor.shape[0])]
    sentiment_global = sentiment_global + sentiment_text
    
           
    batch_text = [str(" " + ' '.join(str(dataframe.loc[dataframe['textID'] == batch_textid[i]]['text'].item()).split())) for i in range(len(batch_textid))]
    batch_text_global = batch_text_global + batch_text

    if not is_test:
      batch_gnd_seltext = [str(dataframe.loc[dataframe['textID'] == batch_textid[i]]['selected_text'].item()) for i in range(len(batch_textid))] # very raw, no split and prefix addition
      for idx, item in enumerate(batch_text):

        sel_idstart = test_offsets[idx, testy[idx, 0].item(), 0].item()
        sel_idend = test_offsets[idx, testy[idx, 1].item(), 1].item()
        select_text = select_text + [batch_gnd_seltext[idx]]


    with torch.no_grad():
        testoutput = model(testx[:, :, 0], testx[:, :, 1], testx[:, :, 2])
    

    # Prediction
    _, pred_starts = testoutput[:, :, 0].squeeze(dim=-1).topk(1, dim=-1) # batch_size indices
    _, pred_ends = testoutput[:, :, 1].squeeze(dim=-1).topk(1, dim=-1) # batch_size indices
    
    for idx, item in enumerate(batch_text):
      instance_count += 1
      idstart = test_offsets[idx, pred_starts[idx].item(), 0].item()
      idend = test_offsets[idx, pred_ends[idx].item(), 1].item()

      if len(item[idstart:idend]) == 0:
          pred_seltext.append(pp(item,item))
      else:
          pred_seltext.append(pp(item[idstart:idend],item))

      if enable_log:
            print('--------------------------------------------------------------------------------')
            print('startidx\t\t\t: {0}'.format(pred_starts[idx].item()))
            print('Sentiment\t\t\t: {0}'.format(sentiment_text[idx]))
            print('Text\t\t\t\t: {0}'.format(item))
            print('Predicted sentiment text\t: {0}'.format(pred_seltext[-1]))
            print('Predicted offset tensor\t\t: [{0}, {1}]'.format(pred_starts[idx].item(),pred_ends[idx].item()))
            print('Offset tensors\t\t\t: {0}'.format(test_offsets[idx, :50, :].T))
      if not is_test and enable_log:
            print('Actual Offset tensors\t\t: {0}'.format(testy[idx, :]))
            print('Actual sentiment text\t\t: {0}'.format(batch_gnd_seltext[idx]))
            print('Jaccard score\t\t\t: {0}'.format(jaccard(pred_seltext[-1], batch_gnd_seltext[idx])))
            print('--------------------------------------------------------------------------------')
    
    
      
    
    pred_seltext_global = pred_seltext_global + pred_seltext
    if not is_test:
      jacc_score = 0.0
      for idx, seltext in enumerate(pred_seltext):
        jacc_score += jaccard(seltext,select_text[idx])
      
      jacc_score_overall += jacc_score
      if batchidx % every_batch == every_batch - 1:
        print(f'Jaccard avg score of {batchidx + 1}-th batch: {jacc_score/len(pred_seltext)}')
    

  if not is_test:
    return {'pred': pred_seltext_global,
            'textID': batch_textid_global,
            'text': batch_text_global,
            'predstart': pred_starts,
            'predend': pred_ends,
            'jaccard': jacc_score_overall/instance_count,
            'sentiment' : sentiment_global}
  else:
    return {'pred': pred_seltext_global,
            'textID': batch_textid_global,
            'sentiment' : sentiment_global}
small_val_loader = ()

for i, val in enumerate(val_loader):
    if i < 1:
        small_val_loader = small_val_loader + (val,)

print(str(len(small_val_loader)) + " batch(es) selected!")
eval_result = eval_jacc(testmodel, dev, robdist_detokenize, small_val_loader, criterion_mod, train_dataframe, enable_log=True, is_test=False, every_batch=50)
eval_result['jaccard']
test_x_ids = stackup_ids(test_preprocess['BPEencoded'],
                         test_preprocess['att_mask'],
                         test_preprocess['tok_typeids'],
                         test_preprocess['textID'],
                         test_preprocess['offsets'])

# Validation data loader
test_x_ids_dataset = TensorDataset(test_x_ids)
test_loader = DataLoader(test_x_ids_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

testeval_result = eval_jacc(testmodel, dev, robdist_detokenize, test_loader, criterion_mod, test_dataframe, enable_log=False, is_test=True, every_batch=10)

# for manual QC check
testoutdataframe = pd.DataFrame(data={'textID': testeval_result['textID'],
                                      'selected_text': testeval_result['pred'],
                                      'sentiment': testeval_result['sentiment']})
testoutdataframe.to_csv(path_or_buf=os.path.join('/kaggle/working', 'testdataset_out.csv'), index=False)


# for Kaggle submission
submission = pd.DataFrame(data={'textID': testeval_result['textID'],
                                'selected_text': testeval_result['pred']})
submission.to_csv(path_or_buf=os.path.join('/kaggle/working', 'submission.csv'), index=False)
testoutdataframe.head(50)