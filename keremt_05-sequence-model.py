from fastai.vision.all import *
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100
embspath = Path("/kaggle/input/rsnaperawembs256v2//")
embspath.ls()
train_feat_df = pd.read_csv(embspath/'train.csv')
# TODO: Make sure to normalize views for pe: left-right-central
train_feat_df.head(2)
embs = torch.cat([torch.load(embspath/f'train_embs-{o}.pkl') for o in [0,1,2,'final']])

embs.shape
# add padding embedding and idx

embs_len = len(embs)

pad_emb = torch.zeros_like(embs[:1])

embs = torch.cat([embs, pad_emb])

input_pad_idx = embs_len
embs[input_pad_idx], embs.shape, input_pad_idx
seq_lens = train_feat_df.groupby("StudyInstanceUID").apply(len)

seq_lens.hist();
(seq_lens < 300).mean()
metadata_path = Path("/kaggle/input/rsnastrpemetadata/")
train_metadf = pd.concat([pd.read_parquet(metadata_path/f"train_metadf_part{i}.pqt") for i in range(1,3)]).reset_index(drop=True)
train_metadf.head()
train_meta_feats = train_metadf[['StudyInstanceUID', 'SOPInstanceUID', 'ImagePositionPatient2', 'img_min', 'img_max', 'img_mean', 'img_std', 'img_pct_window']]
def minmax_scaler(o): return (o - min(o))/(max(o) - min(o))
scaled_pos = train_meta_feats.groupby('StudyInstanceUID')['ImagePositionPatient2'].apply(minmax_scaler)

train_meta_feats['scaled_position'] = scaled_pos
train_meta_feats.isna().sum()
train_meta_feats.sort_values('ImagePositionPatient2')
meta_feats = ['img_min', 'img_max', 'img_mean', 'img_std', 'scaled_position']

mean_std = {}

for f in meta_feats:

    mean,std = train_meta_feats[f].mean(), train_meta_feats[f].std()

    train_meta_feats[f] = (train_meta_feats[f] - mean) / std

    mean_std[f] = (mean, std)
train_meta_feats
meta_feats_dict = dict(zip(train_meta_feats['SOPInstanceUID'], train_meta_feats[meta_feats].to_numpy()))
meta_feats_dict['cc96a7a2e72c']
len(meta_feats_dict)
meta_embs = np.vstack([meta_feats_dict[o] for o in train_feat_df['SOPInstanceUID'].values])
meta_embs.shape
meta_embs = tensor(np.vstack((meta_embs, np.zeros((1,meta_embs.shape[1])))))
meta_embs.shape
from fastai.text.all import *
# TODO: Use saved pids

unique_pids = train_feat_df.StudyInstanceUID.unique()

n = len(unique_pids)

nvalid = int(n*0.05); nvalid, n

unique_pids = np.random.permutation(unique_pids)

train_pids = unique_pids[nvalid:]

valid_pids = unique_pids[:nvalid]

len(train_pids), len(valid_pids)
files_dict = defaultdict(list)

for i, (_, row) in enumerate(train_feat_df.iterrows()):

    fn = Path(row['fname'])

    slice_no = int(fn.stem.split("_")[0])

    sid = fn.parent.parent.name

    files_dict[sid].append(({**dict(row), **{"slice_no":slice_no, "embs_idx":i}}))
dict(row)
image_targets = L(['pe_present_on_image'])

exam_targets = L([

#           'positive_exam_for_pe'

            'negative_exam_for_pe',

            'indeterminate',



            'rv_lv_ratio_gte_1',

            'rv_lv_ratio_lt_1',

    # none



            'leftsided_pe',

            'rightsided_pe',

            'central_pe',



            'chronic_pe',

            'acute_and_chronic_pe',           

            # neither chronic or acute_and_chronic

          

    

    

#             'qa_motion',

#             'qa_contrast',

#             'flow_artifact',

#             'true_filling_defect_not_pe',

             ]); exam_targets
len(files_dict)
trn_pid = np.random.choice(train_pids)
def get_x(pid):

    o = files_dict[pid]    

    l = sorted(o, key=lambda x: x['slice_no']) 

    return tensor([o['embs_idx'] for o in l])



def get_img_y(pid):

    o = files_dict[pid]    

    l = sorted(o, key=lambda x: x['slice_no']) 

    img_y = [o['pe_present_on_image'] for o in l]

    exam_y = [max(img_y)] + [o[0][t] for t in exam_targets]

    return tensor(img_y)



def get_exam_y(pid):

    """

    'POSITIVE','negative_exam_for_pe','indeterminate',

    'rv_lv_ratio_gte_1','rv_lv_ratio_lt_1', 'NEITHER'

    'leftsided_pe','rightsided_pe','central_pe',

    'chronic_pe','acute_and_chronic_pe','NEITHER'

    """

    o = files_dict[pid]    

    l = sorted(o, key=lambda x: x['slice_no']) 

    img_y = [o['pe_present_on_image'] for o in l]

    

    exam_y = [max(img_y)] + [o[0][t] for t in exam_targets]

    

    none_chro_acute = [exam_y[-1] == exam_y[-2]]

    exam_y += none_chro_acute

    

    none_rv_lv = [exam_y[3] == exam_y[4]]

    exam_y = exam_y[:4] + none_rv_lv + exam_y[4:]

    

    

    return tensor(exam_y)



# before_batch: after collecting samples before collating

targ_pad_idx = 666

def SequenceBlock():       return  TransformBlock(type_tfms=[get_x], dl_type=SortedDL, dls_kwargs={'before_batch':

                                                       [partial(pad_input, pad_idx=input_pad_idx),

                                                        partial(pad_input, pad_idx=targ_pad_idx, pad_fields=1)]})

def SequenceTargetBlock(): return TransformBlock(type_tfms=[get_img_y])

def TargetBlock():         return TransformBlock(type_tfms=[get_exam_y])
data = DataBlock(blocks=(SequenceBlock,SequenceTargetBlock,TargetBlock), 

                 n_inp=1, 

                 splitter=FuncSplitter(lambda o: o in valid_pids))

dls = data.dataloaders(list(train_pids)+list(valid_pids), bs=128)

b = dls.one_batch()
learner = Learner(dls, nn.Linear(10,10), loss_func=noop)

learner._split(b)

len(learner.xb), len(learner.yb)
learner.xb[0].shape, learner.yb[0].shape, learner.yb[1].shape
embs[learner.xb[0]].shape, meta_embs[learner.xb[0]].shape
torch.cat([embs[learner.xb[0]], meta_embs[learner.xb[0]]], -1).shape
device = default_device()
class MultiHeadedSequenceClassifier(Module):

    "dim: input sequence feature dim"

    def __init__(self, input_pad_idx=input_pad_idx, dim=1024):

        

        store_attr('input_pad_idx')

        self.lstm1 = nn.LSTM(dim+5, dim//16, bidirectional=True)

#         self.lstm1 = nn.LSTM(dim+5, dim//8, bidirectional=True)

#         self.lstm2 = nn.LSTM(dim//4, dim//16, bidirectional=True)

        

        # image level preds

        self.seq_cls_head = nn.Linear(dim//8, 1)

    

        

        # positive, negative, indeterminate

        self.pe_head = nn.Linear(dim//4, 3) # softmax

        # rv / lv >=,  < 1 or neither

        self.rv_lv_head = nn.Linear(dim//4, 3) # softmax

        # l,r,c pe

        self.pe_position_head = nn.Linear(dim//4, 3) # sigmoid

        # chronic, ac-chr or neither

        self.chronic_pe_head = nn.Linear(dim//4, 3) # softmax

        

    

    def forward(self, x):

        

        # get mask from non-pad idxs and then features

        mask = x != self.input_pad_idx

        x = torch.cat([embs[x], meta_embs[x]], dim=-1).to(device)

        

        # sequence outs

        x, _ = self.lstm1(x) 

#         x, _ = self.lstm2(x)

        seq_cls_out = self.seq_cls_head(x).squeeze(-1)

        

        

        #masked concat pool

        pooled_x = []

        for i in range(x.size(0)):

            xi = x[i, mask[i], :]

            pooled_x.append(torch.cat([xi.mean(0), xi.max(0).values]).unsqueeze(0))

        pooled_x = torch.cat(pooled_x)

        



        # 'POSITIVE','negative_exam_for_pe','indeterminate'

        out1 = self.pe_head(pooled_x)



        # 'rv_lv_ratio_gte_1','rv_lv_ratio_lt_1', 'NEITHER'

        out2 = self.rv_lv_head(pooled_x)



        # 'leftsided_pe','rightsided_pe','central_pe',

        out3 = self.pe_position_head(pooled_x)



        # 'chronic_pe','acute_and_chronic_pe','NEITHER'

        out4 = self.chronic_pe_head(pooled_x)



        return (seq_cls_out, out1, out2, out3, out4)
model = MultiHeadedSequenceClassifier()
# outs = model(*learner.xb)
class MultiLoss(Module):

    

    def __init__(self, targ_pad_idx=666):

        store_attr("targ_pad_idx")

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.ce_loss = CrossEntropyLossFlat()



    def forward(self, inp, yb0, yb1):



        seq_cls_out, out1, out2, out3, out4 = inp



        mask = yb0 != self.targ_pad_idx

        loss0 = self.bce_loss(seq_cls_out[mask], yb0[mask].float())





        # loss 1 p/n/i

        loss1 = self.ce_loss(out1, torch.where(yb1[:,:3])[1])



        # loss 2 rv/lv/neither

        loss2 = self.ce_loss(out2, torch.where(yb1[:,3:6])[1])



        # loss 3 L/R/C

        loss3 = self.bce_loss(out3, yb1[:,6:9].float())



        # loss 3 chro/acute/neither

        loss4 = self.ce_loss(out4, torch.where(yb1[:,9:])[1])



        

        return (loss0 + loss1 + loss2 + loss3 + loss4) / 5
loss_func = MultiLoss()
# loss = loss_func(outs, *learner.yb); loss
# 'POSITIVE','negative_exam_for_pe','indeterminate',

# 'rv_lv_ratio_gte_1','rv_lv_ratio_lt_1', 'NEITHER'

# 'leftsided_pe','rightsided_pe','central_pe',

# 'chronic_pe','acute_and_chronic_pe','NEITHER'
bce_loss = BCEWithLogitsLossFlat()
# seq_cls_out, out1, out2, out3, out4 = outs

# yb0, yb1 = learner.yb
neg_pe_wgt = 0.0736196319

indeterminate_wgt = 0.09202453988



rv_lv_gte_1_wgt = 0.2346625767

rv_lv_lt_1_wgt = 0.0782208589



left_pe_wgt = 0.06257668712

right_pe_wgt = 0.06257668712

central_pe_wgt = 0.1877300613



chronic_wgt = 0.1042944785

acute_chronic_wgt = 0.1042944785



bce_loss = BCEWithLogitsLossFlat()



def metric(preds, yb0, yb1):

    

    seq_cls_out, out1, out2, out3, out4 = preds

    

    bs = out1.size(0)

    

    out1 = F.softmax(out1, 1)

    out2 = F.softmax(out2, 1)

    out3 = torch.sigmoid(out3)

    out4 = F.softmax(out4, 1)

    

    neg_pe_loss = F.binary_cross_entropy(out1[:,1], yb1[:,1].float())

    indeterminate_loss = F.binary_cross_entropy(out1[:,2], yb1[:,2].float())



    rv_lv_gte_1_loss = F.binary_cross_entropy(out2[:,0], yb1[:,3].float())

    rv_lv_lt_1_loss = F.binary_cross_entropy(out2[:,1], yb1[:,4].float())



    left_pe_wgt = F.binary_cross_entropy(out3[:,0], yb1[:,6].float())

    right_pe_wgt = F.binary_cross_entropy(out3[:,0], yb1[:,7].float())

    central_pe_wgt = F.binary_cross_entropy(out3[:,0], yb1[:,8].float())



    chronic_loss = F.binary_cross_entropy(out4[:,0], yb1[:,9].float())

    acute_chronic_loss = F.binary_cross_entropy(out4[:,1], yb1[:,10].float())



    

    tot_exam_loss = 0

    tot_exam_wgts = 0



    tot_exam_loss += neg_pe_loss*bs*neg_pe_wgt

    tot_exam_loss += indeterminate_loss*bs*indeterminate_wgt

    tot_exam_loss += rv_lv_gte_1_loss*bs*rv_lv_gte_1_wgt

    tot_exam_loss += rv_lv_lt_1_loss*bs*rv_lv_lt_1_wgt

    tot_exam_loss += left_pe_wgt*bs*left_pe_wgt

    tot_exam_loss += right_pe_wgt*bs*right_pe_wgt

    tot_exam_loss += central_pe_wgt*bs*central_pe_wgt

    tot_exam_loss += chronic_loss*bs*chronic_wgt

    tot_exam_loss += acute_chronic_loss*bs*acute_chronic_wgt



    tot_exam_wgts += bs*neg_pe_wgt

    tot_exam_wgts += bs*indeterminate_wgt

    tot_exam_wgts += bs*rv_lv_gte_1_wgt

    tot_exam_wgts += bs*rv_lv_lt_1_wgt

    tot_exam_wgts += bs*left_pe_wgt

    tot_exam_wgts += bs*right_pe_wgt

    tot_exam_wgts += bs*central_pe_wgt

    tot_exam_wgts += bs*chronic_wgt

    tot_exam_wgts += bs*acute_chronic_wgt



    tot_exam_loss, tot_exam_wgts = tot_exam_loss.item(), tot_exam_wgts.item()

    

    

    

    # Image-level weighted log loss (single batch)

    w_img = 0.07361963

    tot_img_loss = 0

    tot_img_wgts = 0

    for img_preds, img_targs in zip(seq_cls_out, yb0):

        mask = img_targs != targ_pad_idx

        img_preds = img_preds[mask]

        img_targs = img_targs[mask]



        n_imgs = sum(mask)



        qi = img_targs.float().mean()    

        img_loss = bce_loss(img_preds, img_targs)    

        wgt = w_img*qi

        tot_img_wgts += (wgt).item()*n_imgs

        tot_img_loss += (wgt*img_loss).item()*n_imgs



    

    return (tot_exam_loss + tot_img_loss) / (tot_exam_wgts + tot_img_wgts)
data = DataBlock(blocks=(SequenceBlock,SequenceTargetBlock,TargetBlock), 

                 n_inp=1, splitter=FuncSplitter(lambda o: o in valid_pids))

dls = data.dataloaders(list(train_pids)+list(valid_pids), bs=64)

model = MultiHeadedSequenceClassifier(dim=1024)

loss_func = MultiLoss()

learner = Learner(dls, model, loss_func=loss_func, metrics=[metric], cbs=[SaveModelCallback(fname="best_seqmodel")])
learner.validate()
learner.lr_find()
learner.fit_flat_cos(20, lr=0.01)
learner.save("init_seq_model")
learner.export("best_seqmodel_export.pkl")