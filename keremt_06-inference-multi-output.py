!cp ../input/gdcm-conda-install/gdcm.tar .

!tar  -xzf gdcm.tar

!conda install -q --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
# !pip install -qq ../input/lungmask/lungmask-master/lungmask-master/
from fastai.vision.all import *

from fastai.medical.imaging import *
datapath = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")

cnnmodelpath = Path("/kaggle/input/rsnastrpecnnmodel/")

seqmodelpath = Path("/kaggle/input/rsnastrpeseqmodel/")

test_df = pd.read_csv(datapath/'test.csv')

sub_df = pd.read_csv(datapath/'sample_submission.csv')
[o for o in sub_df['id'].values if "df06fad17bc3" in o]
test_study_dirnames = [datapath/'test'/o for o in test_df['StudyInstanceUID'].unique()]

study_dirname = test_study_dirnames[0]
# RGB windows

lung_window = (1500, -600)

pe_window = (700, 100)

mediastinal_window = (400, 40)

windows = (lung_window, pe_window, mediastinal_window)



def read_dcm_img(dcm, windows=windows):

    "Read single slice in RGB"

    return torch.stack([dcm.windowed(*w) for w in windows])
# Load CNN model

def get_dls(tensors, size=256, bs=128):

    "Get study dataloader"

    tfms = [[RandomResizedCropGPU(size, min_scale=0.9)], []]



    dsets = Datasets(tensors, tfms=tfms, splits=([0,1], [2,3]))



    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    dls = dsets.dataloaders(bs=bs, after_batch=batch_tfms, num_workers=2)

    return dls



dls = get_dls(torch.zeros(4, 3, 224, 224), bs=32)

dls.c = 2

learn = cnn_learner(dls, xresnet34, pretrained=False, loss_func=nn.CrossEntropyLoss())

learn.path = Path("/kaggle/input/rsnastrpecnnmodel/")

learn.load('xresnet34-256_3');
input_pad_idx = None
class MultiHeadedSequenceClassifier(Module):

    "dim: input sequence feature dim"

    def __init__(self, input_pad_idx=input_pad_idx, dim=1024):

        

        store_attr('input_pad_idx')

        self.lstm1 = nn.LSTM(dim+5, dim//16, bidirectional=True)

        

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

    

    def predict(self, x):

        

        # sequence outs

        x, _ = self.lstm1(x) 

        seq_cls_out = self.seq_cls_head(x).squeeze(-1)

        

        pooled_x = torch.cat([x.mean(1), x.max(1).values], dim=1)

        



        # 'POSITIVE','negative_exam_for_pe','indeterminate'

        out1 = self.pe_head(pooled_x)



        # 'rv_lv_ratio_gte_1','rv_lv_ratio_lt_1', 'NEITHER'

        out2 = self.rv_lv_head(pooled_x)



        # 'leftsided_pe','rightsided_pe','central_pe',

        out3 = self.pe_position_head(pooled_x)



        # 'chronic_pe','acute_and_chronic_pe','NEITHER'

        out4 = self.chronic_pe_head(pooled_x)



        return (seq_cls_out, out1, out2, out3, out4)
# Load Sequence Model

seqmodel = MultiHeadedSequenceClassifier()

seqmodel.load_state_dict(torch.load("/kaggle/input/rsnastrpeseqmodel/models/best_seqmodel.pth"));

device = default_device()

seqmodel.eval()

seqmodel.to(device);
meta_mean_std = {'img_min': (-1409.7525910396214, 920.6624071834135),

                 'img_max': (2997.565154356599, 1375.5195189199717),

                 'img_mean': (159.1868599739921, 280.4988584140103),

                 'img_std': (916.7543430215497, 378.53540952883),

                 'scaled_position': (0.5078721739409208, 0.29139548181397373)}
class EmbeddingHook:

    def __init__(self, m):

        self.embeddings, self.m = tensor([]).to(device), m

        if len(m._forward_hooks) > 0: self.reset()

        self.hook = Hook(m, self.hook_fn, cpu=False)

       

    def hook_fn(self, m, inp, out): 

        "Stack and save computed embeddings"

        self.embeddings = torch.cat([self.embeddings, out])

    

    def reset(self): 

        self.m._forward_hooks = OrderedDict()
meta_feats = ['img_min', 'img_max', 'img_mean', 'img_std', 'scaled_position']
def minmax_scaler(o): return (o - min(o))/(max(o) - min(o))
# from time import time

# study_dirname = np.random.choice(test_study_dirnames)

# s = time()

# # get metadata

# dcmfiles = get_dicom_files(study_dirname)

# dcm_metadf = (pd.DataFrame.from_dicoms(dcmfiles, window=pe_window)

#                           .sort_values(['ImagePositionPatient2'])

#                           .reset_index(drop=True))

# study_fnames = dcm_metadf['fname'].values

# sop_ids = [Path(o).stem for o in study_fnames]

# e = time()

# print(e-s)



# s = time()

# # get ordered imgs

# dcm_ds = [Path(o).dcmread() for o in study_fnames]

# imgs = torch.stack([read_dcm_img(o) for o in dcm_ds])

# e = time()

# print(e-s)



# imgs.shape



# s = time()

# hook = EmbeddingHook(learn.model[1][1])

# test_dl = dls.test_dl(imgs.numpy(), bs=32)

# cnn_preds,_ = learn.get_preds(dl=test_dl)

# features = hook.embeddings.unsqueeze(0)

# e = time()

# print(e-s)



# s = time()

# dcm_metadf['scaled_position'] = (dcm_metadf.groupby('StudyInstanceUID')['ImagePositionPatient2']

#                                            .apply(minmax_scaler))

# for f in meta_feats: dcm_metadf[f] = (dcm_metadf[f] - meta_mean_std[f][0]) / meta_mean_std[f][1]

# meta_features = tensor(dcm_metadf[meta_feats].to_numpy()).to(device)

# multi_preds = to_detach(seqmodel.predict(torch.cat([features, meta_features[None, ...]], dim=-1)))

# e = time()

# print(e-s)
def predict_study(study_dirname):

    # get metadata

    dcmfiles = get_dicom_files(study_dirname)

    dcm_metadf = (pd.DataFrame.from_dicoms(dcmfiles, window=pe_window)

                              .sort_values(['ImagePositionPatient2'])

                              .reset_index(drop=True))

    study_fnames = dcm_metadf['fname'].values

    sop_ids = [Path(o).stem for o in study_fnames]



    # get ordered imgs

    dcm_ds = [Path(o).dcmread() for o in study_fnames]

    imgs = torch.stack([read_dcm_img(o) for o in dcm_ds])



    # get predictions

    with torch.no_grad():

        hook = EmbeddingHook(learn.model[1][1])

        test_dl = dls.test_dl(imgs.numpy(), bs=32)

        cnn_preds,_ = learn.get_preds(dl=test_dl)

        features = hook.embeddings.unsqueeze(0)

        dcm_metadf['scaled_position'] = (dcm_metadf.groupby('StudyInstanceUID')['ImagePositionPatient2']

                                                   .apply(minmax_scaler))

        for f in meta_feats: dcm_metadf[f] = (dcm_metadf[f] - meta_mean_std[f][0]) / meta_mean_std[f][1]

        meta_features = tensor(dcm_metadf[meta_feats].to_numpy()).to(device)

        multi_preds = to_detach(seqmodel.predict(torch.cat([features, meta_features[None, ...]], dim=-1)))

    

    return (multi_preds, sop_ids, cnn_preds)
def get_study_probas(sid, seq_cls_out, out1, out2, out3, out4):

    sub_res = []

    # image probas

    for sopid, p in zip(sop_ids, to_np(seq_cls_out.sigmoid()[0])):

        sub_res.append((sopid, p))



    # exam probas

    pos_pe_proba, neg_pe_proba, ind_pe_proba = to_np(out1[0].softmax(0))

    rvlv_gte, rvlv_lt, rvlv_none = to_np(out2[0].softmax(0))

    left_pe, right_pe, central_pe = to_np(torch.sigmoid(out3[0]))

    chronic, acute_chronic, chronic_none = to_np(out4[0].softmax(0))

    sub_res += [(f"{sid}_negative_exam_for_pe", neg_pe_proba),

                (f"{sid}_indeterminate", ind_pe_proba),

                (f"{sid}_rv_lv_ratio_gte_1", rvlv_gte),

                (f"{sid}_rv_lv_ratio_lt_1", rvlv_lt),

                (f"{sid}_leftsided_pe", left_pe),

                (f"{sid}_rightsided_pe", right_pe),

                (f"{sid}_central_pe", central_pe),

                (f"{sid}_chronic_pe", chronic),

                (f"{sid}_acute_and_chronic_pe", acute_chronic)]

    return sub_res

do_full = False

n = 20



if Path('../input/rsna-str-pulmonary-embolism-detection/train').exists() and not do_full: 

    test_study_dirnames = [datapath/'test'/o for o in test_df['StudyInstanceUID'].unique()]

    test_study_dirnames = np.random.choice(test_study_dirnames, n, replace=False)



sub_res = []

for study_dirname in test_study_dirnames:

    (seq_cls_out, out1, out2, out3, out4), sop_ids, cls_preds = predict_study(study_dirname)

    study_res = get_study_probas(study_dirname.stem, seq_cls_out, out1, out2, out3, out4)

    sub_res += study_res



final_sub_df = pd.DataFrame(sub_res, columns=['id', 'label'])

final_sub_df.to_csv("submission.csv", index=False)