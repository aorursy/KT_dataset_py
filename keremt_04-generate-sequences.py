from fastai.vision.all import *
datapath = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")
train_df = pd.read_csv(datapath/'train.csv')
test_df = pd.read_csv(datapath/'test.csv')
imagepath = Path("/kaggle/input/rsna-str-pe-detection-jpeg-256/")
train_df.head()
trn_files = get_image_files(imagepath)
sids = [o.parent.parent.name for o in trn_files]
sopids = [o.stem.split("_")[1] for o in trn_files]
img_df = pd.DataFrame({"StudyInstanceUID":sids, "SOPInstanceUID":sopids, "fname":trn_files})
img_df.head()
train_df = train_df.merge(img_df, on=['StudyInstanceUID', 'SOPInstanceUID'])
train_df.head()
assert train_df['fname'].isna().sum() == 0
def get_dls(files, size=256, bs=128):
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=0.9)], 
            [lambda o: 0, Categorize()]]

    dsets = Datasets(files, tfms=tfms, splits=([0,1], [2,3]))

    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, after_batch=batch_tfms, num_workers=2)
    return dls
dls = get_dls(trn_files)
dls.c = 2
learn = cnn_learner(dls, xresnet34, pretrained=True)
learn.path = Path("/kaggle/input/rsnastrpecnnmodel/")
learn.load('xresnet34-256_3');
class EmbeddingHook:
    def __init__(self, m, csz=500000, n_init=0):
        self.embeddings = tensor([])
        self.m = m
        if len(m._forward_hooks) > 0: self.reset()
        self.hook = Hook(m, self.hook_fn, cpu=True)
        self.save_iter = n_init
        self.chunk_size = csz
    
    def hook_fn(self, m, inp, out): 
        "Stack and save computed embeddings"
        self.embeddings = torch.cat([self.embeddings, out])
        if self.embeddings.shape[0] > self.chunk_size:
            self.save()
            self.embeddings = tensor([])
    
    def reset(self): 
        self.m._forward_hooks = OrderedDict()
        
    def save(self): 
        torch.save(self.embeddings.to(torch.float16), f"train_embs-{self.save_iter}.pkl")
        self.save_iter += 1
emb_hook = EmbeddingHook(learn.model[1][1], n_init=0)
test_dl = learn.dls.test_dl(train_df['fname'])
test_dl.show_batch(max_n=16)
_, _ = learn.get_preds(dl=test_dl)
torch.save(emb_hook.embeddings.to(torch.float16), "train_embs-final.pkl")
train_df.to_csv("train.csv", index=False)
