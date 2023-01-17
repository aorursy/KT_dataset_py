!pip install -q fastai2
from fastai2.tabular.all import *
input_dir = Path("/kaggle/input/liverpool-ion-switching")
train_df = pd.read_csv(input_dir/"train.csv")

test_df  = pd.read_csv(input_dir/"test.csv")
train_df.head()
test_df.tail()
size = 500_000



def prepare(df):

    start = 0

    for b in range(size, len(df)+1, size):

        df.loc[start:b-1,"time"] = list(range(size))

        start += size
prepare(train_df)
prepare(test_df)
train_df.tail()
BS=50_000
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



seed_everything(42)
# cont_names = ['time', 'signal']

# procs = [Categorify, FillMissing, Normalize]

# splits = RandomSplitter(seed=43)(range_of(train_df))

# to = TabularPandas(train_df, procs, cont_names, y_names="open_channels", splits=splits, y_block=CategoryBlock)
# dls = TabDataLoader(to, bs=BS, shuffle=False, drop_last=False)
# dls = to.dataloaders(bs=BS, shuffle_train=False)
dls.show_batch()
dls = TabularDataLoaders.from_df(train_df, path=".", y_names="open_channels",

                                 cont_names = ['time', 'signal'], y_block=CategoryBlock,

                                 procs = [Categorify, FillMissing, Normalize], bs=BS,

                                 shuffle_train=False)
# dls.show_batch()
learn = tabular_learner(dls, metrics=F1Score(average="macro"))
# learn.lr_find()
lr = 3e-3
learn.fit_one_cycle(10, lr_max=lr)
learn.fit_one_cycle(10, lr_max=lr/10)
dl = learn.dls.test_dl(test_df)
preds, _ = learn.get_preds(dl=dl)
subm = pd.read_csv(input_dir/"sample_submission.csv")
subm["open_channels"] = preds.argmax(1)
subm.to_csv("submission.csv", float_format='%0.4f', index=False)
pd.read_csv("submission.csv")