!pip3 install --upgrade fastai

!pip install fast_tabnet # Not Required
from fastai.test_utils import *

show_install(True)
from fastai.tabular.all import *

from fast_tabnet.core import *

from pathlib import Path
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

seed_everything(1234)
root = "../input/titanic"

train = Path(root)/"train.csv"

test = Path(root )/"test.csv"

sub = Path(root )/"gender_submission.csv"
df = pd.read_csv(train)

dft = pd.read_csv(test)
df = pd.read_csv(train)

df.Cabin = df.Cabin.apply(lambda x:x  if type(x) == float else x[0])

df.head()
dep_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'Embarked', 'Ticket', 'Cabin']

cont_names = ['Age', 'Fare', 'SibSp', 'Parch']

procs = [Categorify, FillMissing, Normalize]
split_sample = np.random.choice(df.shape[0], 200)

dls = TabularDataLoaders.from_df(df, root, procs, cat_names, cont_names, y_names=dep_var, valid_idx=split_sample, bs=64, y_block=CategoryBlock)

learn = tabular_learner(dls, model_dir="/tmp/model/", metrics=[accuracy]).to_fp16()
dls.valid.show_batch()
learn.lr_find()
learn.fine_tune(3, 1e-2)
learn.lr_find()
learn.fit_one_cycle(5, lr_max=1e-3)
learn.lr_find()
learn.fit_one_cycle(10, lr_max=2e-7)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
dft = pd.read_csv(test)

dft.loc[dft.Fare.isnull(), 'Fare'] = np.median(df.Fare)
dft.Cabin = dft.Cabin.apply(lambda x:x  if type(x) == float else x[0])

dl = learn.dls.test_dl(dft, bs=64)
dlp, _  = learn.get_preds(dl=dl)
submission = pd.read_csv(sub)

submission.Survived = np.argmax(dlp, axis=1)

submission.to_csv('submission.csv', index=False)