! pip install fastai2 nbdev
# Numpy and pandas by default assume a narrow screen - this fixes that

from fastai2.vision.all import *

from nbdev.showdoc import *

from ipywidgets import widgets

from pandas.api.types import CategoricalDtype



import matplotlib as mpl

# mpl.rcParams['figure.dpi']= 200

mpl.rcParams['savefig.dpi']= 200

mpl.rcParams['font.size']=12



set_seed(42)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

pd.set_option('display.max_columns',999)

np.set_printoptions(linewidth=200)

torch.set_printoptions(linewidth=200)



import graphviz

def gv(s): return graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')



def get_image_files_sorted(path, recurse=True, folders=None): return get_image_files(path, recurse, folders).sorted()





def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):

    x = torch.linspace(min,max)

    fig,ax = plt.subplots(figsize=figsize)

    ax.plot(x,f(x))

    if tx is not None: ax.set_xlabel(tx)

    if ty is not None: ax.set_ylabel(ty)

    if title is not None: ax.set_title(title)



from sklearn.tree import export_graphviz



def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,

                      special_characters=True, rotate=False, precision=precision, **kwargs)

    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))





from scipy.cluster import hierarchy as hc



def cluster_columns(df, figsize=(10,6), font_size=12):

    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)

    corr_condensed = hc.distance.squareform(1-corr)

    z = hc.linkage(corr_condensed, method='average')

    fig = plt.figure(figsize=figsize)

    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)

    plt.show()
INPUT = "/kaggle/input/plant-pathology-2020-fgvc7/"

OUTPUT = "/kaggle/working"
df_train = pd.read_csv( Path(INPUT)/"train.csv")

df_train.head()
df_train.set_index('image_id',inplace=True)



df_train = df_train[df_train==1].stack().reset_index().drop(0,1)

df_train['image_id'] = df_train['image_id']+".jpg"

df_train.rename(columns = {'level_1':'class'},inplace = True)

df_train.head()
data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),

                   splitter=RandomSplitter(),

                   get_x=ColReader('image_id', pref=Path(INPUT)/"images"),

                   get_y=ColReader('class'),

                   item_tfms=Resize(224),

                   batch_tfms=aug_transforms(flip_vert=True, ))

dls = data_block.dataloaders(df_train)

dls.show_batch()
learn = cnn_learner(dls, xresnet50, metrics=error_rate)

learn.fine_tune(5)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(3, nrows=2)
learn.export()
learn_inf = load_learner(Path(OUTPUT)/'export.pkl')

learn_inf.predict (Path("/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1.jpg"))
preds, targs = learn.tta(ds_idx=1, n=4)
df_test = pd.read_csv( Path(INPUT)/"test.csv")

df_test['image_id'] = df_test['image_id'] + ".jpg"

df_test.head()

test_dl = dls.test_dl(df_test)

test_dl

test_preds, _ = learn.get_preds(dl=test_dl)
test_preds[0].tolist()
df_pred = pd.DataFrame(test_preds, columns=['healthy','multiple_diseases','rust','scab'])

df_pred.head()
df_sub = pd.read_csv( Path(INPUT)/"sample_submission.csv")

df_sub.head()
df_sub['healthy'] = df_pred['healthy']

df_sub['multiple_diseases'] = df_pred['multiple_diseases']

df_sub['rust'] = df_pred['rust']

df_sub['scab'] = df_pred['scab']

df_sub.head()
df_sub.to_csv(OUTPUT +"/sample_submission.csv",index=False)