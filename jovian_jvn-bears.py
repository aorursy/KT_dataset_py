!pip install -q fastai2

!pip install -q azure-cognitiveservices-search-imagesearch

!pip install -q jovian
#hide

# from utils import *

from fastai2.vision.widgets import *
from fastai2.vision.all import *

from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api

from msrest.authentication import CognitiveServicesCredentials as auth

def search_images_bing(key, term, min_sz=128):

    client = api('https://api.cognitive.microsoft.com', auth(key))

    return L(client.images.search(query=term, count=150, min_height=min_sz, min_width=min_sz).value)
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value = user_secrets.get_secret("bing")



key = secret_value
search_images_bing
results = search_images_bing(key, 'grizzly bear')

ims = results.attrgot('content_url')

len(ims)
bear_types = 'grizzly','black','teddy'

path = Path('bears')
if not path.exists():

    path.mkdir()

    for o in bear_types:

        dest = (path/o)

        dest.mkdir(exist_ok=True)

        results = search_images_bing(key, f'{o} bear')

        download_images(dest, urls=results.attrgot('content_url'))
fns = get_image_files(path)

fns
failed = verify_images(fns)

failed
failed.map(Path.unlink);
bears = DataBlock(

    blocks=(ImageBlock, CategoryBlock), 

    get_items=get_image_files, 

    splitter=RandomSplitter(valid_pct=0.3, seed=42),

    get_y=parent_label,

    item_tfms=Resize(128))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))

dls = bears.dataloaders(path)

dls.train.show_batch(max_n=4, nrows=1, unique=True)
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))

dls = bears.dataloaders(path)

dls.train.show_batch(max_n=8, nrows=2, unique=True)
bears = bears.new(

    item_tfms=RandomResizedCrop(224, min_scale=0.5),

    batch_tfms=aug_transforms())

dls = bears.dataloaders(path)
learn = cnn_learner(dls, resnet18, metrics=error_rate)

learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1)
cleaner = ImageClassifierCleaner(learn)

cleaner
#hide

for idx in cleaner.delete(): cleaner.fns[idx].unlink()

for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
learn.export()
path = Path()

path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
learn_inf.dls.vocab
learn_inf.predict(Path('bears/teddy/00000057.jpg'))
btn_upload = widgets.FileUpload()

btn_upload
img = PILImage.create(btn_upload.data[0])
out_pl = widgets.Output()

out_pl.clear_output()

with out_pl: display(img.to_thumb(128,128))

out_pl
pred,pred_idx,probs = learn_inf.predict(img)
lbl_pred = widgets.Label()

lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

lbl_pred
import jovian
!ls
jovian.commit(outputs='export.pkl')