#hide

from utils import *

from fastai2.vision.widgets import *
key = '04328cd4d18b4dae8efd3fb90330a1d3'
search_images_bing
results = search_images_bing(key, 'grizzly bear')

ims = results.attrgot('content_url')

len(ims)
#hide

ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
dest = 'images/grizzly.jpg'

download_url(ims[0], dest)
im = Image.open(dest)

im.to_thumb(128,128)
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
path.ls()
dls
dls.valid.show_batch(max_n=4, rows=1)
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, rows=1)
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, rows=1)
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))

dls = bears.dataloaders(path)

dls.train.get_idxs = lambda: Inf.ones

dls.train.show_batch(max_n=4, rows=1)
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))

dls = bears.dataloaders(path)

dls.train.get_idxs = lambda: Inf.ones

dls.train.show_batch(max_n=8, rows=2)
bears = bears.new(

    item_tfms=RandomResizedCrop(224, min_scale=0.5),

    batch_tfms=aug_transforms())

dls = bears.dataloaders(path)
learn = cnn_learner(dls, resnet18, metrics=error_rate)

learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(5, rows=1)
cleaner = ImageClassifierCleaner(learn)

cleaner
learn.export()
path = Path()

path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
learn_inf.predict('images/grizzly.jpg')
learn_inf.dls.vocab
btn_upload = widgets.FileUpload()

btn_upload
#hide

# For the book, we can't actually click an upload button, so we fake it

btn_upload = SimpleNamespace(data = ['images/grizzly.jpg'])
img = PILImage.create(btn_upload.data[-1])
out_pl = widgets.Output()

out_pl.clear_output()

with out_pl: display(img.to_thumb(128,128))

out_pl
pred,pred_idx,probs = learn_inf.predict(img)
lbl_pred = widgets.Label()

lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

lbl_pred
btn_run = widgets.Button(description='Classify')

btn_run
def on_click_classify(change):

    img = PILImage.create(btn_upload.data[-1])

    out_pl.clear_output()

    with out_pl: display(img.to_thumb(128,128))

    pred,pred_idx,probs = learn_inf.predict(img)

    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'



btn_run.on_click(on_click_classify)
#hide

#Putting back btn_upload to a widget for next cell

btn_upload = widgets.FileUpload()
VBox([widgets.Label('Select your bear!'), 

      btn_upload, btn_run, out_pl, lbl_pred])
!pip install jovian --upgrade
import jovian
jovian.commit()