#!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install fastai==2.0.13
import torch
torch.cuda.is_available()
#from utils import *

#from fastai.vision.utils import *
from fastai.interpret import ClassificationInterpretation

from fastai.vision.all import *

from fastai.vision.widgets import *
from IPython.display import display,HTML
path=Path("../input")
files = get_image_files(path)

files
d=[1,2]
files[1,2]
files[1]
parent_label(files[1])
desserts = DataBlock(

    blocks=(ImageBlock, CategoryBlock), 

    get_items=get_image_files, 

    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    get_y=parent_label,

    item_tfms=Resize(128))
desserts=desserts.new(item_tfms=RandomResizedCrop(128,min_scale=0.3),

            batch_tfms=aug_transforms())
dataloader = desserts.dataloaders(path)
dataloader.valid.show_batch(max_n=4, nrows=1)
learn = cnn_learner(dataloader, resnet18, metrics=error_rate)

learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(4, nrows=4)
clean= ImageClassifierCleaner(learn)
clean
clean.delete()
clean.fns[0]
def changeLabel(cleanobj, path):

    for idx,label in cleanobj.change():

        shutil.move(str(cleanobj.fns[idx]), path/label)
def deleteLabel(cleanobj):

    for idx in cleanobj.delete():

        cleanobj.fns[idx].unlink
deleteLabel(clean)
learn.fine_tune(4)
test_file=files[0]

test_file
output_file=Path("/kaggle/working")

learn.path=output_file

learn.export()

#learn_inf = load_learner(output_file/'export.pkl')

#learn_inf.predict(test_file)