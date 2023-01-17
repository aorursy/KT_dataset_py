from fastai.vision.all import *
from fastai.tabular.all import *
from fastai.medical.imaging import *
source = '../input/rsna-str-pulmonary-embolism-detection'
files = os.listdir(source)
files
df = pd.read_csv(f'{source}/train.csv')
df.head()
get_x = lambda x:f'{source}/train/{x.StudyInstanceUID}/{x.SeriesInstanceUID}/{x.SOPInstanceUID}.dcm'
get_y = ColReader('pe_present_on_image')

blocks = (ImageBlock(cls=PILDicom), CategoryBlock)
set_seed(7)
pe = DataBlock(blocks=blocks,
                get_x=get_x,
                splitter=RandomSplitter(),
                item_tfms=[Resize(512)],
                get_y=get_y,
                batch_tfms=aug_transforms(size=512))
dls = pe.dataloaders(df[:1000], bs=16, n_workers=0)
dls.show_batch(max_n=4, nrows=1, ncols=4, figsize=(20,20))
class LungWindow(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        if isinstance(fn,bytes): im = pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn))
        if isinstance(fn,(Path,str)): im = dcmread(fn)
        scaled = np.array(im.windowed(l=-600, w=1500).numpy())*255
        scaled = scaled.astype(np.uint8)
        return cls(Image.fromarray(scaled))
class PEWindow(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        if isinstance(fn,bytes): im = pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn))
        if isinstance(fn,(Path,str)): im = dcmread(fn)
        scaled = np.array(im.windowed(l=100, w=700).numpy())*255
        scaled = scaled.astype(np.uint8)
        return cls(Image.fromarray(scaled))
class MedistinalWindow(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        if isinstance(fn,bytes): im = pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn))
        if isinstance(fn,(Path,str)): im = dcmread(fn)
        scaled = np.array(im.windowed(l=40, w=400).numpy())*255
        scaled = scaled.astype(np.uint8)
        return cls(Image.fromarray(scaled))
set_seed(7)
lung = DataBlock(blocks=(ImageBlock(cls=LungWindow), CategoryBlock),
                get_x=get_x,
                splitter=RandomSplitter(),
                item_tfms=[Resize(512)],
                get_y=get_y,
                batch_tfms=aug_transforms(size=512))

dls = lung.dataloaders(df[:1000], bs=16, n_workers=0)
dls.show_batch(max_n=4, nrows=1, ncols=4, figsize=(20,20))
set_seed(7)
pew = DataBlock(blocks=(ImageBlock(cls=PEWindow), CategoryBlock),
                get_x=get_x,
                splitter=RandomSplitter(),
                item_tfms=[Resize(512)],
                get_y=get_y,
                batch_tfms=aug_transforms(size=512))

dls = pew.dataloaders(df[:1000], bs=16, n_workers=0)
dls.show_batch(max_n=4, nrows=1, ncols=4, figsize=(20,20))
set_seed(7)
med = DataBlock(blocks=(ImageBlock(cls=MedistinalWindow), CategoryBlock),
                get_x=get_x,
                splitter=RandomSplitter(),
                item_tfms=[Resize(512)],
                get_y=get_y,
                batch_tfms=aug_transforms(size=512))

dls = med.dataloaders(df[:1000], bs=16, n_workers=0)
dls.show_batch(max_n=4, nrows=1, ncols=4, figsize=(20,20))
blocks = (
          ImageBlock(cls=LungWindow),
          ImageBlock(cls=PEWindow),
          ImageBlock(cls=MedistinalWindow),
          CategoryBlock

          )

getters = [
          get_x,
          get_x,
          get_x,
          ColReader('pe_present_on_image')
          ]

multiimage = DataBlock(blocks=blocks,
              getters=getters,
              item_tfms=Resize(256),
              batch_tfms=aug_transforms(size=256)
              )
multiimage.summary(df[:1000])
dls = multiimage.dataloaders(df[:1000], bs=16)
dls.show_batch(max_n=8, figsize=(7,7))