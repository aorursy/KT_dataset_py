!pip install pytorch-lightning
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install git+git://github.com/lgvaz/mantisshrimp.git
from mantisshrimp.imports import *
from mantisshrimp import *
import pandas as pd
import albumentations as A
source = Path('../input/global-wheat-detection/')
df = pd.read_csv(source / "train.csv")
df.head()
class WheatParser(DefaultImageInfoParser, FasterRCNNParser):
    def __init__(self, df, source):
        self.df = df
        self.source = source
        self.imageid_map = IDMap()

    def __iter__(self):
        yield from self.df.itertuples()

    def __len__(self):
        return len(self.df)

    def imageid(self, o) -> int:
        return self.imageid_map[o.image_id]

    def filepath(self, o) -> Union[str, Path]:
        return self.source / f"{o.image_id}.jpg"

    def height(self, o) -> int:
        return o.height

    def width(self, o) -> int:
        return o.width

    def label(self, o) -> int:
        return 1

    def bbox(self, o) -> BBox:
        return BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))
data_splitter = RandomSplitter([.8, .2])
parser = WheatParser(df, source / "train")
train_rs, valid_rs = parser.parse(data_splitter)
show_record(train_rs[0], label=False)
train_tfm = AlbuTransform([A.Flip()])
train_ds = Dataset(train_rs, train_tfm)
valid_ds = Dataset(valid_rs)
class WheatModel(MantisFasterRCNN):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), 1e-3, momentum=0.9)
        return opt
model = WheatModel(2)
train_dl = model.dataloader(train_ds, shuffle=True, batch_size=4, num_workers=2)
valid_dl = model.dataloader(valid_ds, batch_size=4, num_workers=2)
trainer = Trainer(max_epochs=1, gpus=1)
trainer.fit(model, train_dl, valid_dl)