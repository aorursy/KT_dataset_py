!pip install -q 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
! pip -q install git+git://github.com/lgvaz/mantisshrimp.git
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

    def label(self, o) -> List[int]:
        return [1]

    def bbox(self, o) -> List[BBox]:
        return [BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))]
data_splitter = RandomSplitter([.8, .2])
parser = WheatParser(df, source / "train")
train_rs, valid_rs = parser.parse(data_splitter)
show_record(train_rs[0], label=False)
train_tfm = AlbuTransform([A.Flip()])
train_ds = Dataset(train_rs, train_tfm)
valid_ds = Dataset(valid_rs)
from mantisshrimp.models.mantis_rcnn import *
resnet_101_backbone = MantisFasterRCNN.get_backbone_by_name("resnet101", fpn=True, pretrained=True)
resnet_152_backbone = MantisFasterRCNN.get_backbone_by_name("resnet152", fpn=True, pretrained=True)
mobibenet_v2_backbone = MantisFasterRCNN.get_backbone_by_name("mobilenet", fpn=False, pretrained=True)
class WheatModel(MantisFasterRCNN):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), 1e-3, momentum=0.9)
        return opt
model = WheatModel(n_class=2, backbone=resnet_101_backbone)
train_dl = model.dataloader(train_ds, shuffle=True, batch_size=8, num_workers=2)
valid_dl = model.dataloader(valid_ds, batch_size=8, num_workers=2)
trainer = Trainer(max_epochs=2, gpus=1)
trainer.fit(model, train_dl, valid_dl)
# Save the model as same as you would do for a Pytorch model
# You can also use lightning features to even automate this.

torch.save(model.state_dict(), "mantiss_faster_rcnn.pt")
detection_threshold = 0.45
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)
model.eval()
model.to(device)
detection_threshold = 1e-8
results = []
device = 'cuda'
for images in os.listdir("../input/global-wheat-detection/test/"):
    image_path = os.path.join("../input/global-wheat-detection/test/", images)
    image = cv2.imread(image_path)
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    model
#     print(image.shape)
    with torch.no_grad():
        outputs = model(image)
    
#     print(outputs)

    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    scores = scores[scores >= detection_threshold]
    image_id = images[:-3]

    result = {
        'image_id': image_id,
        'PredictionString': format_prediction_string(boxes, scores)
    }

    results.append(result)
#     break

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()
test_df.to_csv('submission.csv', index=False)