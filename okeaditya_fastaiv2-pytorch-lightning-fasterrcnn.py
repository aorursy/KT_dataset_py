!pip install -q 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install -q fastai2
!pip install -q pytorch_lightning
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

    def labels(self, o) -> List[int]:
        return [1]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))]
data_splitter = RandomSplitter([.8, .2])
parser = WheatParser(df, source / "train")
train_rs, valid_rs = parser.parse(data_splitter)
show_record(train_rs[0], label=False)
train_tfm = AlbuTransform([A.Flip()])
train_ds = Dataset(train_rs, train_tfm)
valid_ds = Dataset(valid_rs)
from mantisshrimp.models.rcnn.faster_rcnn import *
from mantisshrimp.models.rcnn import *
# Using the FasterRCNN Basic Model with resnet50 fpn backbone
model = faster_rcnn.model(num_classes=2)
from mantisshrimp import backbones
resnet_101_backbone = backbones.resnet_fpn.resnet101(pretrained=True)
resnet_152_backbone = backbones.resnet_fpn.resnet152(pretrained=True)
vgg11_backbone = backbones.vgg.vgg11(pretrained=True)
# metrics = []
# metrics += [COCOMetric(valid_rs, bbox=True, mask=False, keypoint=False)]
train_dl = faster_rcnn.dataloaders.train_dataloader(train_ds, batch_size=4, num_workers=4, shuffle=True)
valid_dl = faster_rcnn.dataloaders.valid_dataloader(valid_ds, batch_size=4, num_workers=4, shuffle=False)
# This creates the default model with resnet50 fpn backbone
# model = faster_rcnn.model(num_classes=2)

# To create model with backbones 
model = faster_rcnn.model(num_classes=2, backbone=resnet_152_backbone)
learner = faster_rcnn.fastai.learner([train_dl, valid_dl], model)
learner.fine_tune(2, lr=1e-4)
from torchvision.models.detection.rpn import AnchorGenerator
# #Imagenet mean and std it will taken automatically if not explicity given
# ft_mean = [0.485, 0.456, 0.406] #ImageNet mean
# ft_std = [0.229, 0.224, 0.225] #ImageNet std
# anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
# aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
# ft_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# ft_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#                 featmap_names=['0', '1', '2', '3'],
#                 output_size=7,
#                 sampling_ratio=2)
# # ft_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256)), aspect_ratios=((0.5, 1.0)))
# # ft_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
train_dl = faster_rcnn.dataloaders.train_dataloader(train_ds, batch_size=4, num_workers=4, shuffle=True)
valid_dl = faster_rcnn.dataloaders.valid_dataloader(valid_ds, batch_size=4, num_workers=4, shuffle=False)
class LightModel(faster_rcnn.lightning.ModelAdapter):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), 2e-4, momentum=0.9)
        return opt

model = faster_rcnn.model(num_classes=2, backbone=resnet_152_backbone)
light_model = LightModel(model=model) #metrics=metrics)
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=2, gpus=1)
trainer.fit(light_model, train_dl, valid_dl)
# Save the model as same as you would do for a Pytorch model
# You can also use lightning features to even automate this.

torch.save(light_model.state_dict(), "mantiss_faster_rcnn.pt")
detection_threshold = 0.45
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)
# Just using the Fastai model for now. You can replace this with Light model as well.
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