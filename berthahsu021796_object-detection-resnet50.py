from google.colab import drive
drive.mount('/content/gdrive')
import pandas as pd
import numpy as np
df = pd.read_csv(F"/content/gdrive/My Drive/AI Project/Label/final label/plastic_label_0428.csv")
df.head()
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import sys, os

def plot(d, ID):
    print(ID)
    path = F"/content/gdrive/My Drive/AI Project/Data/plastic/"
    img_path = os.path.join(path, ID)
    image = Image.open(img_path).convert("RGB")
    
    for x in d:
      try:
        boxInfo = x['bbox']
        x1 = boxInfo['left']
        y1 = boxInfo['top']
        x2 = boxInfo['left']+boxInfo['width']
        y2 = boxInfo['top']+boxInfo['height']
        b_box = [x1,y1,x2,y2]
        draw = ImageDraw.Draw(image)
        draw.rectangle(b_box, outline="yellow", width=10)
        plt.imshow(image)
      except:
        pass
    plt.show()
for i, row in df.iterrows():
  if i == 89:
    d = eval(row['Label']).get('objects')
    plot(d, row['External ID'])
    break
def preprocess(csv_path):

  new_df = pd.DataFrame(None, columns = ['External ID','bbox'])
  df = pd.read_csv(csv_path)

  IDs = []
  Boxes = []
  
  for i, row in df.iterrows():
    d = eval(row['Label']).get('objects') 
    bbox = []
    if d is not None:
      for x in d:
        try:
          bbox.append(x['bbox'])
        except:
          pass
    if len(bbox) > 0:
      Boxes.append(bbox)
      IDs.append(row['External ID'])

  #print(len(IDs), len(Boxes), len(df))
  new_df['External ID'] = IDs
  new_df['bbox'] = Boxes
  return new_df
data = preprocess(F"/content/gdrive/My Drive/AI Project/Label/final label/plastic_label_0428.csv")
#data = preprocess(F"/content/gdrive/My Drive/AI Project/Label/plastic_label_0331.csv")
print(data.shape)
data.head()
def split(data, rate):

  n = len(data)
  n0 = int(round(rate*n))
  
  train_data = data[:n0]
  test_data = data[n0:]
  
  print(len(train_data), len(test_data))
  return train_data, test_data
train_data, test_data = split(data, 0.8)
import torch.utils
import torchvision
from torchvision import transforms as T
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys, os

sys.path.append(F'/content/gdrive/My Drive/AI Project/Source Code')
class MyDataset(object):

    def __init__(self, path, data, trans):
        self.path = path
        self.data = data
        self.trans = trans
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # load images
        ID = self.data.iloc[idx]['External ID']
        img_path = os.path.join(self.path, ID)
        #print(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        y_dim, x_dim, z_dim = img.shape
        #img_tensor = torch.from_numpy(np.array(img))

        box_list = self.data.iloc[idx]['bbox']
        num_objs = len(box_list)

        boxes = []
        areas = []
        
        for boxInfo in box_list:
          x1 = boxInfo['left']
          y1 = boxInfo['top']
          x2 = boxInfo['left']+boxInfo['width']
          y2 = boxInfo['top']+boxInfo['height']
          boxes.append([x1,y1,x2,y2])
          areas.append(boxInfo['height']*boxInfo['width'])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] =  labels = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        #print(img, target)
        if self.trans is not None:
              img = self.trans(img)

        dims = (x_dim, y_dim)
        return img, target, dims, ID
def get_transform():
    transformsList = []
    transformsList.append(T.ToTensor())
    return T.Compose(transformsList)
import utils

trans = get_transform()
path = F"/content/gdrive/My Drive/AI Project/Data/plastic/"

dataset = MyDataset(path, train_data, trans)
train_data_loader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size=1, num_workers=4, pin_memory=True, collate_fn=utils.collate_fn)

dataset = MyDataset(path, test_data, trans)
test_data_loader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size=1, num_workers=4, pin_memory=True, collate_fn=utils.collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
import math
import sys
import time
import torch

def train_model(model, optimizer, data_loader):

  start_time = time.time()
  running_loss = 0.0

  model.train()

  for images, targets, dims, ID in data_loader:
    optimizer.zero_grad()

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    loss_value = losses_reduced.item()
    running_loss += loss_value

    if not math.isfinite(loss_value):
          print("Loss is {}, stopping training".format(loss_value))
          print(loss_dict_reduced)
          sys.exit(1)
          
    losses.backward()
    optimizer.step()

  end_time = time.time()
  running_loss /= len(data_loader)

  print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
  return running_loss
SCORE_THRES = 0.5      
def boundingBoxes(results, add_scores = True):
  """ Return array of bounding boxes (only consider scores >= 0.5) """

  boxes_of_interest = []

  try:
    boxes = results[0]['boxes'].cpu().detach().numpy()
    if not add_scores:
      return boxes
    if add_scores:
      scores = results[0]['scores'].cpu().detach().numpy()
      for i, s in enumerate(scores):
          if s >= SCORE_THRES:
            boxes_of_interest.append([boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]])
    #else:
      #for i, s in enumerate(boxes):
        #boxes_of_intereset.append([boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]])
  except:
    pass
  
  return boxes_of_interest
def recall_score (img_dim_x, img_dim_y, true_boxes, predicted_boxes):

  true = np.zeros((img_dim_x, img_dim_y), dtype=bool)
  pred = np.zeros((img_dim_x, img_dim_y), dtype=bool)
  for b in true_boxes:
    for i in range(int(max(0, round(b[0]))), int(min(img_dim_x-1, round(b[2])+1))):   # x1 to x2; borders included
      for j in range(int(max(0, round(b[1]))), int(min(img_dim_y-1, round(b[3])+1))): # y1 to y2; borders included
        true[i][j] = True
  for b in predicted_boxes:
    for i in range(int(max(0, round(b[0]))), int(min(img_dim_x-1, round(b[2])+1))):   # x1 to x2; borders included
      for j in range(int(max(0, round(b[1]))), int(min(img_dim_y-1, round(b[3])+1))): # y1 to y2; borders included
        pred[i][j] = True

  intersect = np.logical_and(true, pred)
  area_true = np.count_nonzero(true)
  area_pred = np.count_nonzero(pred)
  area_both = np.count_nonzero(intersect)
  return area_both / float(area_true)
def jaccard_score (box_a, box_b):
  x_1 = max(box_a[0], box_b[0])
  y_1 = max(box_a[1], box_b[1])
  x_2 = min(box_a[2], box_b[2])
  y_2 = min(box_a[3], box_b[3])

  intersect = max(0, x_2 - x_1 + 1) * max(0, y_2 - y_1 + 1) # line itself is also in the box
  area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
  area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

  return intersect / float(area_a + area_b - intersect)
def test_model(model, data_loader):

  start_time = time.time()
  average_acc = 0.0

  model.eval()

  for images, targets, dims, ID in data_loader:

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    results = model(images)
    predict_bbox = boundingBoxes(results)
    true_bbox = boundingBoxes(targets, False)
    score = recall_score(dims[0][0], dims[0][1], true_bbox, predict_bbox)
    average_acc += score
    
  end_time = time.time()
  average_acc /= len(data_loader)

  print('Testing Accuracy: ', average_acc, 'Time: ', end_time - start_time, 's')
  return average_acc
import warnings
warnings.filterwarnings("ignore")

score = test_model(model, test_data_loader)
print("Baseline Model (before custom training) Accuracy: ", score)
import warnings
warnings.filterwarnings("ignore")

# let's train it for 15 epochs
num_epochs = 15

for epoch in range(num_epochs):
      print("epoch "+str(epoch)+":")
      train_model(model, optimizer, train_data_loader)
      test_model(model, test_data_loader)
      # update the learning rate
      lr_scheduler.step()
      print("-------------------------------------------------")
      
# save only model parameters
PATH = "./detection_model.pt"
torch.save(model.state_dict(), PATH)

# load a saved model parameters
#model_save_name = 'classifier_v2.pt'
#path = F"/content/gdrive/My Drive/{model_save_name}"

#model.load_state_dict(torch.load(path))

## Less optimised approaches ->
# saving the entire model
#torch.save(model, PATH)
from PIL import Image, ImageDraw
import random

def displayPrediction(model, test_data, num = 10):

  fig=plt.figure(figsize=(20, 20))
  columns = 2
  rows = 5

  i = 0

  for t in range(num):

    path = F"/content/gdrive/My Drive/AI Project/Data/plastic/"
    
    b_boxes_true = []
    
    idx = random.randint(0, len(test_data))
    img_path = os.path.join(path, test_data.iloc[idx]['External ID'])
    
    box_list = test_data.iloc[idx]['bbox']
    for box in box_list:
        x1 = box['left']
        y1 = box['top']
        x2 = box['left']+box['width']
        y2 = box['top']+box['height']
        b_boxes_true.append([x1, y1, x2, y2])

    model.eval()
    image = Image.open(img_path).convert("RGB")
    img = np.array(image)
    y_dim, x_dim, z_dim = img.shape
    predictions = model([trans(img).to(device)])

    draw = ImageDraw.Draw(image)
    b_boxes_pred = boundingBoxes(predictions)
    for b in b_boxes_true:
      draw.rectangle(b, outline="yellow", width=10)
    for b in b_boxes_pred:
      draw.rectangle(b, outline="red", width = 10)

    fig.add_subplot(rows, columns, i+1)
    i += 1
    plt.imshow(image)
    plt.title("Recall Score: %.2f" % recall_score(x_dim, y_dim, b_boxes_true, b_boxes_pred))

  plt.show()
displayPrediction(model, test_data)