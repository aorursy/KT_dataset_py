# test_dir = "/kaggle/input/hiaasq"
# test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
# predictions = predict_on_video_set(test_videos, num_workers=4)
import os, sys, time
import cv2
import numpy as np
import pandas as pd
import random
from random import randint
from PIL import ImageFilter, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.append('../input/efficientnet')
sys.path.append('../input/imutils/imutils-0.5.3')
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/helpers")
sys.path.insert(0, "/kaggle/input/timmmodels") 

import timm
from imutils.video import FileVideoStream 
from efficientnet import EfficientNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

%matplotlib inline
import matplotlib.pyplot as plt
test_dir = "/kaggle/input/czcasa"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from blazeface import BlazeFace
facedet = BlazeFace().to(device)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy") #noidea
_ = facedet.train(False)
input_size = 256
from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
def disable_grad(model):#noidea
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model
        

def weight_preds(preds, weights):#noidea
    final_preds = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if len(final_preds) != len(preds[i]):
                final_preds.append(preds[i][j] * weights[i])
            else:
                final_preds[j] += preds[i][j] * weights[i]
                
    return torch.FloatTensor(final_preds)
from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 120

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video) #get_frames(x, batch_size=frames_per_video)#noidea
face_extractor = FaceExtractor(video_read_fn, facedet)
class MetaModel(nn.Module):#noidea
    def __init__(self, models=None, device='cuda:0', extended=False):
        super(MetaModel, self).__init__()
        
        self.extended = extended
        self.device = device
        self.models = models
        self.len = len(models)
        
        if self.extended:
            self.bn = nn.BatchNorm1d(self.len)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(self.len, 1)
        
    def forward(self, x):
        x = torch.cat(tuple(x), dim=1)
        
        if self.extended:
            x = self.bn(x)
            x = self.relu(x)
            #x = self.dropout(x)
            
        x = self.fc(x)
        
        return x
MODELS_PATH = "/kaggle/input/deepfake-detection-model-20k/"
WEIGTHS_EXT = '.pth'

models = []
weigths = []
    
raw_data_stack = \
[
    ['0.8548137313946486 0.3376769562025044', 'efficientnet-b2'],
#     ['EfficientNetb3 0.8573518024606384 0.34558522378585194', 'efficientnet-b3'],
#     ['EfficientNetb4 0.8579110384582294 0.3383911053075265', 'efficientnet-b4'],
#     ['EfficientNet6 0.8602770369095758 0.33193617861157143', 'efficientnet-b6'],
#     ['EfficientNetb0 t2 0.8616966359803837 0.3698434531609828', 'efficientnet-b0'],
#     ['EfficientNetb1 t2 0.8410909403768391 0.36058002083572327', 'efficientnet-b1'],
#     ['EfficientNetb2 t2 0.8659554331928073 0.35598630783834084', 'efficientnet-b2'],
#     ['EfficientNetb3 t2 0.8486191172674868 0.3611779548592305', 'efficientnet-b3'],
#     ['EfficientNetb3 0.8635894347414609 0.328333642473084', 'efficientnet-b3'],
#     ['EfficientNetb6 0.8593736556826981 0.32286693639934694', 'efficientnet-b6'],
    
#     ['tf_efficientnet_b1_ns 0.8571367116923342 0.3341234226295108', 'tf_efficientnet_b1_ns'],
#     ['tf_efficientnet_b3_ns 0.8712466660930913 0.3277394129117183', 'tf_efficientnet_b3_ns'],
#     ['tf_efficientnet_b4_ns 0.8708595027101437 0.3152573955405342', 'tf_efficientnet_b4_ns'],
#     ['tf_efficientnet_b6_ns 0.8733115374688118 0.3156576980666498', 'tf_efficientnet_b6_ns'],
]

stack_models = []

for raw_model in raw_data_stack:#noidea
    checkpoint = torch.load( MODELS_PATH + raw_model[0] + WEIGTHS_EXT, map_location=device)
    
    if '-' in raw_model[1]:
        model = EfficientNet.from_name(raw_model[1])
        model._fc = nn.Linear(model._fc.in_features, 1)
    else:
        model = timm.create_model(raw_model[1], pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    
    model.load_state_dict(checkpoint)
    _ = model.eval()
    _ = disable_grad(model)
    model = model.to(device)
    stack_models.append(model)

    del checkpoint, model
    

# meta_models = \
# [
#     ['MetaModel 0.30638167556896007', slice(4, 8), False, 0.37780],
#     ['MetaModel 0.2919331893755284', slice(0, 4), False, 0.33357],
#     ['MetaModel 0.30281482560578044', slice(0, 8, None), True, 0.34077],
#     ['MetaModel 0.26302117601197256', slice(0, 10, None), False, 0.35134],
#     ['MetaModel 0.256337642808031', slice(10, 14, None), False, 0.32698],
# ]

# for meta_raw in meta_models:#noidea

#     checkpoint = torch.load(MODELS_PATH + meta_raw[0] + WEIGTHS_EXT, map_location=device)
    
#     model = MetaModel(models=raw_data_stack[meta_raw[1]], extended=meta_raw[2]).to(device)
    
#     model.load_state_dict(checkpoint)
#     _ = model.eval()
#     _ = disable_grad(model)
#     model.to(device)
#     models.append(model)
#     weigths.append(meta_raw[3])

#     del model, checkpoint
    
# total = sum([1-score for score in weigths])
# weigths = [(1-score) / total for score in weigths]#noidea
def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)
        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)

        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
#             imgplot = plt.imshow(faces[10][0])
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            
            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    resized_face = cv2.resize(face, (input_size, input_size))
                    
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1
            
            del faces
            imgplot = plt.imshow(x[10])
            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction
                with torch.no_grad():
                    y_pred = 0
                    stacked_preds = []
                    preds = []
                    
                    for i in range(len(stack_models)):
                        stacked_preds.append(stack_models[i](x).squeeze()[:n].unsqueeze(dim=1)) #shape[120,1]=>[120]=>shape[120,1] 
#                     for i in range(len(models)):
#                         preds.append(models[i](stacked_preds[meta_models[i][1]]))#noidea  #shape[120,1]
#                     del x,stacked_preds
#                     print(len(preds))
#                     y_pred = torch.sigmoid(weight_preds(preds, weigths)).mean().item()
                    y_pred=torch.sigmoid(stacked_preds[0]).mean().item()
#                     print("B0",torch.sigmoid(stacked_preds[0]).mean().item())
                    print("Prediction:",y_pred)
                    print("-------------------------------------------------------------")
                    del preds
                    
                    return y_pred

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    
    
    return 0.5
from concurrent.futures import ThreadPoolExecutor
import gc

def predict_on_video_set(videos, num_workers):#noidea
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))
        
    return list(predictions)
predictions = predict_on_video_set(test_videos, num_workers=4)
print(predictions)
submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)