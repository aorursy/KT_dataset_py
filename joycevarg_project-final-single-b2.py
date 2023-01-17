import os,sys
import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np

sys.path.append('../input/efficient2')
sys.path.insert(0, "/kaggle/input/blazeface")
sys.path.insert(0, "/kaggle/input/faceextractor")

from efficientnet import EfficientNet
from blazeface import BlazeFace
from torchvision.transforms import Normalize
from face_extract_1 import FaceExtractor
path = "/kaggle/input/videos/alrtntfxtd.mp4"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
facedet = BlazeFace().to(device)
facedet.load_weights("/kaggle/input/blazeface/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface/anchors.npy")
_ = facedet.train(False)
input_size = 256

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
def disable_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model
def read_frames_at_indices(path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(0, frame_idxs[-1] + 1):
                ret = capture.grab()
                if not ret:
                    break
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    idxs_read.append(frame_idx)
            if len(frames) > 0:
                return np.stack(frames), idxs_read
            return None
        except:
            return None    

def read_frames(path, num_frames):
        assert num_frames > 0
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None
        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        result = read_frames_at_indices(path,capture,frame_idxs)
        capture.release()
        return result

frames_per_video = 120

video_read_fn = lambda x: read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)
model_path = "/kaggle/input/models/"
model_ext = '.pth'

model_name='model'
model_type='efficientnet-b2'

checkpoint=torch.load(model_path+model_name+model_ext,map_location=device)
model = EfficientNet.from_name(model_type)
model._fc = nn.Linear(model._fc.in_features, 1)
model.load_state_dict(checkpoint)
_ = model.eval()
_ = disable_grad(model)
model = model.to(device)
del checkpoint
def predict_on_video(video_path):
    try:
        batch_size=frames_per_video
        faces = face_extractor.process_video(video_path)
        face_extractor.keep_only_best_face(faces)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = cv2.resize(face, (input_size, input_size))
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
            
            del faces
            if n != 0:
                x = torch.tensor(x, device=device).float()
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                with torch.no_grad():
                    
                    y_pred=model(x)
                    y_pred=torch.sigmoid(y_pred).mean().item()
                    print("Prediction:",y_pred)
                    print("-------------------------------------------------------------")
                    return y_pred

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    
    
    return 0.5
prediction =predict_on_video(path)
