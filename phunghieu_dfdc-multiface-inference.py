# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl



from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

torch_home = get_torch_home()



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18

from facenet_pytorch import MTCNN

from albumentations import Normalize, Compose

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from PIL import Image

from tqdm.notebook import tqdm

import os

import glob

import multiprocessing as mp



if torch.cuda.is_available():

    device = 'cuda:0'

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

else:

    device = 'cpu'

print(f'Running on device: {device}')
TEST_DIR = '/kaggle/input/deepfake-detection-challenge/test_videos/'

MODEL_PATH = '/kaggle/input/dfdcmultifacef5-resnet18/f5_resnet18.pth'



N_FACES = 5

BATCH_SIZE = 64

NUM_WORKERS = mp.cpu_count()



FRAME_SCALE = 0.25

FACE_BATCH_SHAPE = (N_FACES*3, 160, 160)



DEFAULT_PROB = 0.5
class DeepfakeClassifier(nn.Module):

    def __init__(self, encoder, in_channels=3, num_classes=1):

        super(DeepfakeClassifier, self).__init__()

        self.encoder = encoder

        

        # Modify input layer.

        self.encoder.conv1 = nn.Conv2d(

            in_channels,

            64,

            kernel_size=7,

            stride=2,

            padding=3,

            bias=False

        )

        

        # Modify output layer.

        self.encoder.fc = nn.Linear(512 * 1, num_classes)



    def forward(self, x):

        return torch.sigmoid(self.encoder(x))

    

    def freeze_all_layers(self):

        for param in self.encoder.parameters():

            param.requires_grad = False



    def freeze_middle_layers(self):

        self.freeze_all_layers()

        

        for param in self.encoder.conv1.parameters():

            param.requires_grad = True

            

        for param in self.encoder.fc.parameters():

            param.requires_grad = True



    def unfreeze_all_layers(self):

        for param in self.encoder.parameters():

            param.requires_grad = True

            

            

class TestVideoDataset(Dataset):

    def __init__(self, test_dir, frame_resize=None, face_detector=None, n_faces=1, preprocess=None):

        self.test_dir = test_dir

        self.test_video_paths = glob.glob(os.path.join(self.test_dir, '*.mp4'))

        self.face_detector = face_detector

        self.n_faces = n_faces

        self.frame_resize = frame_resize

        self.preprocess = preprocess



    def __len__(self):

        return len(self.test_video_paths)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        test_video_path = self.test_video_paths[idx]

        test_video = test_video_path.split('/')[-1]

        

        # Get faces until enough (try limit: n_faces)

        faces = []

        

        for i in range(self.n_faces):

            # Create video reader and find length

            v_cap = cv2.VideoCapture(test_video_path)

            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            

            stride = int(v_len/(self.n_faces**2))

            sample = np.linspace(i*stride, (v_len - 1) + i*stride, self.n_faces).astype(int)

            frames = []



            # Get frames

            for j in range(v_len):

                success = v_cap.grab()

                

                if j in sample:

                    success, frame = v_cap.retrieve()



                    if not success:

                        continue



                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame = Image.fromarray(frame)



                    # Resize frame to desired size

                    if self.frame_resize is not None:

                        frame = frame.resize([int(d * self.frame_resize) for d in frame.size])

                    frames.append(frame)

            

            if len(frames) > 0:

                all_faces_in_frames = [

                    detected_face

                    for detected_faces in self.face_detector(frames)

                    if detected_faces is not None

                    for detected_face in detected_faces

                ]



                faces.extend(all_faces_in_frames)

            

            if len(faces) >= self.n_faces: # Get enough faces

                break



        v_cap.release()



        if len(faces) >= self.n_faces: # Get enough faces

            faces = faces[:self.n_faces] # Get top

            

            if self.preprocess is not None:

                for j in range(len(faces)):

                    augmented = self.preprocess(image=faces[j].cpu().detach().numpy().transpose(1, 2, 0))

                    faces[j] = augmented['image']

            

            faces = np.concatenate(faces, axis=-1).transpose(2, 0, 1)

            

            return {

                'video_name': test_video,

                'faces': faces,

                'is_valid': True

            }

        else:

            return {

                'video_name': test_video,

                'faces': np.zeros(FACE_BATCH_SHAPE, dtype=np.float32),

                'is_valid': False # Those invalid videos will get DEFAULT_PROB

            }
# Load face detector.

face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, post_process=False, device=device).eval()
encoder = resnet18(pretrained=False)



classifier = DeepfakeClassifier(encoder=encoder, in_channels=3*N_FACES, num_classes=1)

classifier.to(device)

state = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

classifier.load_state_dict(state['state_dict'])

classifier.eval()
preprocess = Compose([

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)

])
test_dataset = TestVideoDataset(

    TEST_DIR,

    frame_resize=FRAME_SCALE,

    face_detector=face_detector,

    n_faces=N_FACES,

    preprocess=preprocess

)



test_dataloader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS

)
submission = []



with torch.no_grad():

    try:

        for videos in tqdm(test_dataloader):

            y_pred = classifier(videos['faces']).squeeze(dim=-1).cpu().detach().numpy()

            submission.extend(list(zip(videos['video_name'], y_pred, videos['is_valid'].cpu().detach().numpy())))

    except Exception as e:

        print(e)

        

submission = pd.DataFrame(submission, columns=['filename', 'label', 'is_valid'])

submission.sort_values('filename', inplace=True)

submission.loc[submission.is_valid == False, 'label'] = DEFAULT_PROB
submission[['filename', 'label']].to_csv('submission.csv', index=False)



plt.hist(submission.label, 20)

plt.show()