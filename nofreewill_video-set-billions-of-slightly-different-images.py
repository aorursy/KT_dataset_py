from fastai.core import Path, progress_bar

from fastai.vision import imagenet_stats, Image

import cv2

import torch

from torchvision import transforms

import PIL

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class capsSet():

    def __init__(self, videos_path, bs=64):

        

        self.videos_path = videos_path

        self.video_paths = videos_path.ls()

        self.videos_num = len(self.video_paths)

        

        self.bs = bs

        self.caps = [cv2.VideoCapture() for i in range(bs)]

        

        self.video_idx = 0

        self.steps_taken = 0

        

        self.tfms = transforms.Compose([transforms.Resize((144,256), interpolation=2), transforms.ToTensor(), transforms.Normalize(*imagenet_stats)])

        

        self.frames_count = None

        

    def get_batch(self):

        frames_batch = [self.get_frame(i) for i in range(self.bs)]

        self.steps_taken += 1

        return torch.stack(frames_batch)

        

    def get_frame(self, cap_id):

        cap = self.caps[cap_id]

        res, frame_bgr = cap.read()

        if not res:

            cap.open(str(self.video_paths[self.video_idx]))

            self.video_idx = (self.video_idx+1) if self.video_idx<(self.videos_num-1) else (0)

            frame_tensor = self.get_frame(cap_id)

        else:

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frame_pil = PIL.Image.fromarray(frame_rgb)

            frame_tensor = self.tfms(frame_pil)

        return frame_tensor

    

    def get_frames_count(self, silent=False):

        if self.frames_count is None:

            n_frames = 0

            cap = cv2.VideoCapture()

            for video_path in progress_bar(self.video_paths, display=(not silent)):

                cap.open(str(video_path))

                n_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.frames_count = n_frames

        return self.frames_count
videos_path = Path('')#/media/nofreewill/Datasets_nvme/Visual/Videos/BestSet/Videos/Raw') # I have all the videos in here

capsset = capsSet(videos_path, bs=64)



print(capsset.videos_num)
# for i in range(1):

#     images = capsset.get_batch()

# print(images.shape)
# img = images[0]

# Image((img - img.min())/(img.max()-img.min()+1e-6))
# capsset.get_frames_count()