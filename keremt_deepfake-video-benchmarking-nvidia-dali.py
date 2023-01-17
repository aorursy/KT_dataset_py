!pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

from fastai.vision import *

import cv2 as cv
train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T.reset_index()

train_sample_metadata.columns = ['fname','label','split','original']

train_sample_metadata.head()
fake_sample_df = train_sample_metadata[train_sample_metadata.label == 'FAKE']

real_sample_df = train_sample_metadata[train_sample_metadata.label == 'REAL']
train_dir = Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')

test_dir = Path('/kaggle/input/deepfake-detection-challenge/test_videos/')

train_video_files = get_files(train_dir, extensions=['.mp4'])

test_video_files = get_files(test_dir, extensions=['.mp4'])
len(train_video_files), len(test_video_files)
from nvidia.dali.pipeline import Pipeline

from nvidia.dali import ops

fname = train_video_files[0]
batch_size=1

sequence_length=30

initial_prefetch_size=16



class VideoPipe(Pipeline):

    "video pipeline for a single video with 30 frames"

    def __init__(self, batch_size, num_threads, device_id, data, shuffle):

        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)

        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,

                                     shard_id=0, num_shards=1,

                                     random_shuffle=shuffle, initial_fill=initial_prefetch_size)

    def define_graph(self):

        output = self.input(name="Reader")

        return output
def dali_batch(fname):

    pipe = VideoPipe(batch_size=batch_size, num_threads=defaults.cpus, device_id=0, data=[fname], shuffle=False)

    pipe.build()

    pipe_out = pipe.run()

    sequences_out = pipe_out[0].as_cpu().as_array()

    data = torch.from_numpy(sequences_out[0])

    data = data.permute(0,3,1,2).cuda()

    return F.interpolate(data.to(torch.float32), (640,640))
%%time

data = dali_batch(train_video_files[0])
img0 = Image(data[0]/255)

img0.show(figsize=(10,10))
def frame_img_generator(path, freq=None):

    "frame generator for a given video file"

    vidcap = cv.VideoCapture(str(path))

    n_frames = 0

    while True:

        success = vidcap.grab()

        if not success: 

            vidcap.release()

            break   

            

        if (freq is None) or (n_frames % freq == 0):

            _, frame = vidcap.retrieve()

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

#             frame = cv.resize(frame, (640,640))

            yield frame    

        

        n_frames += 1

        

    vidcap.release()
%%time

# CPU warm up

frames = list(frame_img_generator(train_video_files[0], 10)); len(frames)
%%timeit

frames = list(frame_img_generator(train_video_files[0], freq=10)); len(frames)
%%time

frames = frame_img_generator(train_video_files[0], 10)

data = torch.from_numpy(array(frames))

data = data.permute(0,3,1,2).cuda()

data = F.interpolate(data.to(torch.float32), (640,640))
img1 = Image(data[0]/255)

img1.show(figsize=(10,10))
del frames

del data; gc.collect()
!pip install -q /kaggle/input/imutils/imutils-0.5.3
from imutils.video import FileVideoStream
def fvs_img_generator(path, freq=None):

    "frame generator for a given video file"

    fvs = FileVideoStream(str(path)).start()

    n_frames = 0

    while fvs.more():

        frame = fvs.read()

        if frame is None: break # https://github.com/jrosebr1/imutils/pull/119

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        

        if (freq is None) or (n_frames % freq == 0):

            yield frame

        

        n_frames += 1

    fvs.stop()
%%timeit

frames = list(fvs_img_generator(str(train_video_files[0]), 10))
%%time

frames = list(fvs_img_generator(str(train_video_files[0]), 10))

data = torch.from_numpy(array(frames))

data = data.permute(0,3,1,2).cuda()

data = F.interpolate(data.to(torch.float32), (640,640))
img2 = Image(data[0]/255)

img2.show(figsize=(10,10))
del frames

del data; gc.collect()
assert torch.all(img1.data == img2.data)
!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh 
sys.path.insert(0,'/kaggle/working/reader/python')



from decord import VideoReader

from decord import cpu, gpu

from decord.bridge import set_bridge

set_bridge('torch')
# GPU warm up

video = VideoReader(str(train_video_files[0]), ctx=gpu())

del video; gc.collect()
%%time

video = VideoReader(str(train_video_files[0]), ctx=gpu())

data = video.get_batch(range(0, len(video), 10))

data = F.interpolate(data.to(torch.float32), (640,640))
img3 = Image(data[0]/255)

img3.show(figsize=(10,10))
del video

del data; gc.collect()
torch.mean(torch.isclose(img1.data, img3.data, atol=0.01).float())
%%time

video = VideoReader(str(train_video_files[0]), ctx=cpu())

data = video.get_batch(range(0, len(video), 10)).cuda()

data = F.interpolate(data.to(torch.float32), (640,640))
img4 = Image(data[0]/255)

img4.show(figsize=(10,10))
del video

del data; gc.collect()
assert torch.all(img1.data == img4.data)
torch.mean(torch.isclose(img0.data, img1.data, atol=0.01).float())