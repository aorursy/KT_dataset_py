import sys, os, glob, gc

import timeit

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import cv2

print('Python         : ' + sys.version.split('\n')[0])

print('Numpy          : ' + np.__version__)

print('OpenCV         : ' + cv2.__version__)

pd.set_option('display.max_colwidth', 100)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 4000)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()

seed = 0



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%capture

!pip install /kaggle/input/nvidia-dali-019-cuda-10/nvidia_dali-0.19.0-1119077-cp36-cp36m-manylinux1_x86_64.whl

!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh
sys.path.insert(0,'/kaggle/working/reader/python')

import decord

from nvidia.dali.pipeline import Pipeline

from nvidia.dali import ops
class CV2VideoReader(object):

    

    def __init__(self, path, num_frames, stride=None, verbose=False):

        self.path = path

        self.num_frames = num_frames

        self.capture = None

        self.frame_count = None

        self.frame_counter = -1

        self.idx = 0

        self.stride = stride

        self.verbose = verbose



    def initialize(self):

        ret = False

        try:

            self.capture = cv2.VideoCapture(self.path)

            if self.capture.isOpened():

                self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

                if self.frame_count < self.num_frames:

                    self.num_frames = self.frame_count            

                if self.stride is None:

                    # Frames at regular interval

                    self.frame_idxs = np.linspace(0, self.frame_count, self.num_frames, endpoint=False, dtype=np.int)

                else:

                    # Frames with stride interval

                    self.frame_idxs = [i*self.stride for i in range(self.num_frames) if i*self.stride < self.frame_count]

                ret = True

        except Exception as ex:

            print("Init Error:", ex)              

        return ret



    def __len__(self):

        return len(self.frame_idxs)



    def __iter__(self):

        return self



    def __next__(self):

        decoded_frame = None

        decoded_frame_idx = None

        ret = None

        # Grab next frame until the selected one

        while ((self.frame_count is not None) and (self.idx < len(self.frame_idxs)) and (self.frame_counter < self.frame_count) and (self.frame_counter < self.frame_idxs[self.idx])):

            self.frame_counter = self.frame_counter + 1

            if self.verbose: print("grab", self.frame_counter)

            ret = self.capture.grab()

            if not ret:

                print("Error grabbing frame %d from %s" % (self.frame_counter, self.path))               

        # Retrieve the frame if possible

        if ret:

            if self.verbose: print("retrieve", self.frame_counter, self.frame_idxs[self.idx])

            ret, frame_tmp = self.capture.retrieve()

            if ret and frame_tmp is not None:

                decoded_frame = cv2.cvtColor(frame_tmp, cv2.COLOR_BGR2RGB)

                decoded_frame_idx = self.frame_counter

            else:

                print("Error retrieving frame %d from %s" % (self.frame_counter, self.path)) 

        # End of stream?

        if (self.frame_counter >= self.frame_count) | (self.idx >= len(self.frame_idxs)):

            self.release()

            raise StopIteration

        self.idx = self.idx + 1

        return (decoded_frame, decoded_frame_idx)



    def release(self):

        if self.capture is not None:

            self.capture.release()

            self.capture = None
# https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/supported_ops.html#nvidia.dali.ops.VideoReader.__call__

class VideoPipe(Pipeline):



    def __init__(self, batch_size, num_threads, device_id, data, shuffle, num_frames, stride):

        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)

        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=num_frames, stride=stride,

                                     shard_id=0, num_shards=1,

                                     random_shuffle=shuffle, initial_fill=16)

    def define_graph(self):

        output = self.input(name="Reader")

        return output
class DALIVideoReader(object):

    

    def __init__(self, path, num_frames, cpus=1, shuffle=False, stride=None, verbose=False):

        self.path = path

        self.num_frames = num_frames

        self.frame_count = None

        self.frame_counter = 0

        self.local_idx = 0

        self.idx = 0

        self.cpus = cpus

        self.shuffle = shuffle

        self.stride = stride

        self.verbose = verbose



    def initialize(self):

        ret = False

        try:

            # How to get total frames with DALI? Is is required to avoid crash when asking frames beyond the end.                

            tmp = cv2.VideoCapture(self.path)

            if tmp.isOpened():

                self.frame_count = int(tmp.get(cv2.CAP_PROP_FRAME_COUNT))

                if self.verbose == True: print("Frames count:", self.frame_count)

                tmp.release()

            if self.frame_count is not None:

                # Limit frames to what is available

                if self.frame_count < self.num_frames:

                    self.num_frames = self.frame_count          

                if self.stride is None:

                    # Frames at regular interval

                    self.frame_idxs = np.linspace(0, self.frame_count, self.num_frames, endpoint=False, dtype=np.int)

                    self.skip = self.frame_idxs[1] - self.frame_idxs[0] - 1

                else:

                    # Frames with stride interval

                    self.frame_idxs = [i*self.stride for i in range(self.num_frames) if i*self.stride < self.frame_count]

                    self.skip = self.stride

                if self.verbose == True: print("Stride:", self.skip)

                self.capture = VideoPipe(batch_size=1, num_threads=self.cpus, device_id=0, data=[self.path], 

                                        shuffle=False, num_frames=len(self.frame_idxs), stride=self.skip)

                self.capture.build()

                if self.capture is not None:

                    pipe_out = self.capture.run()

                    self.batch = pipe_out[0].as_cpu().as_array()[0]

                    if self.verbose == True: print("Sequence:", len(self.batch))

                    ret = True

        except Exception as ex:

            print("Init Error:", ex)                    

        return ret

                

    def __len__(self):

        return len(self.batch)



    def __iter__(self):

        return self



    # return RGB image, frame index tuple

    def __next__(self):

        if self.local_idx >= len(self.batch):

            self.release()

            raise StopIteration 

        decoded_frame = self.batch[self.local_idx]

        decoded_frame_idx = self.frame_idxs[self.frame_counter]

        self.local_idx = self.local_idx + 1

        self.frame_counter = self.frame_counter + 1

        return (decoded_frame, decoded_frame_idx)



    def release(self):

        if self.capture is not None:

            del self.capture

            self.capture = None

            gc.collect()
class DecordVideoReader(object):

    

    def __init__(self, path, num_frames, ctx=decord.cpu(), stride=None, verbose=False):

        self.path = path

        self.num_frames = num_frames

        self.frame_count = None

        self.frame_counter = 0

        self.batch_idx = 0

        self.stride = stride

        self.local_idx = 0

        self.idx = 0

        self.ctx = ctx

        self.verbose = verbose



    def initialize(self):

        ret = False

        try:

            self.capture = decord.VideoReader(self.path, ctx=self.ctx)

            if self.capture is not None:

                self.frame_count = len(self.capture)

                self.shape = self.capture[0].shape

                if self.frame_count < self.num_frames:

                    self.num_frames = self.frame_count

                if self.stride is None:

                    # Frames at regular interval

                    self.frame_idxs = np.linspace(0, self.frame_count, self.num_frames, endpoint=False, dtype=np.int)

                else:

                    # Frames with stride interval

                    self.frame_idxs = [i*self.stride for i in range(self.num_frames) if i*self.stride < self.frame_count]

                self.frames_batch = np.array_split(self.frame_idxs, int(np.ceil(len(self.frame_idxs)/8.0)))

                if self.verbose: print("Frames:", self.num_frames)

                if self.verbose: print("Batches:", len(self.frames_batch))

                self.batch = self.load_next_batch()

                ret = True

        except Exception as ex:

            print("Init Error:", ex)

        return ret



    def load_next_batch(self):

        if self.verbose: print("Load batch:", self.batch_idx)

        batch_content = self.capture.get_batch(self.frames_batch[self.batch_idx]).asnumpy()

        self.batch_idx = self.batch_idx + 1

        return batch_content.copy()

                

    def __len__(self):

        return len(self.frame_idxs)



    def __iter__(self):

        return self



    # return RGB image, frame index tuple

    def __next__(self):

        # Next batch?

        if self.local_idx >= len(self.batch):

            # Next batch available?

            if self.batch_idx < len(self.frames_batch):

                self.batch = self.load_next_batch()

                self.local_idx = 0

            else:

                self.release()

                raise StopIteration 

        decoded_frame = self.batch[self.local_idx]

        decoded_frame_idx = self.frame_idxs[self.frame_counter]

        self.local_idx = self.local_idx + 1

        self.frame_counter = self.frame_counter + 1

        return (decoded_frame, decoded_frame_idx)



    def release(self):

        if self.capture is not None:

            del self.capture

            self.capture = None

            gc.collect()
def get_video_reader(name, video, frames, stride, cpus=1):

    if name == "CV2-CPU":

        return CV2VideoReader(video, frames, stride=stride)

    elif name == "DALI-GPU":

        return DALIVideoReader(video, frames, cpus=cpus, stride=stride)    

    elif name == "Decord-CPU":

        return DecordVideoReader(video, frames, ctx=decord.cpu(), stride=stride)

    elif name == "Decord-GPU":

        return DecordVideoReader(video, frames, ctx=decord.gpu(), stride=stride)  
TEST_HOME = "/kaggle/input/deepfake-detection-challenge/test_videos/"

filenames = glob.glob(TEST_HOME + "*.mp4")

basenames = [(os.path.basename(filename), filename) for filename in filenames]

submission_pd = pd.DataFrame(basenames, columns=["filename", "path"])

submission_pd = submission_pd.sort_values('filename')
# Plot some results

readers = ["CV2-CPU", "Decord-CPU","Decord-GPU", "DALI-GPU"]



video = submission_pd[2:3]["path"].values[0]

print(video)

frames = 3



for stride in [None, 3]:

    img_idx = 0

    columns, rows=(3, 1)

    for reader_name in readers:

        print("\nReader:", reader_name)

        reader_ = get_video_reader(reader_name, video, frames, stride=stride)

        if reader_.initialize() == True:

            try:

                loaded_frames = []

                for frame, idx in reader_:

                    loaded_frames.append(idx)

                    col = img_idx % columns

                    if col == 0: fig = plt.figure(figsize=(26, rows*5))

                    ax = fig.add_subplot(rows, columns, col + 1)

                    ax.imshow(frame)

                    ax.axis("off")

                    ax.set_title("%s [%s, %s] Frame#%s %s - Avg: %.1f" % (reader_name, len(reader_), stride, idx, frame.shape, np.mean(frame)))

                    if (col == columns -1): plt.show()

                    img_idx = img_idx + 1

                print("query frames:  ", list(reader_.frame_idxs))

                print("loaded frames: ", loaded_frames)

            except Exception as ex:

                print("Cannot load:", ex)

        reader_.release()

        plt.show()
videos = submission_pd[0:10]["path"].values

videos
# readers = ["CV2-CPU", "Decord-CPU","Decord-GPU", "DALI-GPU"]

readers = ["DALI-GPU", "CV2-CPU", "Decord-CPU"]

STRIDES = [None, 1, 3, 5]



# Decord-GPU random crashes, DALI crashes after 100.

FRAMES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150, 170, 190, 210, 240, 270, 300]
results = []

for reader_name in readers:

    print("Reader:", reader_name, "videos:", len(videos))

    for frames in FRAMES if "DALI" not in reader_name else [i for i in FRAMES if i <= 100]: # DALI crashes after 100

        for stride in STRIDES:

            start_time = timeit.default_timer()

            loaded_frames = []

            for video in videos:

                reader_ = get_video_reader(reader_name, video, frames, stride=stride)

                if reader_.initialize() == True:

                    for frame, idx in reader_:

                        loaded_frames.append(idx)

                reader_.release()

            elapsed = timeit.default_timer() - start_time

            duration_per_video = elapsed / len(videos)

            fps = (len(loaded_frames) / elapsed)

            results.append((reader_name, frames, stride, duration_per_video, fps))

            # print("[%s, %s, %s, %s] Elapsed %.4f sec. Average per video: %.4f sec." % (reader_name, frames, stride, len(loaded_frames), elapsed, duration_per_video), "FPS: %.4f" % (fps))
results_pd = pd.DataFrame(results, columns=["reader", "frames", "stride", "duration_per_video", "fps"])

results_pd.head()
for reader_name in readers:

    f, ax = plt.subplots(1, 2, figsize=(20, 5))

    for stride in STRIDES:

        d = results_pd[(results_pd["reader"] == reader_name) & (results_pd["stride"].isin([stride]))].plot(kind="line", x="frames", y="duration_per_video", grid=True, ax=ax[0], label="Stride=%s" % stride, title="%s - duration_per_video" % reader_name)

        d = results_pd[(results_pd["reader"] == reader_name) & (results_pd["stride"].isin([stride]))].plot(kind="line", x="frames", y="fps", grid=True, ax=ax[1], label="Stride=%s" % stride, title="%s - fps" % reader_name)

    plt.plot()
!rm -r reader && rm install.sh
!cp /kaggle/input/deepfake-detection-challenge/sample_submission.csv ./submission.csv