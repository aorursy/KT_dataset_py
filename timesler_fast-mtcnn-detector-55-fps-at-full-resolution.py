%%capture

# Install facenet-pytorch (with internet use "pip install facenet-pytorch")

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl

!pip install /kaggle/input/imutils/imutils-0.5.3
from facenet_pytorch import MTCNN

from PIL import Image

import torch

from imutils.video import FileVideoStream

import cv2

import time

import glob

from tqdm.notebook import tqdm



device = 'cuda' if torch.cuda.is_available() else 'cpu'



filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')[:100]
class FastMTCNN(object):

    """Fast MTCNN implementation."""

    

    def __init__(self, stride, resize=1, *args, **kwargs):

        """Constructor for FastMTCNN class.

        

        Arguments:

            stride (int): The detection stride. Faces will be detected every `stride` frames

                and remembered for `stride-1` frames.

        

        Keyword arguments:

            resize (float): Fractional frame scaling. [default: {1}]

            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).

            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).

        """

        self.stride = stride

        self.resize = resize

        self.mtcnn = MTCNN(*args, **kwargs)

        

    def __call__(self, frames):

        """Detect faces in frames using strided MTCNN."""

        if self.resize != 1:

            frames = [

                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))

                    for f in frames

            ]

                      

        boxes, probs = self.mtcnn.detect(frames[::self.stride])



        faces = []

        for i, frame in enumerate(frames):

            box_ind = int(i / self.stride)

            if boxes[box_ind] is None:

                continue

            for box in boxes[box_ind]:

                box = [int(b) for b in box]

                faces.append(frame[box[1]:box[3], box[0]:box[2]])

        

        return faces
fast_mtcnn = FastMTCNN(

    stride=4,

    resize=1,

    margin=14,

    factor=0.6,

    keep_all=True,

    device=device

)
def run_detection(fast_mtcnn, filenames):

    frames = []

    frames_processed = 0

    faces_detected = 0

    batch_size = 60

    start = time.time()



    for filename in tqdm(filenames):



        v_cap = FileVideoStream(filename).start()

        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))



        for j in range(v_len):



            frame = v_cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)



            if len(frames) >= batch_size or j == v_len - 1:



                faces = fast_mtcnn(frames)



                frames_processed += len(frames)

                faces_detected += len(faces)

                frames = []



                print(

                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',

                    f'faces detected: {faces_detected}\r',

                    end=''

                )



        v_cap.stop()



run_detection(fast_mtcnn, filenames)
fast_mtcnn = FastMTCNN(

    stride=4,

    resize=0.5,

    margin=14,

    factor=0.5,

    keep_all=True,

    device=device

)
run_detection(fast_mtcnn, filenames)