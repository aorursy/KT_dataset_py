import os

import cv2

import numpy as np
train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"

video_path = os.path.join(train_dir, np.random.choice(os.listdir(train_dir)))

video_path
def grab_frames_from_video(path, num_frames=10):

    capture = cv2.VideoCapture(path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count, num_frames, endpoint=False, dtype=np.int)



    i = 0

    for frame_idx in range(int(frame_count)):

        # Get the next frame, but don't decode if we're not using it.

        ret = capture.grab()

        if not ret: 

            print("Error grabbing frame %d from movie %s" % (frame_idx, path))



        # Need to look at this frame?

        if frame_idx >= frame_idxs[i]:

            ret, frame = capture.retrieve()

            if not ret or frame is None:

                print("Error retrieving frame %d from movie %s" % (frame_idx, path))

            else:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Do something with `frame`



            i += 1

            if i >= len(frame_idxs):

                break



    capture.release()
%time grab_frames_from_video(video_path, num_frames=10)
%time grab_frames_from_video(video_path, num_frames=50)
# This version.



def grab_frames_from_video(path, num_frames=10):

    capture = cv2.VideoCapture(path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count, num_frames, endpoint=False, dtype=np.int)



    for i, frame_idx in enumerate(frame_idxs):

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = capture.read()

        if not ret or frame is None:

            print("Error retrieving frame %d from movie %s" % (frame_idx, path))

        else:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



    capture.release()
%time grab_frames_from_video(video_path, num_frames=10)
%time grab_frames_from_video(video_path, num_frames=50)