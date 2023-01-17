%matplotlib inline
import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
VIDEO_DIR = os.path.join('..', 'input')
!ls ../input
all_srt_files = glob(os.path.join(VIDEO_DIR, '*.SRT'))
all_videos = [x.replace('SRT', 'MP4') for x in all_srt_files]
all_videos = [x.replace('MP4', 'MOV') if not os.path.exists(x) else x for x in all_videos]
print('Found', len(all_videos), 'videos and', len(all_srt_files), 'flight files')
from collections import defaultdict
def read_srt(in_path):
    col_names = ['id', 'Time Code', 'Home', 'Drone', 'Camera', '']
    with open(in_path, 'r') as f:
        all_lines = defaultdict(dict)
        for i, k in enumerate(f.readlines()):
            all_lines[i//6][col_names[i % 6]] = k.strip()
        cur_df = pd.DataFrame([v for k, v in all_lines.items()])
        cur_df['DLatLongHeight'] = cur_df['Drone'].map(lambda x: [float(y) for y in x.split(' ')[0].replace('GPS(', '').replace(')', '').split(',')])
        cur_df['DLat'] = cur_df['DLatLongHeight'].map(lambda x: x[0])
        cur_df['DLong'] = cur_df['DLatLongHeight'].map(lambda x: x[1])
        return cur_df
test_df = read_srt(all_srt_files[-1])
test_df.sample(3)
plt.plot(test_df['DLat'], test_df['DLong'])
def read_video_segment(in_path, vid_seg = None):
    cap = cv2.VideoCapture(in_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if vid_seg is None:
            vid_seg = np.array([0, 0.25, 0.5, 0.75, 1])
        else:
            vid_seg = np.clip(vid_seg, 0, 1)
            
        frame_ids = np.clip(video_length*vid_seg, 0, video_length-1).astype(int)
        count = 0
        success, image = cap.read()
        print('Loaded', video_length, 'frames at', image.shape, 'resolution')
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()
            count += 1
    return frames
fig, m_axs = plt.subplots(len(all_videos), 5, figsize = (20, len(all_videos)*4))
for c_path, c_srt_path, c_axs in zip(all_videos, all_srt_files, m_axs):
    cur_df = read_srt(c_srt_path)
    c_axs[0].plot(cur_df['DLat'], cur_df['DLong'])
    for c_frame, c_ax in zip(read_video_segment(c_path), 
                             c_axs[1:]):
        c_ax.imshow(c_frame[:, :, ::-1])
        c_ax.set_title(os.path.basename(c_path))
        c_ax.axis('off')
fig.savefig('high_res_frames.png', figdpi = 300)
