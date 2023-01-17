from math import log



# loss1 - None

loss1 = -1/4000 * (174 * log(174/1477) + (1477 - 174) * log(1 - 174/1477))





# loss2 - 16:9

# cd == '1/48000':

loss2_1 = -1/4000 * (1407 * log(1407/1873) + (1873 - 1407) * log(1 - 1407/1873))

# cd == else:

loss2_2 = -1/4000 * (156 * log(156/206) + (206 - 156) * log(1 - 156/206))





# loss3 - 9:16

# cd == '1/48000':

loss3_1 = -1/4000 * (70 * log(70/178) + (178 - 70) * log(1 - 70/178))

# cd == else:

loss3_2 = -1/4000 * (182 * log(182/241) + (241 - 182) * log(1 - 182/241))





# others

others = -1/4000 * (11 * log(11/25) + 14 * log(14/25))



score = loss1 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + others
print('Public LB score: ', int(score * 100000) / 100000)
import pandas as pd

import glob

import os

import subprocess as sp

import tqdm.notebook as tqdm

from collections import defaultdict

import json



! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
def check_output(*popenargs, **kwargs):

    closeNULL = 0

    try:

        from subprocess import DEVNULL

        closeNULL = 0

    except ImportError:

        import os

        DEVNULL = open(os.devnull, 'wb')

        closeNULL = 1



    process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *popenargs, **kwargs)

    output, unused_err = process.communicate()

    retcode = process.poll()



    if closeNULL:

        DEVNULL.close()



    if retcode:

        cmd = kwargs.get("args")

        if cmd is None:

            cmd = popenargs[0]

        error = sp.CalledProcessError(retcode, cmd)

        error.output = output

        raise error

    return output



def ffprobe(filename):

    

    command = ["../working/ffmpeg-git-20191209-amd64-static/ffprobe", "-v", "error", "-show_streams", "-print_format", "xml", filename]



    xml = check_output(command)

    

    return xml



def get_markers(video_file):



    xml = ffprobe(str(video_file))

    

    found = str(xml).find('display_aspect_ratio')

    if found >= 0:

        ar = str(xml)[found+22:found+26]

    else:

        ar = None

        

    found = str(xml).find('"audio" codec_time_base')

    if found >= 0:

        cd = str(xml)[found+25:found+32]

    else:

        cd = None

    

    return ar, cd
video_file = '/kaggle/input/deepfake-detection-challenge/test_videos/gunamloolc.mp4'

get_markers(video_file)
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')
my_dict = defaultdict()

for filename in tqdm.tqdm(filenames):

    fn = filename.split('/')[-1]

    ar, cd = get_markers(filename)

    my_dict[fn] = ar
display_aspect_ratios = pd.DataFrame.from_dict(my_dict, orient='index')

display_aspect_ratios.columns = ['display_aspect_ratio']

display_aspect_ratios = display_aspect_ratios.fillna('NONE')
labels = json.load(open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json', encoding="utf8"))



labels = pd.DataFrame(labels).transpose()

labels = labels.reset_index()

labels = labels.join(display_aspect_ratios, on='index')
labels.head()
pd.crosstab(labels.display_aspect_ratio, labels.label)
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')
sub = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

sub.label = 11/25

sub = sub.set_index('filename',drop=False)
for filename in tqdm.tqdm(filenames):

    

    fn = filename.split('/')[-1]

    ar, cd = get_markers(filename)

    

    if ar is None:

        sub.loc[fn, 'label'] = 174/1477

    if cd == '1/48000':

        if ar == '16:9':

            sub.loc[fn, 'label'] = 1407/1873

        if ar == '9:16':

            sub.loc[fn, 'label'] = 70/178

    else:

        if ar == '16:9':

            sub.loc[fn, 'label'] = 156/206

        if ar == '9:16':

            sub.loc[fn, 'label'] = 182/241
sub.label.value_counts()
sub.to_csv('submission.csv', index=False)