# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# friendly reminder：Meta-information detection has been officially banned

#鸡肋，食之无味，弃之可惜
print(os.listdir('../input'))
print(os.listdir('../input/ddocuments'))
!pip install mfc_video_utils

!yes|apt-get install --no-install-recommends ffmpeg && pip install ffmpeg scikit-video
import sklearn

from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns

from mfc_video_utils import MfcVideoProcessor, BasicTransformer, compute_roc, save_object, load_object, grid_search_forest, grid_search_svm

from tqdm import tqdm
import mfc_video_utils

from mfc_video_utils import MfcVideoProcessor, BasicTransformer, compute_roc, save_object, load_object, grid_search_forest
gs_forest_50 = load_object('../input/ddocuments/my_forest50_icml.pkl')
class MfcVideoProcessor:

    ''' Class to process MFC video datasets easily'''

    frac_list = ['video_@codec_time_base', 'video_@r_frame_rate', 'video_@avg_frame_rate', 

             'video_@time_base', 'audio_@r_frame_rate','audio_@codec_time_base', 

             'audio_@time_base', 'subtitle_@r_frame_rate', 'subtitle_@time_base', 

             'data_@r_frame_rate', 'data_@avg_frame_rate', 'data_@time_base']

    time_list = ['video_tag_@creation_time', 'audio_tag_@creation_time', 'video_tag_@DateTime',

             'video_tag_@DateTimeOriginal', 'video_tag_@DateTimeDigitized', 'data_tag_@creation_time']

    

    def __init__(self, name: str, dataset_abs_path: Path, ref_avail: bool=False, ref_folder: bool=False):

        '''__init__ constructor'''

        if not dataset_abs_path.exists():

            raise Exception("Dataset path does not exist")

        else:

            self.name = name

            self.basepath = Path(dataset_abs_path)

            self.probes = self.basepath / "probe"

            self.ref_avail = ref_avail

            if ref_avail and ref_folder:

                self.reference_basepath = Path(str(dataset_abs_path) + '-Reference')

                if not self.reference_basepath.exists():

                    raise Exception("Path to reference folder does not exist")

                else:

                    self.csv_path = [item for item in (self.reference_basepath / "reference/manipulation-video/").glob("*-manipulation-video-ref.csv")][0]

                    self.csv = pd.read_csv(self.csv_path, sep="|")

                    self.labels = [self.get_video_label(i) for i in range(len(self.csv))]

            elif ref_avail:

                self.csv_path = [item for item in self.basepath.glob("reference/manipulation-video/*-manipulation-video-ref.csv")][0]

                self.csv = pd.read_csv(self.csv_path, sep="|")

                self.labels = [self.get_video_label(i) for i in range(len(self.csv))]

            else:

                pass

#                 self.csv_path = [item for item in self.basepath.glob("indexes/manipulation-video/*-manipulation-video-ref.csv")][0]

#                 self.csv = pd.read_csv(self.csv_path, sep="|")

                

            self.ffprobe_df = self._generate_ffprobe_dataset_dataframe()



    def get_video_path(self, video_id: int) -> str:

        video_name = str(self.csv.at[video_id, 'ProbeFileName'])

        return str(self.basepath / video_name)



    def get_video_label(self, video_id: int) -> int:

        if self.ref_avail:

            return 1 if self.csv.at[video_id, 'IsTarget'] is 'Y' else 0

        else:

            raise Exception("This dataset has no reference available")

            

            

    def _generate_ffprobe_dataset_dataframe(self) -> pd.DataFrame:

        dfs = [] #creates a new dataframe that's empty

        for i in tqdm(os.listdir(self.basepath)):

            if i.endswith("mp4"):

                video_path = f'{self.basepath}/{i}'

                video_metadata = self._flatten(skvideo.io.ffprobe(video_path))

                video_df = pd.DataFrame.from_dict(video_metadata, orient='index', columns=[i])

                dfs.append(video_df)

        vids_df = pd.concat(dfs, axis=1, sort=False)

        vids_df = vids_df.transpose()

        vids_df = vids_df.apply(pd.to_numeric, errors='ignore')

        for col in MfcVideoProcessor.frac_list:

            if col in vids_df.columns:

                vids_df[col] = vids_df[col].apply(self._conv_to_float)

        for col in MfcVideoProcessor.time_list:

            def time_transform(x): 

                if pd.notnull(x) and type(x) is not str:

                    return x.to_datetime64().astype(np.int64) 

                else:

                    return np.nan

            if col in vids_df.columns:

                vids_df[col] = vids_df[col].apply(pd.to_datetime, errors='ignore').apply(time_transform)

        return vids_df



    def _flatten(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:

        items = []

        for k, v in d.items():

            new_key = '{0}{1}{2}'.format(parent_key,sep,k) if parent_key else k

            if isinstance(v, MutableMapping):

                items.extend(self._flatten(v, new_key, sep=sep).items())

            elif isinstance(v, list):

                # apply itself to each element of the list - that's it!

                dic_list = self._flatten_tag(v, parent_key=new_key)

                for k_2, v_2 in dic_list.items():

                    items.append((k_2, v_2))

            else:

                items.append((new_key, v))

        return dict(items)



    def _flatten_tag(self, l: list, parent_key: str = '', sep: str = '_') -> dict:

        items = []

        for ele in l:

            new_key = '{0}{1}{2}{3}'.format(parent_key,sep,'@',ele['@key']) 

            items.append((new_key, ele['@value']))

        return dict(items)



    def _conv_to_float(self, frac: str) -> float:

        if pd.notnull(frac):

            try: return float(fractions.Fraction(frac))

            except ZeroDivisionError: return 0

        else:

            return frac

import skvideo

from collections.abc import MutableMapping

import fractions
test_datasets_location = Path("//kaggle/input/deepfake-detection-challenge/test_videos/")

test = MfcVideoProcessor("dfdc_vid", test_datasets_location, ref_avail=False)
pred = gs_forest_50.predict(test.ffprobe_df)
pred
sub=pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

filenames = os.listdir(test_dir)
for pred,name in zip(pred,filenames):

    name=name.replace('/kaggle/input/deepfake-detection-challenge/test_videos/','')

    sub.iloc[list(sub['filename']).index(name),1]=pred
sub.head(20)
sub.to_csv('submission.csv',index=False)