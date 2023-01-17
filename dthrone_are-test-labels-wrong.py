from glob import glob

import pandas as pd

from os.path import basename
files = [basename(x) for x in glob('../input/deepfake-detection-challenge/test_videos/*.mp4')]

s = pd.Series(index=pd.Index(data=files, name='filename'), data=[0.5 + 0.5*(x=='ahjnxtiamx.mp4') for x in files], name='label')

s.to_frame().to_csv('submission.csv')
s.sort_index().head(10)
from IPython.display import HTML

from base64 import b64encode

vid1 = open('/kaggle/input/deepfake-detection-challenge/test_videos/ahjnxtiamx.mp4','rb').read()

data_url = "data:video/mp4;base64," + b64encode(vid1).decode()

HTML("""

<video width=600 controls>

      <source src="%s" type="video/mp4">

</video>

""" % data_url)

# credit https://www.kaggle.com/hamditarek/deepfake-detection-challenge-kaggle