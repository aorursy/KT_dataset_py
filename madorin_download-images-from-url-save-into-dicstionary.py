import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_data = pd.read_csv('../input/index.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")
# install train picture
from PIL import Image
import io
from urllib.request import urlopen

train_pic_dict = dict()
print(len(train_data['url']))
cnt = 0
for url , id in zip(train_data['url'], train_data['id']):
    print(cnt)
    try:
        file =io.BytesIO(urlopen(url).read())
        img = Image.open(file)
    except:
        continue
    #img - > array
    im_list = np.asarray(img)
    #paste
    train_pic_dict[id] = url
