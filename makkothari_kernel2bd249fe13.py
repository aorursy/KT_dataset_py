# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir("../input/plant-pathology-2020-fgvc7")
!git clone https://github.com/Tessellate-Imaging/monk_v1.git

!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
import sys



import pandas as pd

df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")



combined = [];

from tqdm.notebook import tqdm

for i in tqdm(range(len(df))):

    img_name = df["image_id"][i] + ".jpg";

    if(df["healthy"][i]):

        label = "healthy";

    elif(df["multiple_diseases"][i]):

        label = "multiple_diseases";

    elif(df["rust"][i]):

        label = "rust";

    else:

        label = "scab";

    

    combined.append([img_name, label]);

    

df2 = pd.DataFrame(combined, columns = ['ID', 'Label']) 

df2.to_csv("train.csv", index=False);
import os

import sys

sys.path.append("monk_v1/monk/");
from pytorch_prototype import prototype



ptf = prototype(verbose=1);

ptf.Prototype("Project-Plant-Disease", "Pytorch_Resnet152");
ptf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",

            path_to_csv="train.csv", 

            model_name="resnet152", 

            num_epochs=20);
ptf.update_save_intermediate_models(False);

ptf.Reload();
#ptf.apply_center_crop(224,train=True, val=False,test=False)

#ptf.Reload()

ptf.Summary()
ptf.update_use_gpu(True)

ptf.Reload()
ptf.Train()
ptf.Prototype("Project-Plant-Disease", "Pytorch_Resnet152",eval_infer=True)
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from scipy.special import softmax

df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
for i in tqdm(range(len(df))):

    img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/" + df["image_id"][i] + ".jpg";

    

    #Invoking Monk's nferencing engine inside a loop

    predictions = gtf.Infer(img_name=img_name, return_raw=True);

    out = predictions["raw"]

    

    df["healthy"][i] = out[0];

    df["multiple_diseases"][i] = out[1];

    df["rust"][i] = out[2];

    df["scab"][i] = out[3];
df.to_csv("submission.csv", index=False);
