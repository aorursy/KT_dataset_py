# !pip install -r ../input/setup-r/requirements.txt

# !pip install numpy







# !pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

# !pip install iglovikov-helper-functions

# !pip install pytorch-lightning

# !pip install albumentations

# !pip install pytorch-toolbelt
# from pathlib import Path 

# import shutil





# stored = '/root/.cache/pip/wheels'

# wheels = list(Path(stored).rglob('*.whl'))





# p = Path('./wheels')

# p.mkdir(parents=True, exist_ok=True)





# for whl in wheels: 

#     shutil.copy(whl, p)
# !pip uninstall pillow --yes
# pip install pillow 
# !cat /etc/os-release
!pip install ../input/offwheels/addict-2.2.1-py3-none-any.whl

!pip install ../input/offwheels/mmcv-1.0.4-cp37-cp37m-linux_x86_64.whl

!pip install ../input/offwheels/jpeg4py-0.1.4-py3-none-any.whl

!pip install ../input/offwheels/jsonpointer-2.0-py2.py3-none-any.whl

!pip install ../input/offwheels/jsonpatch-1.26-py2.py3-none-any.whl

!pip install ../input/offwheels/torchfile-0.1.0-py3-none-any.whl 

!pip install ../input/offwheels/visdom-0.1.8.9-py3-none-any.whl

!pip install ../input/offwheels/torchnet-0.0.4-py3-none-any.whl

!pip install ../input/offwheels/Pillow-6.2.2-cp37-cp37m-manylinux1_x86_64.whl

!pip install ../input/offwheels/pytorch_toolbelt-0.3.2-py3-none-any.whl

!pip install ../input/offwheels/pytorch_lightning-0.8.5-py3-none-any.whl

!pip install ../input/offwheels/imagecorruptions-1.1.0-py3-none-any.whl

!pip install ../input/offwheels/albumentations-0.4.6-py3-none-any.whl

!pip install ../input/offwheels/iglovikov_helper_functions-0.0.38-py2.py3-none-any.whl
!cp -r ../input/retinaface . 

!cp -r ../input/configs . 

!ls 
!python -m retinaface.inference -i ../input/global-wheat-detection/test -c ../input/configs/2020-07-20.yaml -o ./test_output -w ../input/weights/epoch29.ckpt -v
!ls test_output
import cv2 

import matplotlib.pyplot as plt 

from PIL import Image 

from pathlib import Path 





def plot_image_examples(files, rows=3, cols=3, title='Image examples'):

    fig, axs = plt.subplots(rows, cols, figsize=(20,20))

    

    i = 0 

    for row in range(rows):

        for col in range(cols):

            img = cv2.imread(str(files[i]))

            axs[row, col].imshow(img)

            axs[row, col].axis('off')

            i += 1



    plt.suptitle(title)



    

image_files = list(Path('./test_output/viz').rglob('*.jpg'))

plot_image_examples(image_files)
!ls ./test_output/viz
import pandas as pd

from pathlib import Path

import json

from operator import itemgetter 





def read_labels(labels_path): 

    label_files = Path(labels_path).rglob('*.json') 

    for label in label_files:  

        with open(label) as f: 

            data = json.load(f)



            file_id = Path(data['file_name']).stem 



            items = []

            for annot in data['annotations']:

                item = (annot['score'], *annot['bbox']) 

                items.append(item)

            items = sorted(items, key=itemgetter(0))



        yield file_id, items 



        

res = []

gen = read_labels('./test_output/labels')

to_format = lambda x: "{0:.4f} {1} {2} {3} {4}".format(x[0], x[1], x[2], x[3], x[4])

for file_id, items in gen:

    preds = [to_format(item) for item in items]

    preds_str = " ".join(preds)

    res.append({'image_id': file_id,'PredictionString': preds_str})



    

res = pd.DataFrame(res, columns=['image_id', 'PredictionString'])

res.sort_values(by='image_id', inplace=True)
sample_sbm = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

sample_sbm = sample_sbm.drop(columns='PredictionString')

sample_sbm 
res = sample_sbm.merge(res, how='left', on=['image_id'])
!rm -rf ./*
res.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv')