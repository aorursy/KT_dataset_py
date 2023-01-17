import os

!git clone https://github.com/dmlc/gluon-cv.git

os.chdir('gluon-cv')

!pip install .

!cp /kaggle/input/editing5/segmentation.py /opt/conda/lib/python3.6/site-packages/gluoncv/data/mscoco/

!cp /kaggle/input/trainingtools4/train.py /kaggle/working

os.chdir('..')

!cp -a /kaggle/input/cocoformat1 /kaggle/working

import os

os.listdir('/kaggle/input/pretrained')
!python train.py --dataset coco --model deeplab --resume /kaggle/input/pretrained/epoch_0018_mIoU_0.7600.params --ngpus 1 --aux --backbone resnet50 --lr 0.001 --syncbn  --checkname res50  --epochs 20 --save-dir weights --batch-size 8 --workers 16 --crop-size 256

import zipfile

def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('/kaggle/working/weights.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('/kaggle/working/weights/', zipf)

zipf.close()
!rm -rf cocoformat1

!rm -rf gluon-cv

!rm -rf weights

!rm train.py