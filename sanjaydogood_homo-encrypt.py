%mkdir homo_encrypt

%cd homo_encrypt

%mkdir model_data models
!git clone https://github.com/tensorflow/models tf-models
!pip install tensorflow-gpu==1.15

!pip install Cython --upgrade

!pip install contextlib2 --upgrade

!pip install pillow --upgrade

!pip install lxml --upgrade

!pip install jupyter --upgrade

!pip install matplotlib --upgrade

%cd tf-models

!git clone https://github.com/cocodataset/cocoapi.git

%cd cocoapi/PythonAPI

!make

%cp -r pycocotools /kaggle/working/homo_encrypt/tf-models/research/
%cd /kaggle/working/homo_encrypt/tf-models/research/

!apt-get install -qq protobuf-compiler

!protoc object_detection/protos/*.proto --python_out=.
%cd /kaggle/working/homo_encrypt/models

!curl -L http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz -o ssd_mobilenet_v2_quantized.tar.gz

!tar -xvf ssd_mobilenet_v2_quantized.tar.gz

!rm ssd_mobilenet_v2_quantized.tar.gz
!cat /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config
!sed -i 's/num_classes: 90/num_classes: 3/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED\/model.ckpt"/fine_tune_checkpoint: "\/kaggle\/working\/homo_encrypt\/models\/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03\/model.ckpt"/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/num_steps: 20000000/num_steps: 3000/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/label_map_path: "PATH_TO_BE_CONFIGURED\/mscoco_label_map.pbtxt"/label_map_path: "\/kaggle\/working\/homo_encrypt\/models\/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03\/label_map.pbtxt"/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/input_path: "PATH_TO_BE_CONFIGURED\/mscoco_train.record-00000-of-00100"/input_path: "\/kaggle\/input\/face-data-tfrecords\/homo_encrypt\/image_data\/train.record"/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/input_path: "PATH_TO_BE_CONFIGURED\/mscoco_val.record-00000-of-00010"/input_path: "\/kaggle\/input\/face-data-tfrecords\/homo_encrypt\/image_data\/test.record"/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/num_examples: 8000/num_examples: 1227/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config

!sed -i 's/batch_size: 24/batch_size: 16/g' /kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config
with open('/kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/label_map.pbtxt','w') as pbtxt:

    pbtxt.write("item {\n\n    id: 1\n    name: 'Sanjay'\n}\n\nitem {\n\n    id: 2\n    name: 'Swathi'\n}\n\nitem {\n\n    id: 3\n    name: 'Trump'\n}")
%set_env PYTHONPATH=/kaggle/lib/kagglegym:/kaggle/lib:/kaggle/working/homo_encrypt/tf-models/research:/kaggle/working/homo_encrypt/tf-models/research/slim
%cd /kaggle/working/homo_encrypt/tf-models/research/object_detection/legacy

# %mkdir /kaggle/working/training

!python train.py --logtostderr --train_dir=/kaggle/working/homo_encrypt/training --pipeline_config_path=/kaggle/working/homo_encrypt/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

# !unzip ngrok-stable-linux-amd64.zip
# %reload_ext tensorboard

# %tensorboard --logdir=/kaggle/working/homo_encrypt/training/
!apt-get install zip

!zip -r /kaggle/working/homo_encrypt.zip /kaggle/working/homo_encrypt/
from IPython.display import FileLink

import os

os.chdir("/kaggle/working")

FileLink(r'homo_encrypt.zip')