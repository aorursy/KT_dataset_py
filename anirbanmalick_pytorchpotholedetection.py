import os
import pandas as pd
import numpy as np
import json
import warnings
import random
import torch
import shutil
from tqdm import tqdm
from pathlib import Path
torch.cuda.is_available()
warnings.filterwarnings('ignore')
def seed_system(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
seed = seed_system()
def flush_folders(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory) 
    return True
input_data_path =  '/kaggle/input/annotatedpotholesdataset/'
working_dir = '/kaggle/working/annotatedpotholesdataset/'
images_path = Path(input_data_path+'annotated-images/')

make_folders = ['/images','/labels']
train_valid = ['/train','/test']


def create_folder_structure(home=working_dir,make_folders=make_folders,train_valid=train_valid):
    if not os.path.exists(home):
        os.makedirs(home)
    for path in tqdm(make_folders):
        for c in train_valid:
            data_path = home+path+c
            if not os.path.exists(data_path):
                os.makedirs(data_path)
    return True

def copy_data(files,destination):
    for f in files:
        shutil.copy(f,destination)
    return
flush=flush_folders(working_dir)
create_folder_structure()
image_files = [f for f in os.listdir(input_data_path+'annotated-images') if f.endswith('.jpg')]
xml_files = [f for f in os.listdir(input_data_path+'annotated-images') if f.endswith('.xml')]
with open(input_data_path+'splits.json','r') as f:
    split = json.load(f)
    
for key in split.keys():
    list_of_indices = split[key]
    print(key)
    filtered_indices = [input_data_path+'annotated-images/'+i[:-4]+'.jpg' for i in xml_files if i in list_of_indices]
    copy_data(filtered_indices,destination=working_dir+'/images/'+key)
    
# os.listdir(working_dir+'images/test')
import xml.etree.ElementTree as ET
from pathlib import Path
def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def generate_anno_df(anno_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        anno['image_name'] = root.find("./filename").text[:-4]
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
        anno_list.append(anno)
    return pd.DataFrame(anno_list)
df_annotation = generate_anno_df(input_data_path+'annotated-images')
df_annotation.head()
train_indices = [f[:-4] for f in os.listdir(working_dir+'images/train')]
valid_indices = [f[:-4] for f in os.listdir(working_dir+'images/test')]
df_annotation_train = df_annotation[df_annotation.image_name.isin(train_indices)]
df_annotation_valid = df_annotation[df_annotation.image_name.isin(valid_indices)]
def get_annotations_to_txt(df_anno,mode):
    for image in list(df_anno.image_name.unique()):
        df_to_txt = df_anno[df_anno.image_name==image]
        df_to_txt.to_csv(working_dir+'labels/'+mode+'/'+image+'.txt', sep=' ', index=False, header=False)
    return True

_ = get_annotations_to_txt(df_annotation_train,mode='train')
_ = get_annotations_to_txt(df_annotation_valid,mode='test')
# install dependencies: 
!pip install pyyaml==5.1 pycocotools>=2.0.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
assert torch.__version__.startswith("1.6")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
from detectron2.structures import BoxMode
import cv2 as cv

label_folder = working_dir + 'labels/'
image_folder = working_dir + 'images/'

def prepare_dataset(path,image_folder=image_folder,label_folder=label_folder):
    list_dataset = []
    files = os.listdir(label_folder+path)
    print("Number of lables in", path, str(len(files)))
    for i,file in enumerate(files):
        try:
            all_vals = []
            with open(label_folder+path+file,'r') as f:
                all_vals.append(f.readlines())
                f.close()
            all_vals = [line.split() for line in all_vals[0]]
            master_dict = {}
            annotations = []
            for i,vals in enumerate(all_vals):
                image_id = file[:-4]
                #label_file_name = label_folder+path+file
                image_file_name= vals[0]
                width,height = int(vals[2]),int(vals[3])
                label = vals[4]
                x_min,y_min = int(vals[5]),int(vals[6])
                x_max,y_max = int(vals[7]),int(vals[8])
                bbox = [x_min,y_min,x_max,y_max]

                obj = {
                    "bbox": [x_min, y_min,x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                annotations.append(obj)
                master_dict['file_name']=image_file_name
                master_dict['image_id'] = image_id
                master_dict['height'] = height
                master_dict['width'] = width
        except Exception as e:
            print(e)
        master_dict['annotations'] = annotations
        list_dataset.append(master_dict)
    return list_dataset


def prepare_master_dict(path):

    unique_image_id_train = list(set(os.listdir(image_folder+path)))
    unique_image_id_train = [i[:-4] for i in unique_image_id_train]
    
    list_dataset = prepare_dataset(path=path)
    return list_dataset
#/kaggle/input/annotatedpotholesdataset/annotated-images/img-430.jpg img-430 451 300 pothole 225 230 371 277

list_dataset = prepare_master_dict(path='test/')
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
for d in ["train/", "test/"]:
    DatasetCatalog.register("bc_v7_" + d, lambda d=d: prepare_master_dict(path=d))
    MetadataCatalog.get("bc_v7_" + d).set(thing_classes=['pothole'])
import random
import matplotlib.pyplot as plt
valid_metadata = MetadataCatalog.get("bc_v7_" + 'test')
for d in random.sample(list_dataset, 1):
    print(d)
    img = cv.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=valid_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :, ::-1])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

model_yaml =  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
cfg.DATASETS.TRAIN = ("bc_v7_train/",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)  
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0005  
cfg.SOLVER.MAX_ITER = 1000  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.DATASETS.TEST = ("bc_v7_test/", )
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
dataset_dicts = prepare_master_dict(path='test/')

for d in random.sample(dataset_dicts, 3):    
    im = cv.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=valid_metadata, 
                   scale=0.5, 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:,:,::-1])
    plt.show()
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("bc_v7_test/", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "bc_v7_test/")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
from detectron2.modeling import build_model
model = build_model(cfg)  
model