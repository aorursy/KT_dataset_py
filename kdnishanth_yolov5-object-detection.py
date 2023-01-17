from IPython.display import Image
Image("../input/yolov5/Screenshot from 2020-08-22 09-12-45.png", width = "800px")
!git clone https://github.com/ultralytics/yolov5.git
%cd ./yolov5/
!pip install -r requirements.txt
!python train.py --img 640 --batch 16 --epochs 500 --data ./data/coco.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --cache

!python detect.py --source "inference/images/bus.jpg"  --conf 0.4
from IPython.display import Image
Image("inference/images/bus.jpg", width = "500px")