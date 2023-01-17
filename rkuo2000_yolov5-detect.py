!git clone https://github.com/ultralytics/yolov5  

!pip install -qr yolov5/requirements.txt  # install dependencies (ignore errors)

%cd yolov5



import torch

from IPython.display import Image, clear_output  # to display images

from utils.google_utils import gdrive_download  # to download models/datasets



clear_output()

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
!python detect.py --weights yolov5s.pt --img-size 416 --conf 0.4 --source inference/images/bus.jpg

Image(filename='inference/output/bus.jpg', width=600)
!python detect.py --weights yolov5s.pt --img-size 640 --conf 0.4 --source inference/images/zidane.jpg

Image(filename='inference/output/zidane.jpg', width=600)
!python detect.py --weights yolov5s.pt --img-size 640 --conf 0.4 --source ../../input/input-video/Shibuya_Crossing_FullHD.mp4