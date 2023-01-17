req = """opencv-contrib-python==3.4.2.17
opencv-python==3.4.2.17
Pillow==6.2.0
progressbar2
keras_applications
"""

!echo {repr(req)} > requirements.txt
!cat ./requirements.txt
!mkdir dep
%cd dep
!pip download -r ../requirements.txt
!mkdir /kaggle/working/github
%cd /kaggle/working/github
!git clone https://github.com/cocodataset/cocoapi.git
!git clone https://github.com/xuannianz/EfficientDet.git
