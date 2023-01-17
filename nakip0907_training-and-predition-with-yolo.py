from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
!apt-get update
!apt-get upgrade
!apt-get install build-essential
!apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
!apt-get install libavcodec-dev libavformat-dev libswscale-d


!apt-get -y install cmake
!which cmake

!cmake --version

#Installing OpenCV
!apt-get install libopencv-dev
!git clone https://github.com/AlexeyAB/darknet/
!apt-get install vim
%cd darknet
!ls
!wget https://pjreddie.com/media/files/yolov2.cfg
!ls
!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
!sed -i 's/GPU=0/GPU=1/g' Makefile
!ls
%cd ../
!ls

!apt install g++-5
!apt install gcc-5

!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
!update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
!update-alternatives --set cc /usr/bin/gcc
!update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
!update-alternatives --set c++ /usr/bin/g++
!apt update -qq;
!wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!apt-get update -qq

#Here were are installing compilers and creating some links
!apt-get install cuda -y -qq #gcc-5 g++-5 
#!ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc
#!ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++
!apt update
!apt upgrade
!apt install cuda-8.0 -y
%cd darknet
!make
%cd /content/darknet
!ls
!./darknet partial /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/yolov3-tiny.cfg /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/yolov3-tiny.weights yolov3-tiny.conv.15 15
!./darknet detector train /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/obj.data /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/yolov3-tiny_obj.cfg /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/yolov3-tiny.conv.15 -dont_show
!./darknet detector test /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/obj.data /content/drive/'My Drive'/Disabled_Person_Detect_by_YOLO/yolov3-tiny_obj.cfg /content/yolov3-tiny_obj_9000.weights /content/'000000.jpg'