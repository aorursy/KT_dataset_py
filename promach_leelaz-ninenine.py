!apt-get update

!apt-get install -y --fix-missing --no-install-recommends cuda-compiler-9-2 cuda-cublas-dev-9-2 cuda-cudart-dev-9-2
!wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.3.1/cudnn-9.2-linux-x64-v7.3.1.20.tgz

!cd /usr/local && tar -xzvf /content/cudnn-9.2-linux-x64-v7.3.1.20.tgz

!chmod a+r /usr/local/cuda/lib64/libcudnn*
!apt-get install -y ninja-build libprotobuf-dev protobuf-compiler

!pip3 install meson
!apt install -y libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
!git clone https://github.com/ihavnoid/leelaz-ninenine/
%cd /content/leelaz-ninenine/src

!ls -al

!make clean && make
%cd /content/leelaz-ninenine/

!cp /content/leelaz-ninenine/example-data/initial_9x9.txt /content/leelaz-ninenine/training/tf

!/content/leelaz-ninenine/minitrain.sh --gpu 0
!/content/leelaz-ninenine/src/leelaz --weights ../example-data/initial_9x9.txt
#!kill -9 -1