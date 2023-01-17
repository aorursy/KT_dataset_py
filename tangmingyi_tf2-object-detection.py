!conda list | grep cuda
%mkdir protoc
%cd protoc
!wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -q
!unzip -o protobuf.zip
!rm protobuf.zip
%cd /kaggle/working
%mkdir code
%cd code
!rm -fr models
!git clone https://github.com/tensorflow/models.git
!rm -fr models/.git
# compile ProtoBuffers
%cd models/research
!../../../protoc/bin/protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
%cp object_detection/packages/tf2/setup.py .
!python -m pip install .
# %cd code/models/research
!pip list | grep obj
!pwd
!python object_detection/builders/model_builder_test.py