# !curl https://raw.githubusercontent.com/jysohn23/xla/env-setup-gce/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !cat pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
!rm -rf *.whl
!gsutil cp gs://tpu-pytorch/wheels/cuda/101/torch-nightly-cp37-cp37m-linux_x86_64.whl .
!gsutil cp gs://tpu-pytorch/wheels/cuda/101/torch_xla-nightly-cp37-cp37m-linux_x86_64.whl .
!gsutil cp gs://tpu-pytorch/wheels/cuda/101/torchvision-nightly-cp37-cp37m-linux_x86_64.whl .
!pip uninstall -y torch torchvision
!pip install *.whl
!apt install libomp5
# this crashes
import torch
import torch_xla
import torchvision
torch
!ldd --version
!nvcc --version