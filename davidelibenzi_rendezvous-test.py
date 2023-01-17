!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
%%writefile train.py
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import time

def simple_map_fn(rank, flags):
  device = xm.xla_device()  
  print("Process", rank ,"is using", xm.xla_real_devices([str(device)])[0])
  xm.rendezvous('init')
  if rank == 0:
    time.sleep(1)

flags = {}
xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')

!TF_CPP_VMODULE=mesh_service=5,computation_client=0 python train.py