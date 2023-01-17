!pip install --upgrade https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.55-cp37-none-manylinux2010_x86_64.whl
!pip install jax
import os
import json

print(os.environ["TPU_NAME"])
import jax

jax.config.update("jax_xla_backend", "tpu_driver")
jax.config.update("jax_backend_target", os.environ["TPU_NAME"])

print(jax.devices())