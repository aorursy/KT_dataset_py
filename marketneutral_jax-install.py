!nvidia-smi
!nvcc --version
!python --version
import os
os.environ['JAX_PYTHON_VERSION'] = 'cp36'

os.environ['JAX_CUDA_VERSION'] = 'cuda100'

os.environ['JAX_PLATFORM'] = 'linux_x86_64'

os.environ['JAX_BASE_URL'] = 'https://storage.googleapis.com/jax-releases'
!pip install --upgrade $JAX_BASE_URL/$JAX_CUDA_VERSION/jaxlib-0.1.36-$JAX_PYTHON_VERSION-none-$JAX_PLATFORM.whl
!pip install --upgrade jax
import unittest



gpu_test = unittest.skipIf(len(os.environ.get('CUDA_VERSION', '')) == 0, 'Not running GPU tests')
import time

from jax import grad, jit

import jax.numpy as np



class TestJAX(unittest.TestCase):

    def tanh(self, x):  # Define a function

      y = np.exp(-2.0 * x)

      return (1.0 - y) / (1.0 + y)



    @gpu_test

    def test_JAX(self):

        grad_tanh = grad(self.tanh)

        ag = grad_tanh(1.0)

        print(f'JAX autograd test: {ag}')

        assert ag==0.4199743

        
unittest.main(argv=[''], verbosity=2, exit=False)
def slow_f(x):

  # Element-wise ops see a large benefit from fusion

  return x * x + x * 2.0



x = np.ones((5000, 5000))

fast_f = jit(slow_f)

%timeit -n10 -r3 fast_f(x)  # ~ 4.5 ms / loop on Titan X

%timeit -n10 -r3 slow_f(x)  # ~ 14.5 ms / loop (also on GPU via JAX)