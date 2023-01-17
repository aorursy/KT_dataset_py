import mxnet as mx
from mxnet import nd

a = nd.array([1,2,3], ctx=mx.gpu())

a
