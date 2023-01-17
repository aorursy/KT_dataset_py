!pip3 install pretrainedmodels
from pretrainedmodels import se_resnet50, vgg16
import torch as th
from time import time
th.backends.cudnn.benchmark = True
batch_size = 32
X = th.randn(batch_size, 3, 224, 224, requires_grad=False).cuda()

def time_model(model_fn, N=100, warmup=10):
    model = model_fn(pretrained=None).cuda()
    with th.no_grad():
        for n in range(warmup):
            out = model(X)
        
        tic = time()
        for n in range(N):
            out = model(X)
        toc = time()
    duration = toc - tic
    per_batch_duration = duration / N
    return per_batch_duration
num_images = 15606
num_batches = num_images / batch_size
for model_fn in [se_resnet50, vgg16]:
    per_batch_duration = time_model(model_fn)
    print("%s: %.2f seconds." % (model_fn.__name__, per_batch_duration))
    total_duration = num_batches * per_batch_duration
    print("[Total] %s: %.2f seconds." % (model_fn.__name__, total_duration))

