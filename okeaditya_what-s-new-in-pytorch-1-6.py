! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torchvision
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from tqdm.notebook import tqdm
import torch.optim as optim
import torch
from PIL import Image
def create_cifar10_dataset(train_transforms, valid_transforms):
    """ Creates CIFAR10 train dataset and a test dataset. 
    Args: 
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """
    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.CIFAR10(
        "./data", download=True, train=True, transform=train_transforms
    )

    valid_set = torchvision.datasets.CIFAR10(
        "./data", download=True, train=False, transform=valid_transforms
    )

    return train_set, valid_set

# Train and validation Transforms which you would like
train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
# For list of supported models use timm.list_models
MODEL_NAME = "resnet18"
NUM_CLASSES = 10
IN_CHANNELS = 3

USE_TORCHVISION = False  # If you need to use timm models set to False.

# USE_TORCHVISION = True # Should use Torchvision Models or timm models
PRETRAINED = True  # If True -> Fine Tuning else Scratch Training
EPOCHS = 3
TRAIN_BATCH_SIZE = 512  # Training Batch Size
VALID_BATCH_SIZE = 512  # Validation Batch Size
NUM_WORKERS = 4  # Workers for training and validation

EARLY_STOPPING = True  # If you need early stoppoing for validation loss
SAVE_PATH = "{}.pt".format(MODEL_NAME)

MOMENTUM = 0.8  # Use only for SGD
LEARNING_RATE = 1e-3  # Learning Rate
SEED = 42

train_dataset, valid_dataset = create_cifar10_dataset(train_transforms, valid_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, VALID_BATCH_SIZE, shuffle=False, num_workers=2)
# This simply instantiates the torchvision model which is pretrained.
# This is normal PyTorch CNN model as you would create. Nothing new here, you can replace this with your own CNN too.
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Model Created. Moving it to CUDA")
else:
    print("Model Created. Training on CPU only")

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
for epoch in tqdm(range(EPOCHS)):
    model.train()
    print(f"Started epochs: {epoch}")
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print(f"Finished epochs: {epoch}")

from torch.cuda import amp
# Creates model and optimizer in default precision
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()
for epoch in tqdm(range(EPOCHS)):
    model.train()
    print(f"Started epochs: {epoch}")
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with amp.autocast():
            output = model(inputs)
            loss = criterion(output, target)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

    print(f"Finished epochs: {epoch}")

import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(inputs)
# NOTE: some columns were removed for brevity
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# Some more statistics can be collected.
# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
model1 = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model2 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
from torchvision.models.resnet import resnet18
from torch.quantization import get_default_qconfig, quantize_jit
model = torchvision.models.resnet18(pretrained=True)
_ = model.eval()
ts_model = torch.jit.script(model).eval() # ts_model = torch.jit.trace(float_model, input)
qconfig = get_default_qconfig('fbgemm')
qconfig_dict = {'': qconfig}
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
# I am using the same valid_loader we created for CIFAR10 example in first section
# It's slightly big, so it can take some time
quantized_model = quantize_jit(ts_model, {'': qconfig}, calibrate, [valid_loader], inplace=False, debug=False)
# print(quantized_model.graph)
# We can see that model is much smaller approximately 3x
graph_mode_model_file = 'resnet18_graph_mode_quantized.pt'
orignal_model = "resnet18_unquantized.pt"
torch.jit.save(quantized_model, graph_mode_model_file)
torch.jit.save(ts_model, orignal_model)
quantized_model = quantize_jit(ts_model, {'': qconfig}, calibrate, [valid_loader], inplace=False, debug=True)
import torch.quantization
import torch.quantization._numeric_suite as ns
from torch.quantization import default_eval_fn, default_qconfig, quantize
# Quantize is false here
float_model = torchvision.models.quantization.resnet18(pretrained=True, quantize=False)

float_model.to('cpu')
float_model.eval()
float_model.fuse_model()
float_model.qconfig = torch.quantization.default_qconfig
img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
# Here is the quantized model
qmodel = quantize(float_model, default_eval_fn, img_data, inplace=False)
wt_compare_dict = ns.compare_weights(float_model.state_dict(), qmodel.state_dict())
print('keys of wt_compare_dict:')
print(wt_compare_dict.keys())

print("\nkeys of wt_compare_dict entry for conv1's weight:")
print(wt_compare_dict['conv1.weight'].keys())
print(wt_compare_dict['conv1.weight']['float'].shape)
print(wt_compare_dict['conv1.weight']['quantized'].shape)
def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)

for key in wt_compare_dict:
    print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))
import matplotlib.pyplot as plt
f = wt_compare_dict['conv1.weight']['float'].flatten()
plt.hist(f, bins = 100)
plt.title("Floating point model weights of conv1")
plt.show()

q = wt_compare_dict['conv1.weight']['quantized'].flatten().dequantize()
plt.hist(q, bins = 100)
plt.title("Quantized model weights of conv1")
plt.show()
import torchvision
import time
net = torchvision.models.resnet18(pretrained=True)
# You can try torch.jit.trace() 
scripted_net = torch.jit.script(net)
# We Set it to eval mode and we freeze the net
scripted_net.eval()
frozen_net = torch.jit.freeze(scripted_net)
def time_model(model, warmup=1, iter=100):
    start = time.time()
    inputs = torch.randn(1, 3, 224, 224) # Image Net single Image
    _ = model(inputs)
    end = time.time()
    print("Warm up time: {0:7.4f}".format(end-start))
    
    start = time.time()
    for i in range(iter):
        inputs = torch.randn(1, 3, 224, 224) # Image Net single Image
        net(inputs)
    end = time.time()
    print("Inference time: {0:7.4f}".format(end-start))

# Let us compare all the three models

# 1. Orignal resnet model

time_model(net)
# 2. Torchscript model without freezing
time_model(scripted_net)
# 3. Frozen Torchscript model
time_model(frozen_net)