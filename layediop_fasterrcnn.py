import torch

import torchvision

import os



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, )
torch.save(model, "fasterrcnn_resnet50_fpn.pt")