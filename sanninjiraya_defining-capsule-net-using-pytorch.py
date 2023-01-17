import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from numpy import prod
from time import time
def squash(vec,dim=-1):
  squared_normal=torch.sum(vec**2,dim=dim,keepdim=True)
  fn=squared_normal / (1 + squared_normal) * vec / (torch.sqrt(squared_normal) + 1e-8)
  return fn
class PrimaryCapsules(nn.Module):
  def __init__(self,in_channels, out_channels, dim_caps,kernel_size=9, stride=2, padding=0):
    super(PrimaryCapsules,self).__init__()
    self.dim_caps=dim_caps
    self._caps_channel = int(out_channels / dim_caps)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
  def forward(self,x):
    out=self.conv(x)
    out = out.view(out.size(0), self._caps_channel, out.size(2), out.size(3), self.dim_caps)
    out = out.view(out.size(0), -1, self.dim_caps)
    out=squash(out)
    return out
class Router(nn.Module):
  def __init__(self,in_dim, in_caps, num_caps, dim_caps, num_routing):
      super(Router,self).__init__()
      self.in_dim = in_dim
      self.in_caps = in_caps
      self.num_caps = num_caps
      self.dim_caps = dim_caps
      self.num_routing = num_routing
      self.W = nn.Parameter( 0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim ))
    
  def __repr__(self):
      
      tab= '  '
      line = '\n'
      next = ' -> '
      res = self.__class__.__name__ + '('
      res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
      res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
      res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
      res = res + 'Routing No =' + str(self.num_routing) + ')'
      res = res + line + ')'
      return res
  
  def forward(self,x):
    batch_size = x.size(0)
    x = x.unsqueeze(1).unsqueeze(4)
    # W @ x =(1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =(batch_size, num_caps, in_caps, dim_caps, 1)
    u_hat = torch.matmul(self.W, x)
    u_hat = u_hat.squeeze(-1)
    #Prevent flow of Gradients
    temp_u_hat = u_hat.detach()
    b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1)
    
    for route_iter in range(self.num_routing-1):
      sc = F.softmax(b, dim=1)
      # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) =(batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->(batch_size, num_caps, dim_caps)
      vec = (sc * temp_u_hat).sum(dim=2)
      v = squash(vec)
      uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
      b += uv
      
    sc = F.softmax(b, dim=1)
    vec = (sc * u_hat).sum(dim=2)
    v = squash(vec)
    return v
class CapsuleNet(nn.Module):
	def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, num_routing, kernel_size=9):
		super(CapsuleNet,self).__init__()
		self.img_shape = img_shape
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=1, bias=True)
		self.relu = nn.ReLU(inplace=True)

		self.primary = PrimaryCapsules(channels, channels, primary_dim, kernel_size)
		
		primary_caps = int(channels / primary_dim * ( img_shape[1] - 2*(kernel_size-1) ) * ( img_shape[2] - 2*(kernel_size-1) ) / 4)
		self.digits = Router(primary_dim, primary_caps, num_classes, out_dim, num_routing)

		self.decoder = nn.Sequential(
			nn.Linear(out_dim * num_classes, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, int(prod(img_shape)) ),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = self.primary(out)
		out = self.digits(out)
		preds = torch.norm(out, dim=-1)

		# Reconstruct the *predicted* image
		_, max_length_idx = preds.max(dim=1)	
		y = torch.eye(self.num_classes)
		y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

		reconstructions = self.decoder( (out*y).view(out.size(0), -1) )
		reconstructions = reconstructions.view(-1, *self.img_shape)

		return preds, reconstructions

class CapsuleLoss(nn.Module):
	def __init__(self,loss_lambda=0.5, recon_loss_scale=5e-4, size_average=False):

		super(CapsuleLoss,self).__init__()
		self.size_average = size_average
		self.margin_loss = MarginLoss(size_average=size_average, loss_lambda=loss_lambda)
		self.reconstruction_loss = nn.MSELoss(size_average=size_average)
		self.recon_loss_scale = recon_loss_scale

	def forward(self,inputs, labels, images, reconstructions):
		margin_loss = self.margin_loss(inputs, labels)
		reconstruction_loss = self.reconstruction_loss(reconstructions, images)
		caps_loss = (margin_loss + self.recon_loss_scale * reconstruction_loss)

		return caps_loss