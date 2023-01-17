import numpy as np
import torch
data_np = np.arange(12).reshape(3, 4)
data_py = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
t_long = torch.LongTensor(data_py)
t_float = torch.FloatTensor(data_np)
t_T = torch.Tensor(data_py)
t_t = torch.tensor(data_np)
t_as = torch.as_tensor(data_py)
t_from = torch.from_numpy(data_np)
print(t_long)
import matplotlib.pyplot as plt

relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()
softplus = torch.nn.Softplus()
leakyrelu = torch.nn.LeakyReLU(0.2)
elu = torch.nn.ELU()
tensor = torch.linspace(-5, 5, 200)

t_relu = relu(tensor).numpy()
t_sigmoid = sigmoid(tensor).numpy()
t_tanh = tanh(tensor).numpy()
t_softplus = softplus(tensor).numpy()
t_leakyrelu = leakyrelu(tensor).numpy()
t_elu = elu(tensor).numpy()

x = tensor.numpy()
plt.figure(1, figsize=(8, 9))
plt.subplot(321)
plt.plot(x, t_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(322)
plt.plot(x, t_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(323)
plt.plot(x, t_relu, c='blue', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(324)
plt.plot(x, t_softplus, c='blue', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.subplot(325)
plt.plot(x, t_leakyrelu, c='blue', label='LeakyReLU')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(326)
plt.plot(x, t_elu, c='blue', label='ELU')
plt.ylim((-2, 6))
plt.legend(loc='best')

plt.show()
