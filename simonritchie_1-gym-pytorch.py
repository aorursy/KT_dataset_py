import gym
gym.__version__
env = gym.make('CartPole-v0')
env.action_space
env.action_space.n
env.observation_space
env.observation_space.shape
env.observation_space.low
env.observation_space.high
env.reset()
obs, reward, done, _ = env.step(action=1)
obs
reward
done
env.reset()
total_reward = 0
step_num = 1
while True:
    obs, reward, done, _ = env.step(action=1)
    total_reward += reward
    print('step', step_num, obs)
    step_num += 1
    
    if done:
        break
print('total_reward', total_reward)
import torch
torch.__version__
tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
tensor
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.tensor(arr)
tensor
tensor = torch.tensor(arr, dtype=torch.float32)
tensor
# 以下のコードはエラーになります。
#tensor = torch.tensor(arr, dtype=np.float32)
summed_tensor = tensor.sum()
summed_tensor
summed_tensor.item()
# 以下のコードはエラーになります。
#tensor.item()
tensor.device
tensor_gpu = tensor.to('cuda')
tensor_gpu
tensor_gpu = tensor.to('cuda:0')
tensor.exp()
tensor
tensor.exp_()
tensor
vector_1 = torch.FloatTensor([1, 2])
vector_1.requires_grad = True
vector_2 = torch.FloatTensor([3, 4])
added_vector = vector_1 + vector_2
added_vector
summed_scalar = added_vector.sum()
summed_scalar
print('vector_1.is_leaf', vector_1.is_leaf)
print('vector_2.is_leaf', vector_2.is_leaf)
print('added_vector.is_leaf', added_vector.is_leaf)
print('summed_scalar.is_leaf', summed_scalar.is_leaf)
print('----------------')
print('vector_1.requires_grad', vector_1.requires_grad)
print('vector_2.requires_grad', vector_2.requires_grad)
print('added_vector.requires_grad', added_vector.requires_grad)
print('summed_scalar.requires_grad', summed_scalar.requires_grad)
summed_scalar.backward()
vector_1.grad
vector_2.grad

