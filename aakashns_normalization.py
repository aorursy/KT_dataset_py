import torch
# Training data
train_data = torch.tensor([[1, 2, 3], 
                           [6, 9, 18], 
                           [4, 12, 24.]])
train_data
# Get column wise mean by setting dim=1
train_mean = torch.mean(train_data, dim=1)
train_mean
# Get column wise standard deviation by setting dim=1
train_std = torch.std(train_data, dim=1)
train_std
# Normalize by broadcasting train_man and train_std to match dimensions of train_data
train_normed = (train_data - train_mean) / train_std
train_normed
# Test
test_data = torch.tensor([[4, 6, 1], 
                          [3, 9, 2], 
                          [5, 11, 17.]])
# Normalize using training data mean & std
test_normed = (test_data - train_mean) / train_std