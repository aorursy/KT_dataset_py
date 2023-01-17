import numpy as np
import torch
import pandas as pd
device = 'cuda'
train_set = np.load('/content/drive/My Drive/datasets/caltech-1D/train_set.npy')
x_train_data=train_set[:,:-1]
y_train_data=train_set[:,-1]


x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)
print(x_train_data.shape)
print(y_train_data.shape)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(512,512,bias=True)
linear2 = torch.nn.Linear(512,512,bias=True)
linear3 = torch.nn.Linear(512,101,bias=True)


relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)
model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3).to(device)
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 0.49
total_batch = len(train_data_loader)
training_epochs = 3
model.train()

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in train_data_loader:

      
        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # Forward 계산
        hypo = model(X)
        # Error 계산
        #hypo = torch.argmax(hypo, 1).view(-1,1)
        cost = loss(hypo, Y)
        
        # Backparopagation
        cost.backward()
        # 가중치 갱신
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost / total_batch
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
test_set = np.load('/content/drive/My Drive/datasets/caltech-1D/test_set.npy')
with torch.no_grad():
    x_test_data=torch.FloatTensor(test_set).to(device)
    prediction = model(x_test_data)
    correct_prediction = torch.argmax(prediction, 1)
sample_submit = pd.read_csv('/content/drive/My Drive/datasets/caltech-1D/sample_submission.csv')
for x in range(len(sample_submit)):
    sample_submit['class'][x] = correct_prediction[x]
sample_submit.to_csv('/content/drive/My Drive/datasets/caltech-1D/submission.csv',index=False)