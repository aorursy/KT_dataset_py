import numpy as np
import torch
import pandas as pd
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls -lha kaggle.json
!kaggle -v
!kaggle competitions download -c caltech101-1d
device = 'cuda'
!unzip caltech101-1d.zip

train_set = np.load('train_set.npy')
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
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 0.49
total_batch = len(train_data_loader)
training_epochs = 3
model.train()

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in train_data_loader:



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
test_set = np.load('test_set.npy')
with torch.no_grad():
    x_test_data=torch.FloatTensor(test_set)
    prediction = model(x_test_data)
    correct_prediction = torch.argmax(prediction, 1)
sample_submit = pd.read_csv('sample_submission (4).csv')
for x in range(len(sample_submit)):
    sample_submit['class'][x] = correct_prediction[x]
sample_submit.to_csv('submission.csv',index=False)
!kaggle competitions submit -c caltech101-1d -f submission.csv -m "Message"
