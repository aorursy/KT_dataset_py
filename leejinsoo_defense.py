import numpy as np
import torch
import pandas as pd
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(72731)
torch.manual_seed(72731)
if device == 'cuda':
  torch.cuda.manual_seed_all(72731)
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp /content/drive/My\ Drive/Colab\ Notebooks/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c caltech101-1d
!unzip caltech101-1d.zip
pd_train = pd.read_csv('train_1D.csv')
train_set = pd_train.to_numpy()
train_set.shape
x_train_data=train_set[:5205,:-1]
y_train_data=train_set[:5205,-1]

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)

x_val_data = train_set[5205:,:-1]
y_val_data = train_set[5205:,-1]

x_val_data=torch.FloatTensor(x_val_data)
y_val_data=torch.FloatTensor(y_val_data)
print(x_train_data.shape)
print(y_train_data.shape)
print(x_val_data.shape)
print(y_val_data.shape)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=128,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(512,1024,bias=True)
linear2 = torch.nn.Linear(1024,1024,bias=True)
linear3 = torch.nn.Linear(1024,512,bias=True)
linear4 = torch.nn.Linear(512,101,bias=True)


relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
model = torch.nn.Sequential(linear1, relu,#dropout,
                            linear2, relu,#dropout,
                            linear3, relu,#dropout,
                            linear4
                            ).to(device)
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 0.49
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
def evaluation(model, x_val_data, y_val_data):
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        model.eval()
        x_val_data = x_val_data.to(device)
        prediction = model(x_val_data)
        prediction = torch.argmax(prediction, 1)
        correct = prediction.cpu() == y_val_data
        score = correct.sum().item()/len(y_val_data)

    return score*100
total_batch = len(train_data_loader)
training_epochs = 100
model.train()
best_score = 0
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in train_data_loader:

      
        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # Forward 계산
        hypo = model(X)
        
        cost = loss(hypo, Y)
        
        # Backparopagation
        cost.backward()
        # 가중치 갱신
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost / total_batch

    lr_ = optimizer.state_dict()
    lr_ = lr_['param_groups'][0]['lr']
    score = evaluation(model, x_val_data, y_val_data)
    scheduler.step(score)
    if best_score < score:
        best_score = score
        param = torch.save(model.state_dict(), 'param.pth')
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'score  = ','{:.5f}'.format(score),'best_score = ','{:.5f}'.format(best_score),'lr = ','{:.10f}'.format(lr_))

print('Learning finished')
model.load_state_dict(torch.load('param.pth'))
pd_train = pd.read_csv('test_1D.csv')
test_set = pd_train.to_numpy()
test_set.shape
model.eval()
with torch.no_grad():
    x_test_data=torch.FloatTensor(test_set).to(device)
    prediction = model(x_test_data)
    correct_prediction = torch.argmax(prediction, 1)
    correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)
sample_submit = pd.read_csv('sample_submission_1D.csv')
for x in range(len(sample_submit)):
    sample_submit['class'][x] = correct_prediction[x]
sample_submit
sample_submit.to_csv('submission.csv',index=False)
!kaggle competitions submit -c caltech101-1d -f submission.csv -m "submit_defense"