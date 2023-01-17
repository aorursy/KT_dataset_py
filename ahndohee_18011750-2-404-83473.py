learning_rate = 0.0006
training_epochs = 300
batch_size =50
drop_prob = 0.3
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('train_sweetpotato_price.csv')
test_data=pd.read_csv('test_sweetpotato_price.csv')
train_data['year']=train_data['year']%10000/100 
x_train_data = train_data.loc[:,[i for i in train_data.keys()[:-1]]]
y_train_data=train_data[train_data.keys()[-1]]
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data = torch.FloatTensor(x_train_data)
y_train_data = torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(5,5,bias=True)
linear2 = torch.nn.Linear(5,5,bias=True)
linear3 = torch.nn.Linear(5,5,bias=True)
linear4 = torch.nn.Linear(5,1,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=drop_prob)
torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch= len(data_loader)
model.train()
for epoch in range(training_epochs):
  avg_cost = 0

  for X,Y in data_loader:
    X=X.to(device)
    Y= Y.to(device)
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/ total_batch 
  print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning finised')
with torch.no_grad():
  test_data['year']=test_data['year']%10000/100 
  x_test_data = test_data.loc[:,[i for i in test_data.keys()[:]]]
  x_test_data=np.array(x_test_data)
  x_test_data=Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submit_sample.csv')

for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()
submit