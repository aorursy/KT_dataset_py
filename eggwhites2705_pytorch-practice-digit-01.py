# Creating the CUDA environment

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# Working with the data and creating the test train split

from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/digit-recognizer/train.csv', dtype=np.float32)
targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 
# create feature and targets tensor for train set. 
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long
# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

import matplotlib.pyplot as plt

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()
# Defining the model network

class Model(torch.nn.Module):

    ## Initialize
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.l1 = torch.nn.Linear(32 * 4 * 4, 10)

        self.max = torch.nn.MaxPool2d(2)

    ## Forward
    def forward(self, x):
        x = torch.nn.functional.relu(self.max(self.conv1(x)))
        x = torch.nn.functional.relu(self.max(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        return torch.nn.functional.log_softmax(x)

model = Model()
model.to(device)
# Define the loss function and optimizer

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimus = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.05)
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        inputs = images.view(100,1,28,28).to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimus.zero_grad()
        
        # Forward propagation
        outputs = model(inputs)
        
        # Calculate softmax and ross entropy loss
        loss = criterion(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimus.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                inputs = images.view(100,1,28,28).to(device)
                labels = labels.to(device)
                # Forward propagation
                outputs = model(inputs)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
# Now we are going to work on the test data set. Model.eval and test dataloader

test = pd.read_csv('../input/digit-recognizer/test.csv', dtype=np.float32)
features_numpy = test.loc[:,test.columns != "label"].values/255 # normalization
featurestest = torch.from_numpy(features_numpy)
model.eval()
predicted = []
for i in range(0,featurestest.shape[0]):
    inputs = featurestest[i].view(1,1,28,28).to(device)
    output = model(inputs)
    predicted.append(torch.max(output.data, 1)[1].item())
a= np.array(range(1,len(predicted)+1))
b=np.array(predicted)
df = pd.DataFrame({'ImageId':a,'Label':b})
df.to_csv('submission.csv', index=False)