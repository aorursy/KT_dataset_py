 # Bibliotecas funcionais
import numpy as np
import matplotlib.pyplot as plt

epochs = 10
batch_size = 70

# Biblioteca do TensorFlow
import tensorflow as tf

# Biblioteca do PyTorch
import torch
import torchvision
!pip install --upgrade pip
!pip install torchsummary
from torchsummary import summary
# Obtendo dados do TensorFlow
(x_trainTF_, y_trainTF_), _ = tf.keras.datasets.mnist.load_data()
x_trainTF = x_trainTF_.reshape(60000, 784).astype('float32')/255
y_trainTF = tf.keras.utils.to_categorical(y_trainTF_, num_classes=10)
# Obtendo dados do PyTorch
xy_trainPT = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

xy_trainPT_loader = torch.utils.data.DataLoader(xy_trainPT, batch_size=batch_size)
# Verificando os dados do TensorFlow
print("TensorFlow:")
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(x_trainTF_[idx], cmap=plt.cm.binary)
    ax.set_title(str(y_trainTF_[idx]))
# Verificando os dados do Pytorch
print("Pytorch:")
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    image, label = xy_trainPT [idx]
    ax.imshow(torch.squeeze(image, dim = 0).numpy(), cmap=plt.cm.binary)
    ax.set_title(str(label))
  # Modelo do TensorFlow com a API Keras
modelTF = tf.keras.Sequential([
                             tf.keras.layers.Dense(10,activation='sigmoid',input_shape=(784,)),
                             tf.keras.layers.Dense(10,activation='softmax')
            ])  
modelTF.summary()
  # Modelo do PyTorch
modelPT=torch.nn.Sequential(torch.nn.Linear(784,10), 
                          torch.nn.Sigmoid(), 
                          torch.nn.Linear(10,10), 
                          torch.nn.LogSoftmax(dim=1) 
                         )  

print(modelPT)
# Definindo o TensorFlow
modelTF.compile(
               loss="categorical_crossentropy",
               optimizer=tf.optimizers.SGD(lr=0.01),
               metrics = ['accuracy']
               )
# Definindo o PyTorch
criterion = torch.nn.NLLLoss() 
optimizer = torch.optim.SGD(modelPT.parameters(), lr=0.01)
# Treinamento do TensorFlow
_ = modelTF.fit(x_trainTF, y_trainTF, epochs=epochs, batch_size=batch_size, verbose = 0)
# Treinamento do PyTorch
for e in range(epochs):
    for images, labels in xy_trainPT_loader:
        images = images.view(images.shape[0], -1) 
        loss = criterion(modelPT(images), labels)        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
# Medindo acurácia do TensorFlow
_, (x_testTF_, y_testTF_)= tf.keras.datasets.mnist.load_data()
x_testTF = x_testTF_.reshape(10000, 784).astype('float32')/255
y_testTF = tf.keras.utils.to_categorical(y_testTF_, num_classes=10)

_ , test_accTF = modelTF.evaluate(x_testTF, y_testTF, verbose=0)
print('\n Acurácia do modelo TensorFlow = ', test_accTF)
# Medindo acurácia do PyTorch

xy_testPT = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
xy_test_loaderPT  = torch.utils.data.DataLoader(xy_testPT)

correct_count, all_count = 0, 0
for images,labels in xy_test_loaderPT:
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    logps = modelPT(img)
    ps = torch.exp(logps)
    probab = list(ps.detach().numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("\n Acurácia do modelo PyTorch = ", (correct_count/all_count))
# Escolhendo uma imagem para predizer qual é o número com TensorFlow 
image = 7 
_ = plt.imshow(x_testTF_[image], cmap=plt.cm.binary)
prediction = modelTF.predict(x_testTF)
print("Predição do modelo: ", np.argmax(prediction[image]))
# Escolhendo uma imagem para predizer qual é o número com PyTorch 
img = 7
image, label = xy_testPT[img]
_ = plt.imshow(torch.squeeze(image, dim = 0).numpy(), cmap=plt.cm.binary)
# Necessário implmentar funções descritas em https://github.com/davidezordan/ImageClassifier para exibir a predição