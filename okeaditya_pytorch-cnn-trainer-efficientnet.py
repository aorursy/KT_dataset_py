!pip install -q git+git://github.com/oke-aditya/pytorch_cnn_trainer.git
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from pytorch_cnn_trainer import dataset
from pytorch_cnn_trainer import model_factory
from pytorch_cnn_trainer import utils
from pytorch_cnn_trainer import engine
import timm
# To list all the models supported 
# print(timm.list_models())
MODEL_NAME = "efficientnet_b3"
NUM_ClASSES = 10
IN_CHANNELS = 1
PRETRAINED = True  # If True -> Fine Tuning else Scratch Training
EPOCHS = 5
EARLY_STOPPING = True  # If you need early stoppoing for validation loss
SAVE_PATH = "{}.pt".format(MODEL_NAME)
SEED = 42
# Train and validation Transforms which you would like
train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sets seed for your entire run
utils.seed_everything(SEED)
print("Setting Seed for the run, seed = {}".format(SEED))
# For example I provide ready to use CIFAR10, you can create your own dataset too.
print("Creating Train and Validation Dataset")
train_set, valid_set = dataset.create_fashion_mnist_dataset(train_transforms, valid_transforms)
print("Train and Validation Datasets Created")
print("Creating DataLoaders")
train_loader, valid_loader = dataset.create_loaders(train_set, train_set, train_batch_size=256, valid_batch_size=256)
print("Train and Validation Dataloaders Created")
print("Creating Model")
model = model_factory.create_timm_model(MODEL_NAME, num_classes=NUM_ClASSES, in_channels=IN_CHANNELS, pretrained=True)
if torch.cuda.is_available():
    print("Model Created. Moving it to CUDA")
else:
    print("Model Created. Training on CPU only")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # All classification problems usually need Cross entropy loss
early_stopper = utils.EarlyStopping(patience=7, verbose=True, path=SAVE_PATH)
history = engine.fit(
    epochs=EPOCHS,
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    device=device,
    optimizer=optimizer,
    early_stopper=early_stopper,
)
print("Done !!")
for epoch in tqdm(range(EPOCHS)):
    print()
    print("Training Epoch = {}".format(epoch))
    train_metrics = engine.train_step(model, train_loader, criterion, device, optimizer)
    print()
    print("Validating Epoch = {}".format(epoch))
    valid_metrics = engine.val_step(model, valid_loader, criterion, device)
    validation_loss = valid_metrics["loss"]
    early_stopper(validation_loss, model=model)

    if early_stopper.early_stop:
        print("Saving Model and Early Stopping")
        print("Early Stopping. Ran out of Patience for validation loss")
        break

    print("Done Training, Model Saved to Disk")