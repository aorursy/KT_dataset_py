import mxnet as mx

from mxnet import nd, autograd, gluon

from mxnet.gluon import nn

from mxnet.gluon.data import vision

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm

from distutils.dir_util import copy_tree

import os

plt.style.use('seaborn')



ctx = mx.gpu()
fromDirectory = "../input/natural-images/data/natural_images"

toDirectory = "../../training/"



from distutils.dir_util import copy_tree

copy_tree(fromDirectory, toDirectory)
os.chdir('../../')

os.listdir('./')
train_root = "./training/"

val_root = "./val/"

test_root = "./test/"



os.mkdir(val_root)

os.mkdir(test_root)



categories = os.listdir(train_root)

categories.sort()



print("Categories:", categories)

print("Total Categories:", len(categories))



for category in categories:

    os.mkdir(os.path.join(val_root, category))    

    os.mkdir(os.path.join(test_root, category))

    print(f"{len(os.listdir(train_root + category))} images for '{category}' category")
for category in categories:

    print(f"Creating validation and testing dataset for '{category}' category")

    for _ in range(10):

        images = os.listdir(train_root + category)

        idx = int(nd.random.randint(0, len(images)).asscalar())

        image = images[idx]

        os.rename(os.path.join(train_root, category, image), os.path.join(val_root, category, image))



    for _ in range(150):

        images = os.listdir(train_root + category)

        idx = int(nd.random.randint(0, len(images)).asscalar())

        image = images[idx]

        os.rename(os.path.join(train_root, category, image), os.path.join(test_root, category, image))
train_counts = []

for category in categories:

    train_counts.append(len(os.listdir(train_root + category)))

plt.figure(figsize = (8,8))

plt.bar(categories, train_counts)

plt.title('Training images in each category')

plt.xlabel('Categories')

plt.ylabel('Counts')

plt.show()
val_counts = []

for category in categories:

    val_counts.append(len(os.listdir(val_root + category)))

plt.figure(figsize = (8,8))

plt.bar(categories, val_counts)

plt.title('Validation images in each category')

plt.xlabel('Categories')

plt.ylabel('Counts')

plt.show()
test_counts = []

for category in categories:

    test_counts.append(len(os.listdir(test_root + category)))

plt.figure(figsize = (8,8))

plt.bar(categories, test_counts)

plt.title('Testing images in each category')

plt.xlabel('Categories')

plt.ylabel('Counts')

plt.show()
train_transform = vision.transforms.Compose([vision.transforms.RandomSaturation(saturation = 0.1),

                                vision.transforms.RandomLighting(alpha = 0.2),

                                vision.transforms.RandomHue(hue = 0.1),

                                vision.transforms.RandomFlipLeftRight(),

                                vision.transforms.RandomContrast(contrast = 0.2),

                                vision.transforms.RandomColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),

                                vision.transforms.Resize(128),

                                vision.transforms.ToTensor()])



transform = vision.transforms.Compose([vision.transforms.Resize(128),

                                       vision.transforms.ToTensor()])



batch_size = 64



train_data = gluon.data.DataLoader(

                vision.ImageFolderDataset(root = train_root, flag = 1).transform_first(train_transform),

                batch_size = batch_size, shuffle = True)



val_data = gluon.data.DataLoader(

                vision.ImageFolderDataset(root = val_root, flag = 1).transform_first(transform),

                batch_size = batch_size, shuffle = False)



test_data = gluon.data.DataLoader(

                vision.ImageFolderDataset(root = test_root, flag = 1).transform_first(transform),

                batch_size = batch_size, shuffle = False)



print(f"{len(train_data)} batches in training data")

print(f"{len(val_data)} batches in validation data")

print(f"{len(test_data)} batches in testing data")
for features, labels in train_data:

    break



print(f"features.shape: {features.shape}")

print(f"labels.shape: {labels.shape}")



print(f"features.max(): {features.max().asscalar()}")

print(f"features.min(): {features.min().asscalar()}")
plt.figure(figsize = (10, 12))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.imshow(features[i].transpose((1, 2, 0)).asnumpy())

    plt.title(categories[int(labels[i].asscalar())].title())

    plt.axis("off")

plt.show()
model = gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained = True, ctx = ctx)



with model.name_scope():

    model.output.add(nn.Dropout(0.5))

    model.output.add(nn.Dense(len(categories)))

model.output.initialize(mx.init.Xavier(), ctx = ctx)



print(model)
model.summary(features.as_in_context(ctx))
mx.viz.plot_network(model(mx.sym.var(name = 'data')), node_attrs={"fixedsize":"false"})
model.hybridize()
objective = gluon.loss.SoftmaxCrossEntropyLoss()

optimizer = mx.optimizer.Adam(learning_rate = 0.0005)

trainer = gluon.Trainer(model.collect_params(), optimizer)

metric = mx.metric.Accuracy()
epochs = 10

batches = len(train_data)



train_losses = []

train_accs = []

val_losses = []

val_accs = []



best_val = 0.0



for epoch in range(epochs):

    metric.reset()

    cum_loss = 0.0

    for features, labels in tqdm(train_data, desc = f'Epoch: {epoch + 1} Completed', ncols = 800):

        features = features.as_in_context(ctx)

        labels = labels.as_in_context(ctx)



        with autograd.record():

            outputs = model(features)

            loss = objective(outputs, labels)

        loss.backward()

        trainer.step(batch_size)



        cum_loss += loss.mean()

        metric.update(labels, outputs)



    train_loss = cum_loss.asscalar()/batches

    train_acc = metric.get()[1]



    train_losses.append(train_loss)

    train_accs.append(train_acc)



    metric.reset()

    cum_loss = 0.0

    for features, labels in test_data:

        features = features.as_in_context(ctx)

        labels = labels.as_in_context(ctx)

        outputs = model(features)

        metric.update(labels, outputs)

        cum_loss += objective(outputs, labels).mean()



    val_loss = cum_loss.asscalar()/batches

    val_acc = metric.get()[1]



    val_losses.append(val_loss)

    val_accs.append(val_acc)

    

    print(f'Training Loss:\t {train_loss:.5f} | Training Accuracy:   {train_acc:.5f}')

    print(f'Validation Loss: {val_loss:.5f} | Validation Accuracy: {val_acc:.5f}')

    if val_acc > best_val:

        print('Saving model for best validation accuracy')

        model.save_parameters('model.params')        

        best_val = val_acc
plt.figure(figsize = (10, 5))

plt.plot(train_accs, label = 'Training Accuracy')

plt.plot(val_accs, label = 'Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.show()
plt.figure(figsize = (10, 5))

plt.plot(train_losses, label = 'Training Loss')

plt.plot(val_losses, label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model.load_parameters('model.params')

metric.reset()

for features, labels in test_data:

    features = features.as_in_context(ctx)

    labels = labels.as_in_context(ctx)

    outputs = model(features)

    metric.update(labels, outputs)

print(f'Testing Accuracy: {metric.get()[1]}')