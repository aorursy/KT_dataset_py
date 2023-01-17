import mxnet as mx

from mxnet import nd, autograd, gluon

from mxnet.gluon import nn

from mxnet.gluon.data import vision, DataLoader

import pandas as pd

import os



ctx = mx.gpu()
root = '../../kaggle/input/applications-of-deep-learningwustl-spring-2020/'

os.listdir(root)
df = pd.read_csv(root + 'train.csv')

df
df.describe()
x = df.drop(["id", "glasses"], axis = 1).to_numpy()

x = nd.array(x, ctx = ctx)

# x = (x - x.min())/(x.max() - x.min())

print(x)

print(x.max().asscalar(), x.min().asscalar())
y = df.glasses.to_numpy()

y = nd.array(y, ctx = ctx)

y
val_frac = 0.1

train_x = x[:int(len(x) - (len(x) * val_frac))]

train_y = y[:int(len(y) - (len(y) * val_frac))]



val_x = x[int(len(x) - (len(x) * val_frac)):]

val_y = y[int(len(y) - (len(y) * val_frac)):]



print(train_x.shape)

print(train_y.shape)



print(val_x.shape)

print(val_y.shape)
batch_size = 32

train_data = mx.io.NDArrayIter(data = train_x, label = train_y, batch_size = batch_size, shuffle = True)

val_data = mx.io.NDArrayIter(data = val_x, label = val_y, batch_size = batch_size, shuffle = False)
model = nn.HybridSequential()

model.add(nn.Dense(1024))

model.add(nn.BatchNorm())

model.add(nn.LeakyReLU(0.2))

model.add(nn.Dropout(0.3))



model.add(nn.Dense(1024))

model.add(nn.BatchNorm())

model.add(nn.LeakyReLU(0.2))

model.add(nn.Dropout(0.3))



model.add(nn.Dense(512))

model.add(nn.BatchNorm())

model.add(nn.LeakyReLU(0.2))



model.add(nn.Dense(256))

model.add(nn.BatchNorm())

model.add(nn.LeakyReLU(0.2))



model.add(nn.Dense(2))



model.initialize(init = mx.init.Xavier(), ctx = ctx)

model.hybridize(static_alloc = True, static_shape = True)

model
def evaluate_validation_data():

    cum_loss = 0.0

    metric.reset()

    val_data.reset()

    for batch in val_data:

        features = batch.data[0]

        labels = batch.label[0]

        outputs = model(features)

        cum_loss += objective(outputs, labels).mean().asscalar()

        metric.update(labels, outputs)

    return cum_loss/val_batches, metric.get()[1]



def evaluate_training_data():

    cum_loss = 0.0

    train_data.reset()

    metric.reset()

    for batch in train_data:

        features = batch.data[0]

        labels = batch.label[0]

        outputs = model(features)

        cum_loss += objective(outputs, labels).mean().asscalar()

        metric.update(labels, outputs)

    return cum_loss/batches, metric.get()[1]
objective = gluon.loss.SoftmaxCrossEntropyLoss()

epochs = 200

batches = int(len(train_x)//batch_size)

val_batches = int(len(val_x)//batch_size)

max_update = int((epochs/2) * batches)

base_lr = 0.01

final_lr = 0.00000001

scheduler = mx.lr_scheduler.PolyScheduler(max_update = max_update, base_lr = base_lr, final_lr = final_lr)

optimizer = mx.optimizer.SGD(lr_scheduler = scheduler, momentum = 0.999)

trainer = gluon.Trainer(model.collect_params(), optimizer)
best_val = 0.0

best_loss = 1.0

metric = mx.metric.Accuracy()



for epoch in range(epochs):

    train_data.reset()

    for batch in train_data:

        features = batch.data[0]

        labels = batch.label[0]



        with autograd.record():

            outputs = model(features)

            loss = objective(outputs, labels)

        loss.backward()

        trainer.step(batch_size)

        

    train_loss, train_acc = evaluate_training_data()

    val_loss, val_acc = evaluate_validation_data()

    

    print(f'\nEpoch: ({epoch + 1}/{epochs})')

    print(f'Training Loss: {train_loss:.12f} | Training Accuracy: {train_acc:.5f}')

    print(f'Validation Loss: {val_loss:.12f} | Validation Accuracy: {val_acc:.5f}')

    

    if val_acc >= best_val and val_loss <= best_loss:

        best_val = val_acc

        best_loss = val_loss

        print('Best Validation Accuracy\n')

        print('WOWOWOWOW!! Saving model for best validation accuracy\n\n')

        model.export('Best Model')
best_loss, best_val
model = gluon.SymbolBlock.imports('Best Model-symbol.json', ['data'], 'Best Model-0000.params', ctx = ctx)

metric = mx.metric.Accuracy()

cum_loss = 0.0

metric.reset()

val_data.reset()

for batch in val_data:

    features = batch.data[0]

    labels = batch.label[0]

    outputs = model(features)

    cum_loss += objective(outputs, labels).mean().asscalar()

    metric.update(labels, outputs)

val_loss = cum_loss/val_batches

print('Validation loss:', val_loss)

print('Validation accuracy:', metric.get()[1])
test_df = pd.read_csv(root + 'test.csv')

test_x = test_df.drop(["id"], axis = 1).to_numpy()

test_x = nd.array(test_x, ctx = ctx)

test_x
predictions = model(test_x).argmax(axis = 1).asnumpy()

predictions
sample = pd.read_csv(root + 'sample.csv')

sample
solutions = pd.DataFrame({"id": test_df.id, 'glasses': predictions})

solutions
solutions.to_csv('submission.csv', index = False)