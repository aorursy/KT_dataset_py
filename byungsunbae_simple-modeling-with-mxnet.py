from matplotlib import pyplot as plt # plotting image

import seaborn as sns
!pip install --upgrade mxnet-cu92mkl
import numpy as np # array processing



# using mxnet

import mxnet as mx

from mxnet import nd, gluon, autograd

from mxnet.io import NDArrayIter

from mxnet.gluon import nn



import os

from glob import glob

# from shutil import copyfile



import pandas as pd
def plot_mx_array(array):

    assert array.shape[2] == 3, "RGB Channel should be last"

    plt.imshow((array.clip(0, 255)/255).asnumpy())
def augmentor(data, label):

    # Normalizing pixel value : 0 ~ 1

    data = data.astype('float32') / 255.

    

    # Augmentation list

    aug_list = [

        mx.image.ForceResizeAug(size=(128, 128)), # Resizing

        mx.image.HorizontalFlipAug(p=0.5), # Horizontal Flip

        mx.image.BrightnessJitterAug(brightness=0.2), # Jittering brightness

        mx.image.HueJitterAug(hue=0.2), # Jittering Hue

        mx.image.ContrastJitterAug(contrast=0.2) # Jittering Contrast 

    ]

    

    # Random Order Augmentation operation

    augs = mx.image.RandomOrderAug(aug_list)

    

    # apply to data

    data = augs(data)

    

    if np.random.rand() > 0.5:

        data = data.swapaxes(0, 1)

        

    data = data.swapaxes(0, 2)

    return data, label
def testset_augmentor(data, label):

    data = data.astype('float32') / 255.

    aug = mx.image.ForceResizeAug(size=(128, 128)) # For predictions

    data = aug(data)

    data = data.swapaxes(0, 2)

    return data, label
training_dataset = mx.gluon.data.vision.ImageFolderDataset("../input/split-data/images/train", transform=augmentor)

test_dataset = mx.gluon.data.vision.ImageFolderDataset("../input/split-data/images/test", transform=testset_augmentor)
sample = training_dataset[0]

sample_data = sample[0]
print("%d is Uninfected Cell" % sample[1])

plot_mx_array(sample_data.swapaxes(0, 2) * 255)
sample = training_dataset[16900]

sample_data = sample[0]
print("%d is Infected Cell" % sample[1])

plot_mx_array(sample_data.swapaxes(0, 2) * 255)
batch_size = 100

train_loader = mx.gluon.data.DataLoader(training_dataset, shuffle=True, batch_size=batch_size)

test_loader = mx.gluon.data.DataLoader(test_dataset, shuffle=False, batch_size=10)
lenet = nn.HybridSequential(prefix='LeNet_')

with lenet.name_scope():

    lenet.add(

        nn.Conv2D(channels=8, kernel_size=(5, 5), activation='relu'),

        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        nn.Conv2D(channels=8, kernel_size=(5, 5), activation='relu'),

        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        nn.Conv2D(channels=16, kernel_size=(3, 3), activation='relu'),

        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        nn.Conv2D(channels=16, kernel_size=(2, 2), activation='relu'),

        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        nn.Conv2D(channels=32, kernel_size=(2, 2), activation='relu'),

        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        nn.Flatten(),

        nn.Dense(128, activation='relu'),

        nn.Dense(2, activation=None),

    )
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

lenet.initialize(mx.init.Xavier(), ctx=ctx)
lenet.summary(nd.zeros((1, 3, 128, 128), ctx=ctx))
trainer = gluon.Trainer(

    params=lenet.collect_params(),

    optimizer='adam',

    optimizer_params={'learning_rate': 0.001},

)
metric = mx.metric.Accuracy()

loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
num_epochs = 20



for epoch in range(num_epochs):

    for inputs, labels in train_loader:

        # Possibly copy inputs and labels to the GPU

        inputs = inputs.as_in_context(ctx)

        labels = labels.as_in_context(ctx)



        # The forward pass and the loss computation need to be wrapped

        # in a `record()` scope to make sure the computational graph is

        # recorded in order to automatically compute the gradients

        # during the backward pass.

        with autograd.record():

            outputs = lenet(inputs)

            loss = loss_function(outputs, labels)



        # Compute gradients by backpropagation and update the evaluation

        # metric

        loss.backward()

        metric.update(labels, outputs)



        # Update the parameters by stepping the trainer; the batch size

        # is required to normalize the gradients by `1 / batch_size`.

        trainer.step(batch_size=inputs.shape[0])



    # Print the evaluation metric and reset it for the next epoch

    name, acc = metric.get()

    print('After epoch {}: {} = {}'.format(epoch + 1, name, acc))

    metric.reset()

#     if epoch % 20 == 0 or epoch == num_epochs - 1:

    metric_test = mx.metric.Accuracy()

    for inputs, labels in test_loader:

        # Possibly copy inputs and labels to the GPU

        inputs = inputs.as_in_context(ctx)

        labels = labels.as_in_context(ctx)

        metric_test.update(labels, lenet(inputs))

    print('\tTest: {} = {}'.format(*metric_test.get()))

    
from sklearn import metrics
train_loader_pred = mx.gluon.data.DataLoader(training_dataset, shuffle=False, batch_size=20)

test_loader_pred = mx.gluon.data.DataLoader(test_dataset, shuffle=False, batch_size=20)
scores_train = []

train_label = []

train_imgs = []

for inputs, labels in train_loader_pred:

    # Possibly copy inputs and labels to the GPU

    train_imgs.append(inputs)

    train_label.append(labels)

    inputs = inputs.as_in_context(ctx)

    labels = labels.as_in_context(ctx)

    outputs = lenet(inputs)

    outputs = outputs.as_in_context(mx.cpu(0)).asnumpy()

    scores_train.append(outputs)

    

train_imgs = nd.concat(*train_imgs, dim = 0)

train_label = nd.concat(*train_label, dim = 0)
scores_test = []

test_label = []

test_imgs = []

for inputs, labels in test_loader_pred:

    # Possibly copy inputs and labels to the GPU

    test_imgs.append(inputs)

    test_label.append(labels)

    inputs = inputs.as_in_context(ctx)

    labels = labels.as_in_context(ctx)

    outputs = lenet(inputs)

    outputs = outputs.as_in_context(mx.cpu(0)).asnumpy()

    scores_test.append(outputs)



test_imgs = nd.concat(*test_imgs, dim = 0)

test_label = nd.concat(*test_label, dim = 0)
def compute_prob(x):

    return np.exp(x) / np.exp(x).sum()
train_scores = np.vstack([np.apply_along_axis(compute_prob, 1, arr) for arr in scores_train])

train_scores_inf_prob = train_scores[:, 1]



test_scores = np.vstack([np.apply_along_axis(compute_prob, 1, arr) for arr in scores_test])

test_scores_inf_prob = test_scores[:, 1]
fpr_train, tpr_train, _ = metrics.roc_curve(train_label.asnumpy(), train_scores_inf_prob)

fpr_test, tpr_test, _ = metrics.roc_curve(test_label.asnumpy(), test_scores_inf_prob)
# plotting

plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_train, tpr_train, label='Train')

plt.plot(fpr_test, tpr_test, label='Test')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='Set')

plt.show()
train_auc = metrics.auc(fpr_train, tpr_train)

test_auc = metrics.auc(fpr_test, tpr_test)
print("Train AUC : ", train_auc)

print("Test AUC : ", test_auc)
test_pred_dt = pd.DataFrame(data={"idx": [i for i in range(0, len(test_label))], "label": list(test_label.asnumpy()), "prob": list(test_scores_inf_prob)})
test_pred_dt = test_pred_dt.sort_values("prob")
test_pred_dt.reset_index(inplace=True, drop=True)
test_pred_dt.head()
test_pred_dt.tail()
plt.subplots(figsize=(10,10))

label_dict = {"0": "Uninfection", "1": "Infection"}

for label in [0, 1]:

    # Subset to the airline

    subset = test_pred_dt[test_pred_dt['label'] == label]

    

    # Draw the density plot

    sns.distplot(subset['prob'], hist = False, kde = True, 

                 kde_kws = {'shade': True, 'linewidth': 2, 'bw': 0.01},

                 label = label_dict[str(label)])
print("Number of Observation Per bin :", len(test_pred_dt) // 10)

print("Number of Observation of Last bin :", len(test_pred_dt) // 10 + len(test_pred_dt) % 10)
bins = []

for i in range(1, 11):

    if i == 10:

        bins.append(np.repeat(i, (len(test_pred_dt) // 10) + len(test_pred_dt) % 10))

    else:

        bins.append(np.repeat(i, len(test_pred_dt) // 10))        
bins_np = np.hstack(bins)

test_pred_dt["bins"] = bins_np
pd.concat([test_pred_dt.head(3), test_pred_dt.tail(3)]) # For printing
test_pred_dt["bins"] = test_pred_dt["bins"].astype('category')
df1 = test_pred_dt[test_pred_dt["label"] == 0].groupby(['bins']).size().reset_index(name='count')

df2 = test_pred_dt[test_pred_dt["label"] == 1].groupby(['bins']).size().reset_index(name='count')
by_bins = pd.merge(df1, df2, on="bins")

by_bins.rename(index=str, columns={"count_x": "Uninf_count", "count_y": "Inf_count"}, inplace=True)
by_bins["total_count"] = by_bins["Uninf_count"] + by_bins["Inf_count"]
by_bins["Uninf_cul_count"] = by_bins["Uninf_count"].cumsum()

by_bins["Inf_cul_count"] = by_bins["Inf_count"].cumsum()
by_bins["Uninf_ratio"] = by_bins["Uninf_count"] / by_bins["total_count"]

by_bins["Inf_ratio"] = by_bins["Inf_count"] / by_bins["total_count"]
by_bins["Uninf_cul_ratio"] = by_bins["Uninf_cul_count"]/by_bins["Uninf_count"].sum()

by_bins["Inf_cul_ratio"] = by_bins["Inf_cul_count"]/by_bins["Inf_count"].sum()

by_bins["Diff"] = by_bins["Uninf_cul_ratio"] - by_bins["Inf_cul_ratio"]
result = by_bins.loc[:, ["bins", "total_count", "Inf_count",  "Inf_cul_count",  "Inf_ratio", 

                         "Uninf_cul_ratio", "Inf_cul_ratio", "Diff"]]
result
plt.subplots(figsize=(10,10))

sns.lineplot(x="bins", y="Uninf_cul_ratio", data=result, color="darkblue", label="Uninfection", markers=True, dashes=True)

sns.lineplot(x="bins", y="Inf_cul_ratio", data=result, color="orange", label="Infection", markers=True, dashes=True)

plt.axvline(result.loc[result["Diff"] == result["Diff"].max(), "bins"][0], 0.05, 0.95, color="red")
Normal = int(len(test_pred_dt) * 0.30)

Ambiguous = int(len(test_pred_dt) * 0.45)

EndStage = len(test_pred_dt) - (Normal + Ambiguous)
print("Normal : %d\tAmbiguous : %d\tEndStage : %d" % (Normal, Ambiguous, EndStage))
test_pred_dt["Sugg_Grade"] = np.hstack([np.repeat("Normal", Normal), 

                                        np.repeat("Ambiguous", Ambiguous), 

                                        np.repeat("EndStage", EndStage)])
Normal_InfRatio = test_pred_dt.loc[(test_pred_dt["Sugg_Grade"] == "Normal"), "label"].mean()

Ambiguous_InfRatio = test_pred_dt.loc[(test_pred_dt["Sugg_Grade"] == "Ambiguous"), "label"].mean()

EndStage_InfRatio = test_pred_dt.loc[(test_pred_dt["Sugg_Grade"] == "EndStage"), "label"].mean()
print("Normal : %.5f\tAmbiguous : %.5f\tEndStage : %.5f" % (Normal_InfRatio, Ambiguous_InfRatio, EndStage_InfRatio))
cutoff_df = test_pred_dt.loc[:, ["prob", "Sugg_Grade", "label"]].groupby(['Sugg_Grade']).agg({"prob": ['min', 'max'], "Sugg_Grade": ['count']})

cutoff_df.columns = ["_".join(x) for x in cutoff_df.columns.ravel()]

cutoff_df.sort_values("prob_min", inplace=True)
cutoff_df