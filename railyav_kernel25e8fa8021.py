import torch

from torch import nn

import torch.nn.functional as f

import numpy as np

import os

import time

import imageio as im

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter





device = torch.device("cuda:0")

data_path = '/kaggle/input/omniglot/omniglot/'

train_path = os.path.join(data_path, 'images_background')

test_path = os.path.join(data_path, 'images_evaluation')

checkp_dir = './'



log_dir = './'

writer = SummaryWriter(log_dir=log_dir)

images_num_per_class = 20

large_b_size = 64

batch_size = 32

margin = 0.5

embedding_len = 128

lr = 0.001

epochs = 20

channels = 1

d = 1.2





class Model(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(Model, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1),

            # nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),

            # nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),

            # nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

            # nn.BatchNorm2d(16),

            nn.ReLU())

        self.fc = nn.Linear(16 * 13 * 13, out_ch)



    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        sh = np.prod(np.array(x.shape)[1:])

        x = x.view(-1, sh)

        x = self.fc(x)

        x = torch.nn.functional.normalize(x, p=2)

        return x





def load_images(path):

    # dataset = dict()

    x = []

    for alph_folder in os.listdir(path):

        alph_path = os.path.join(path, alph_folder)

        for img_folder in os.listdir(alph_path):

            images = []

            img_path = os.path.join(alph_path, img_folder)

            for img in os.listdir(img_path):

                image = im.imread(os.path.join(img_path, img))

                image = (image / 127.5) - 1.0

                # image = image / 255

                # dataset[str(img[:7])] = image

                images.append(image)

            x.append(np.stack(images))

    x = np.stack(x)

    return x





def generate_batch(data, large_batch_size):

    n_classes, n_examples, w, h = data.shape



    triplets = [np.zeros((large_batch_size, h, w, 1)) for _ in range(3)]

    anchors_classes = np.random.choice(n_classes, size=(large_batch_size,), replace=False)



    for j in range(large_batch_size):

        anchor_class = anchors_classes[j]

        negative_class = (anchor_class + np.random.randint(1, n_classes)) % n_classes



        idxs = np.random.choice(n_examples, size=(3,), replace=False)

        anchor_image = data[anchor_class, idxs[0]]

        positive_image = data[anchor_class, idxs[1]]

        negative_image = data[negative_class, idxs[2]]



        triplets[0][j, :, :, :] = anchor_image.reshape(w, h, 1)

        triplets[1][j, :, :, :] = positive_image.reshape(w, h, 1)

        triplets[2][j, :, :, :] = negative_image.reshape(w, h, 1)

    triplets = [torch.from_numpy(triplets[k].transpose((0, 3, 1, 2))).to(device, dtype=torch.float) for k

                in range(3)]

    return triplets





def generate_pairs(data, b_size):

    n_classes, n_examples, w, h = data.shape

    same_pair = [np.zeros((b_size, h, w, 1)) for _ in range(2)]

    dif_pair = [np.zeros((b_size, h, w, 1)) for _ in range(2)]



    anchors_classes = np.random.choice(n_classes, size=(b_size,), replace=False)

    for j in range(b_size):

        same_class = anchors_classes[j]



        dif1_class = (same_class + np.random.randint(1, n_classes)) % n_classes

        dif2_class = (dif1_class + np.random.randint(1, n_classes)) % n_classes



        idxs = np.random.choice(n_examples, size=(4,), replace=False)

        same1_image = data[same_class, idxs[0]]

        same2_image = data[same_class, idxs[1]]



        dif1_image = data[dif1_class, idxs[2]]

        dif2_image = data[dif2_class, idxs[3]]



        same_pair[0][j, :, :, :] = same1_image.reshape(w, h, 1)

        same_pair[1][j, :, :, :] = same2_image.reshape(w, h, 1)



        dif_pair[0][j, :, :, :] = dif1_image.reshape(w, h, 1)

        dif_pair[1][j, :, :, :] = dif2_image.reshape(w, h, 1)



    same_pair = [torch.from_numpy(same_pair[k].transpose((0, 3, 1, 2))).to(device, dtype=torch.float) for k

                 in range(2)]

    dif_pair = [torch.from_numpy(dif_pair[k].transpose((0, 3, 1, 2))).to(device, dtype=torch.float) for k

                in range(2)]

    return same_pair, dif_pair





def val_accuracy(pos_dist, neg_dist):

    # mask_ = pos_dist + margin < neg_dist

    # mask_ = np.logical_and(pos_dist <= d, neg_dist > d)

    tp, tn = pos_dist <= d, neg_dist > d

    tp = [k for k, el in enumerate(np.array(tp)) if el]

    tn = [k for k, el in enumerate(np.array(tn)) if el]

    num_correctly_classified = len(tp) + len(tn)

    num_all_pairs = len(pos_dist) + len(neg_dist)

    accuracy = num_correctly_classified / num_all_pairs



    # validation rate VAL(d), false accept rate FAR(d)

    true_accepts = pos_dist <= d

    false_accepts = neg_dist <= d



    val = len([k for k, el in enumerate(np.array(true_accepts)) if el]) / len(pos_dist)

    far = len([k for k, el in enumerate(np.array(false_accepts)) if el]) / len(neg_dist)

    return accuracy, val, far





def validation(x, batches_num_per_epoch):

    losses, dist_p, dist_n, acc_lst, val_lst, far_lst = [], [], [], [], [], []

    for _ in range(batches_num_per_epoch):

        triplets = generate_batch(x, batch_size)

        f_anchor_v, f_pos_v, f_neg_v = model(triplets[0]), model(triplets[1]), model(triplets[2])

        a_pos_d = torch.sum((f_anchor_v - f_pos_v) ** 2, dim=1)

        a_neg_d = torch.sum((f_anchor_v - f_neg_v) ** 2, dim=1)

        dist_v = a_pos_d - a_neg_d + margin

        loss_val = torch.mean(torch.max(dist_v, torch.zeros(list(dist_v.shape), device=device, dtype=torch.float)), 0)



        same_pair, dif_pair = generate_pairs(x, batch_size)

        f_same1, f_same2 = model(same_pair[0]), model(same_pair[1])

        f_dif1, f_dif2 = model(dif_pair[0]), model(dif_pair[1])

        a_p_d_pairs = torch.sum((f_same1 - f_same2) ** 2, dim=1)

        a_n_d_pairs = torch.sum((f_dif1 - f_dif2) ** 2, dim=1)

        

        losses.append(loss_val.cpu().detach().numpy())

        dist_p.append(np.mean(a_pos_d.cpu().detach().numpy()))

        dist_n.append(np.mean(a_neg_d.cpu().detach().numpy()))

        acc_v, val_v, far_v = val_accuracy(a_p_d_pairs.cpu().detach().numpy(), a_n_d_pairs.cpu().detach().numpy())

        acc_lst.append(acc_v)

        val_lst.append(val_v)

        far_lst.append(far_v)

    return losses, dist_p, dist_n, acc_lst, val_lst, far_lst





# def get_all_possible_pairs(data):

# #     all_pos_same_pairs = 0

# #     all_pos_dif_pairs = 0

# #     return all_pos_pairs





print('Loading training set')

x_train = load_images(train_path)

num_tr_samples = x_train.shape[0] * x_train.shape[1]

num_tr_classes = x_train.shape[0]

print('Loading test set')

x_test = load_images(test_path)

num_val_classes = x_test.shape[0] // 2

x_val = x_test[:num_val_classes]

x_test = x_test[num_val_classes:]

num_test_samples = x_test.shape[0] * x_test.shape[1]

num_val_samples = x_val.shape[0] * x_val.shape[1]



print('training set shape: ', x_train.shape)

print(f'training samples: {num_tr_samples}')

print('test set shape: ', x_test.shape)

print(f'test samples: {num_test_samples}')

print('validation set shape: ', x_val.shape)

print(f'validation samples: {num_val_samples}')



# all_possible_pairs = get_all_possible_pairs(x_train)



model = Model(channels, embedding_len).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

start_epoch = 0

# try:

#     checkpoint = torch.load(os.path.join(checkp_dir, 'ckpt.pth'))

#     model.load_state_dict(checkpoint['model'])

#     best_acc = checkpoint['acc']

#     start_epoch = checkpoint['epoch']

# except FileNotFoundError:

#     print('ckpt is not found')



losses_tr, dist_p_tr, dist_n_tr, tr_acc, tr_Val, tr_Far, iters = [], [], [], [], [], [], []

losses_val, iter_val, losses_tr_lst = [], [], []

v_acc = 0

b_num_per_epoch = num_tr_samples // batch_size

b_num_per_epoch_val = num_val_samples // batch_size

st_time = time.time()



for e in range(start_epoch, start_epoch + epochs):

    print(f'epoch: {e + 1}')

    for it in range(b_num_per_epoch):

        triplets_tr = generate_batch(x_train, large_b_size)



        f_a = model(triplets_tr[0])

        f_p = model(triplets_tr[1])

        f_n = model(triplets_tr[2])



        a_p_dist = torch.sum((f_a - f_p) ** 2, dim=1)

        a_n_dist = torch.sum((f_a - f_n) ** 2, dim=1)

        a_p_dist_ = a_p_dist.cpu().detach().numpy()

        a_n_dist_ = a_n_dist.cpu().detach().numpy()



        mask = np.logical_and((a_p_dist_ < a_n_dist_),

                              (a_n_dist_ < (a_p_dist_ + margin)))

        semi_hard_idxs = np.argsort(mask)[-batch_size:]



        dist_pos_tr = a_p_dist[semi_hard_idxs]

        dist_neg_tr = a_n_dist[semi_hard_idxs]

        dist = dist_pos_tr - dist_neg_tr + margin

        loss = torch.max(dist, torch.zeros(list(dist.shape), device=device, dtype=torch.float))

        loss = torch.mean(loss, 0)

        global_step = it + e * b_num_per_epoch + 1

        writer.add_scalar('Loss_train', loss, global_step)



        losses_tr.append(loss.cpu().detach().numpy())

        # iters.append(it + e * b_num_per_epoch)

        dist_p_tr.append(np.mean(dist_pos_tr.cpu().detach().numpy()))

        dist_n_tr.append(np.mean(dist_neg_tr.cpu().detach().numpy()))

        acc, val, far = val_accuracy(dist_pos_tr.cpu().detach().numpy(), dist_neg_tr.cpu().detach().numpy())

        writer.add_scalar('Accuracy_train', acc, global_step)

        tr_acc.append(acc)

        tr_Val.append(val)

        tr_Far.append(far)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if it % 50 == 0:

            print(f'Epoch [{e + 1}/{epochs}], Step [{it + 1}/{b_num_per_epoch}], Loss: {loss.item()}')

            print('TRAINING:')

            print(f'tr mean dist_p: {np.mean(dist_p_tr)}')

            print(f'tr mean dist_n: {np.mean(dist_n_tr)}\n')

            print(f'tr_loss_mean: {np.mean(losses_tr)}')

            print(f'tr_accuracy: {np.mean(tr_acc)}\n')

            print(f'tr_Val: {np.mean(tr_Val)}')

            print(f'tr_Far: {np.mean(tr_Far)}\n')



            print('VALIDATION:')

            losses_v, dist_p_v, dist_n_v, v_acc, v_Val, v_Far = validation(x_val, b_num_per_epoch_val)

            print(f'val mean dist_p: {np.mean(dist_p_v)}')

            print(f'val mean dist_n: {np.mean(dist_n_v)}\n')

            print(f'validation loss: {np.mean(losses_v)}')

            print(f'val acc: {np.mean(v_acc)}\n')

            print(f'val Val: {np.mean(v_Val)}')

            print(f'val Far: {np.mean(v_Far)}\n')

            losses_val.append(np.mean(losses_v))

            iter_val.append(it + e * b_num_per_epoch)

            losses_tr_lst.append(np.mean(losses_tr))



            writer.add_scalar('Loss_validation', np.mean(losses_v), global_step)

            writer.add_scalar('Accuracy_validation', np.mean(v_acc), global_step)



            losses_tr.clear()

            dist_p_tr.clear()

            dist_n_tr.clear()



    print('Saving..')

    state = {

        'model': model.state_dict(),

        'acc': v_acc,

        'epoch': e,

    }

    torch.save(state, (os.path.join(checkp_dir, 'ckpt.pth')))



    print(f'epoch tr_accuracy: {np.mean(tr_acc)}\n')

    tr_acc.clear()



fin_time = time.time()

b_num_per_epoch_test = num_test_samples // batch_size

losses_t, dist_p_t, dist_n_t, t_acc, t_Val, t_Far = validation(x_test, b_num_per_epoch_test)



print('TESTING:')

print(f'max_pos_dist: {np.max(dist_p_t)}')

print(f'min_neg_dist: {np.min(dist_n_t)}\n')

print(f'test loss: {np.mean(losses_t)}')

print(f'test acc: {np.mean(t_acc)}\n')

print(f'test Val: {np.mean(t_Val)}')

print(f'test Far: {np.mean(t_Far)}\n')

print(f'tr_time: {(fin_time - st_time) / 60} min')



# plt.subplot(1, 2, 1)

# plt.plot(iter_val, losses_tr_lst)

# plt.title('train loss')

# plt.subplot(1, 2, 2)

# plt.plot(iter_val, losses_val)

# plt.title('valid loss')

# plt.show()