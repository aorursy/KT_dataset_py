import time

import torch

import torch.nn as nn

from torch.optim import Adam

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader

from open_stt_utils import Labels, TextDataset, LanguageModel, AverageMeter, detach
labels = Labels()

num_labels = len(labels)

print(num_labels)
bptt = 8

batch_size = 64



train = [

    ['open-stt-text', 'public_youtube1120_hq.csv'],

    ['open-stt-text', 'public_youtube1120.csv'],

    ['open-stt-text', 'public_youtube700.csv']

]



test = [

    ['open-stt-text', 'asr_calls_2_val.csv'],

    ['open-stt-text', 'buriy_audiobooks_2_val.csv'],

    ['open-stt-text', 'public_youtube700_val.csv']

]



train = TextDataset(train, labels, batch_size)

test = TextDataset(test, labels, batch_size)



print(len(train), len(test))



test = DataLoader(test, pin_memory=True, num_workers=2, batch_size=bptt, drop_last=True)
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)



model = LanguageModel(128, 512, 256, num_labels, n_layers=3, dropout=0.3)

model.cuda()



criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=9000, gamma=0.99)
for epoch in range(10):

    

    start = time.time()



    model.train()

    hidden = model.step_init(batch_size)



    grd_train = AverageMeter('gradient')

    err_train = AverageMeter('train')

    err_valid = AverageMeter('valid')



    train.shuffle(epoch)



    loader = DataLoader(train, pin_memory=True, num_workers=2, batch_size=bptt, drop_last=True)



    for inputs, targets in loader:

        inputs = inputs.cuda(non_blocking=True)

        targets = targets.cuda(non_blocking=True)



        # Starting each batch, we detach the hidden state from how it was previously produced.

        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        hidden = detach(hidden)



        optimizer.zero_grad()



        output, hidden = model.step_forward(inputs, hidden)



        loss = criterion(output.view(-1, num_labels), targets.view(-1))

        loss.backward()



        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)



        optimizer.step()

        scheduler.step()



        err_train.update(loss.item())

        grd_train.update(grad_norm)



    model.eval()

    hidden = model.step_init(batch_size)



    with torch.no_grad():

        for inputs, targets in test:

            inputs = inputs.cuda(non_blocking=True)

            targets = targets.cuda(non_blocking=True)

            output, hidden = model.step_forward(inputs, hidden)

            loss = criterion(output.view(-1, num_labels), targets.view(-1))

            err_valid.update(loss.item())

            

    minutes = (time.time() - start) // 60

    

    with open('lm.log', 'a') as log:

        log.write('epoch %d lr %.6f %s %s %s time %d\n' % (epoch + 1, scheduler.get_lr()[0], grd_train, err_train, err_valid, minutes))



    torch.save(model.state_dict(), 'lm.bin')