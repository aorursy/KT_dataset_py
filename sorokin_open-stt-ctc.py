!pip install torch-baidu-ctc
import time

import torch

from torch_baidu_ctc import ctc_loss

from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_

from torch.nn.functional import log_softmax

from torch.optim import Adam

from torch.optim.lr_scheduler import StepLR

from open_stt_utils import Labels, AcousticModel, AudioDataset, BucketingSampler, AverageMeter, collate_fn_ctc, unpad, uncat, cer, wer
labels = Labels()

print(len(labels))
model = AcousticModel(40, 512, 256, len(labels), n_layers=3, dropout=0.4)

model.cuda()
train = [

    ['open-stt-public-youtube1120-hq', 'data.csv']

]



test = [

    ['open-stt-val', 'asr_calls_2_val.csv'],

    ['open-stt-val', 'buriy_audiobooks_2_val.csv'],

    ['open-stt-val', 'public_youtube700_val.csv']

]



train = AudioDataset(train, labels)

test = AudioDataset(test, labels)



train.filter_by_conv(model.conv)

train.filter_by_length(400)



test.filter_by_conv(model.conv)

test.filter_by_length(1000)



sampler = BucketingSampler(train, 32)



train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_ctc, batch_sampler=sampler)

test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn_ctc, batch_size=32)
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=900, gamma=0.99)
for epoch in range(15):

    

    start = time.time()



    sampler.shuffle(epoch)



    model.train()



    grd_train = AverageMeter('gradient')

    err_train = AverageMeter('train')

    err_valid = AverageMeter('valid')

    cer_valid = AverageMeter('cer')

    wer_valid = AverageMeter('wer')



    for xs, ys, xn, yn in train:



        optimizer.zero_grad()



        xs, xn = model(xs.cuda(non_blocking=True), xn)

        xs = log_softmax(xs, dim=-1)



        loss = ctc_loss(xs, ys, xn, yn, average_frames=False, reduction="mean")

        loss.backward()



        grad_norm = clip_grad_norm_(model.parameters(), 150)



        optimizer.step()

        scheduler.step()



        err_train.update(loss.item())

        grd_train.update(grad_norm)



    model.eval()



    with torch.no_grad():

        for xs, ys, xn, yn in test:



            xs, xn = model(xs.cuda(non_blocking=True), xn)

            xs = log_softmax(xs, dim=-1)



            loss = ctc_loss(xs, ys, xn, yn, average_frames=False, reduction="mean")

            err_valid.update(loss.item())

            

            xs = xs.transpose(0, 1).argmax(2)



            hypothesis = unpad(xs, xn, labels, remove_repetitions=True)

            references = uncat(ys, yn, labels)



            for h, r in zip(hypothesis, references):

                cer_valid.update(cer(h, r))

                wer_valid.update(wer(h, r))

    

    minutes = (time.time() - start) // 60

    

    with open('am.log', 'a') as log:

        log.write('epoch %d lr %.6f %s %s %s %s %s time %d\n' % (epoch + 1, scheduler.get_lr()[0], grd_train, err_train, err_valid, cer_valid, wer_valid, minutes))

    torch.save(model.state_dict(), 'am.bin')