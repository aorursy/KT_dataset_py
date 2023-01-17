!pip install warp-rnnt
import time

import torch

from warp_rnnt import rnnt_loss

from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_

from torch.nn.functional import log_softmax

from torch.optim import Adam

from torch.optim.lr_scheduler import StepLR

from open_stt_utils import Labels, Transducer, AudioDataset, BucketingSampler, AverageMeter, collate_fn_rnnt, unpad, uncat, cer, wer
labels = Labels()

print(len(labels))
model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.3,

                   am_checkpoint='/kaggle/input/open-stt-ctc/am.bin',

                   lm_checkpoint='/kaggle/input/open-stt-lm/lm.bin')

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



train.filter_by_conv(model.encoder.conv)

train.filter_by_length(500)



test.filter_by_conv(model.encoder.conv)

test.filter_by_length(1000)



sampler = BucketingSampler(train, 32)



train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_sampler=sampler)

test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_size=32)
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=900, gamma=0.99)
for epoch in range(8):

    

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

        

        xs = xs.cuda(non_blocking=True)

        ys = ys.cuda(non_blocking=True)

        xn = xn.cuda(non_blocking=True)

        yn = yn.cuda(non_blocking=True)



        zs, xs, xn = model(xs, ys, xn, yn)

        

        ys = ys.t().contiguous()



        loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

        loss.backward()



        grad_norm = clip_grad_norm_(model.parameters(), 50)



        optimizer.step()

        scheduler.step()



        err_train.update(loss.item())

        grd_train.update(grad_norm)



    model.eval()



    with torch.no_grad():

        for xs, ys, xn, yn in test:



            xs = xs.cuda(non_blocking=True)

            ys = ys.cuda(non_blocking=True)

            xn = xn.cuda(non_blocking=True)

            yn = yn.cuda(non_blocking=True)



            zs, xs, xn = model(xs, ys, xn, yn)



            ys = ys.t().contiguous()



            loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")

            err_valid.update(loss.item())

            

            xs = model.greedy_decode(xs)



            hypothesis = unpad(xs, xn, labels)

            references = unpad(ys, yn, labels)



            for h, r in zip(hypothesis, references):

                cer_valid.update(cer(h, r))

                wer_valid.update(wer(h, r))

    

    minutes = (time.time() - start) // 60

    

    with open('asr.log', 'a') as log:

        log.write('epoch %d lr %.6f %s %s %s %s %s time %d\n' % (epoch + 1, scheduler.get_lr()[0], grd_train, err_train, err_valid, cer_valid, wer_valid, minutes))

    torch.save(model.state_dict(), 'asr.bin')