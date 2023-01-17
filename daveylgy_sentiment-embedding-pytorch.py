import torch

from torchtext import data



SEED = 1234



torch.manual_seed(SEED) #为CPU设置随机种子

torch.cuda.manual_seed(SEED)#为GPU设置随机种子



#在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。

torch.backends.cudnn.deterministic = True  



#用来定义字段的处理方法（文本字段，标签字段）

TEXT = data.Field(tokenize='spacy')#torchtext.data.Field : 

LABEL = data.LabelField(dtype=torch.float)
from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')

print(f'Number of testing examples: {len(test_data)}')
import random

train_data, valid_data = train_data.split(random_state=random.seed(SEED))
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d",

                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data), 

    batch_size=BATCH_SIZE,

    device=device)
import torch.nn as nn

import torch.nn.functional as F



class WordAVGModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

        

    def forward(self, text):

        embedded = self.embedding(text) # [sent_len, batch _size, emb_size]

        embedded = embedded.permute(1, 0, 2) # [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]

        return self.fc(pooled)
INPUT_DIM = len(TEXT.vocab) #词个数

EMBEDDING_DIM = 100 #词嵌入维度

OUTPUT_DIM = 1 #输出维度

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] #pad索引



model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
import torch.optim as optim



optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)

criterion = criterion.to(device)
def binary_accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division 

    acc = correct.sum()/len(correct)

    return acc
def train(model, iterator, optimizer, criterion):

    

    

    epoch_loss = 0

    epoch_acc = 0

    total_len = 0

    model.train() #model.train()代表了训练模式

    #这步一定要加，是为了区分model训练和测试的模式的。

    #有时候训练时会用到dropout、归一化等方法，但是测试的时候不能用dropout等方法。

    

    

    

    for batch in iterator: #iterator为train_iterator

        optimizer.zero_grad() #加这步防止梯度叠加

        

        predictions = model(batch.text).squeeze(1)

        #batch.text 就是上面forward函数的参数text

        #压缩维度，不然跟batch.label维度对不上

        

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        

        

        loss.backward() #反向传播

        optimizer.step() #梯度下降

        

        epoch_loss += loss.item() * len(batch.label)

        #loss.item()已经本身除以了len(batch.label)

        #所以得再乘一次，得到一个batch的损失，累加得到所有样本损失。

        

        epoch_acc += acc.item() * len(batch.label)

        #（acc.item()：一个batch的正确率） *batch数 = 正确数

        #train_iterator所有batch的正确数累加。

        

        total_len += len(batch.label)

        #计算train_iterator所有样本的数量，不出意外应该是17500

        

    return epoch_loss / total_len, epoch_acc / total_len

    #epoch_loss / total_len ：train_iterator所有batch的损失

    #epoch_acc / total_len ：train_iterator所有batch的正确率
def evaluate(model, iterator, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    model.eval()

    

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
#import time

N_EPOCHS = 10



best_valid_loss = float('inf') #无穷大



for epoch in range(N_EPOCHS):



    #start_time = time.time()

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    

    #end_time = time.time()



    #epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss: #只要模型效果变好，就存模型

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'wordavg-model.pt')

    

    #print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')