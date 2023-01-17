



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import metrics



import os



# Any results you write to the current directory are saved as output.
import torch



# import torch_xla.distributed.xla_multiprocessing as xmp



import torch.nn as nn



import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule



# import torch_xla

# import torch_xla.debug.metrics as met

# import torch_xla.distributed.data_parallel as dp

# import torch_xla.distributed.parallel_loader as pl

# import torch_xla.utils.utils as xu

# import torch_xla.core.xla_model as xm

# import torch_xla.distributed.xla_multiprocessing as xmp

# import torch_xla.test.test_utils as test_utils

    

class JigsawDataset:

    def __init__(self, comment_text, tokenizer, max_length):

        self.comment_text = comment_text

        self.tokenizer = tokenizer

        self.max_length = max_length



    def __len__(self):

        return len(self.comment_text)



    def __getitem__(self, item):

        comment_text = str(self.comment_text[item])

        comment_text = " ".join(comment_text.split())



        inputs = self.tokenizer.encode_plus(

            comment_text,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)

        }

    

class BERTBaseMultiUncased(nn.Module):

    def __init__(self, bert_path):

        super(BERTBaseMultiUncased, self).__init__()

        self.bert_path = bert_path

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768 * 2, 1)



    def forward(

            self,

            ids,

            mask,

            token_type_ids

    ):

        o1, o2 = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids)

        

        apool = torch.mean(o1, 1)

        mpool, _ = torch.max(o1, 1)

        cat = torch.cat((apool, mpool), 1)



        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        return p2
df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")
tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-multilingual-uncased/", do_lower_case=True)
device = "cuda"

model = BERTBaseMultiUncased(bert_path="../input/bert-base-multilingual-uncased/").to(device)

model.load_state_dict(torch.load("../input/jigsawmulti-bertmulti/model_multi.bin"))

model.eval()
valid_dataset = JigsawDataset(

        comment_text=df.content.values,

        tokenizer=tokenizer,

        max_length=192

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=64,

    drop_last=False,

    num_workers=1,

    shuffle=False

)
from tqdm import tqdm
with torch.no_grad():

    fin_outputs = []

    for bi, d in tqdm(enumerate(valid_data_loader)):

        ids = d["ids"]

        mask = d["mask"]

        token_type_ids = d["token_type_ids"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)



        outputs = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )



        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_outputs.extend(outputs_np)
sample = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

sample.loc[:, "toxic"] = fin_outputs

sample.to_csv("submission.csv", index=False)
sample.head()