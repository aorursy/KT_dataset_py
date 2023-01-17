import torch

import torch.nn as nn

import transformers

from transformers import (

    BertPreTrainedModel,

    BertConfig,

    RobertaConfig,

    BertModel,

    RobertaModel

)
BERT_PATH = '../input/bert-base-uncased'

ROBERTA_PATH = '../input/roberta-base'
bert_config = BertConfig.from_pretrained(

    f'{BERT_PATH}/config.json'

)

bert_config.output_hidden_states = True

roberta_config = RobertaConfig.from_pretrained(

    f'{ROBERTA_PATH}/config.json',

)

roberta_config.output_hidden_states = True
class BERTEmbedder(BertPreTrainedModel):

    def __init__(self, config=bert_config, freeze_weights=False):

        super().__init__(config=bert_config)

        self.bert = BertModel.from_pretrained(

            BERT_PATH,

            config=bert_config

        )

        if freeze_weights is True:

            for param in self.bert.parameters():

                param.requires_grad = False



    def forward(self, ids, mask, token_type_ids):

        _, _, hidden_states = self.bert(

            ids,

            mask,

            token_type_ids

        )

        return hidden_states





class RoBERTaEmbedder(BertPreTrainedModel):

    def __init__(self, config=roberta_config, freeze_weights=False):

        super().__init__(config=roberta_config)

        self.roberta = RobertaModel.from_pretrained(

            ROBERTA_PATH,

            config=roberta_config

        )

        if freeze_weights is True:

            for param in self.roberta.parameters():

                param.requires_grad = False



    def forward(self, ids, mask, token_type_ids):

        _, _, hidden_states = self.roberta(

            ids,

            mask,

            token_type_ids

        )

        return hidden_states

    
class ClassifierConcatLastTwo(nn.Module):

    """ concatenate output of last two hidden layers """

    def __init__(self, hidden_size=768):

        super().__init__()

        self.drop_out = nn.Dropout(0.1)

        self.linear = nn.Linear(hidden_size * 2, 2)

        torch.nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, hidden_states):

        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)

        out = self.drop_out(out)

        logits = self.linear(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits





class ClassifierConcatLastThree(nn.Module):

    """ concatenate output of last three hidden layers """

    def __init__(self, hidden_size=768):

        super().__init__()

        self.drop_out = nn.Dropout(0.1)

        self.linear = nn.Linear(hidden_size * 3, 2)

        torch.nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, hidden_states):

        out = torch.cat(

            (hidden_states[-1], hidden_states[-2], hidden_states[-3]),

            dim=-1

        )

        out = self.drop_out(out)

        logits = self.linear(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits





class ClassifierAverageLastThree(nn.Module):

    """ average output of last three hidden layers """

    def __init__(self, hidden_size=768):

        super().__init__()

        self.drop_out = nn.Dropout(0.1)

        self.linear = nn.Linear(hidden_size, 2)

        torch.nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, hidden_states):

        out = torch.stack(

            (hidden_states[-1], hidden_states[-2], hidden_states[-3])

        )

        out = self.drop_out(out)

        logits = self.linear(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
class CombinedModel(nn.Module):

    def __init__(self, embedder, classifier):

        super().__init__()

        self.embedder = embedder

        self.classifier = classifier



    def forward(self, ids, mask, token_type_ids):

        hidden_states = self.embedder(ids, mask, token_type_ids)

        logits = self.classifier(hidden_states)

        start_logits = logits[0]

        end_logits = logits[1]

        return start_logits, end_logit
EMBEDDER_DISPATCHER = {

    'bert': BERTEmbedder(),

    'roberta': RoBERTaEmbedder()

}



CLASSIFIER_DISPATCHER = {

    'concat_last_two': ClassifierConcatLastTwo(),

    'concat_last_three': ClassifierConcatLastThree(),

    'average_last_three': ClassifierAverageLastThree()

}
embedder = EMBEDDER_DISPATCHER['bert']

classifier = CLASSIFIER_DISPATCHER['concat_last_two']

model = CombinedModel(embedder, classifier)

print(model)
embedder = EMBEDDER_DISPATCHER['roberta']

classifier = CLASSIFIER_DISPATCHER['concat_last_two']

model = CombinedModel(embedder, classifier)

print(model)