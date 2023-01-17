# To perform boosting, firstly, we need to import these trained models

# Currently, we have some fine-tuned RoBERTa models and a fine-tuned BERT model



# Import neccessary modules first

!pip install -q transformers



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path



import os

from tqdm.notebook import tqdm



import torch

import torch.optim as optim



import random 



# fastai

from fastai import *

from fastai.text import *

from fastai.callbacks import *



# transformers

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig



from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig

from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import fastai

import transformers

print('fastai version :', fastai.__version__)

print('transformers version :', transformers.__version__)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import the training and testing data, then, cleaning

base_dir =  '/kaggle/input/tridata'

train = pd.read_csv(os.path.join(base_dir, 'train.csv'))

test = pd.read_csv(os.path.join(base_dir, 'test.csv'))

print(train.shape,test.shape)



# Eliminate noisy reviews

noisy_row = [31, 50, 2235, 5244, 10409, 11748, 12384, 14395, 15215, 17629, 20819, 23691, 32089, 39532, 40530, 43954, 48186, 50500, 55834, 60088,

             60442, 61095, 62982, 63803, 67464, 70791, 74861, 73636, 74119, 76275, 79789, 85745, 91058, 91663, 91800, 93204, 99295, 100903, 101177, 103155,

             109166, 109566, 109651, 109724, 110115, 110441, 111461, 113175, 115782, 116903, 118099, 118328, 118414, 119071, 125338, 125340, 129496, 129640, 

             132027, 138212, 131626, 134715, 133248, 136217, 141377, 143707, 145045, 146485, 37301]



train.drop(noisy_row, inplace = True)

train['review_id'] = list(range(train.shape[0]))

train.index = list(range(train.shape[0]))



# Dealing with imbalanced data, choose under because the oversampling may amplify the noise in the training data

# ... and indeed, it did...

balance = 'under'

if balance == 'over':

    from imblearn.over_sampling import RandomOverSampler

    oversample = RandomOverSampler(sampling_strategy = 'all')

    X = train.review.values

    y = train.rating.values - 1

    X, y = oversample.fit_resample(X.reshape(-1,1), y)

elif balance == 'under':

    from imblearn.under_sampling import RandomUnderSampler

    undersample = RandomUnderSampler(sampling_strategy = 'all')

    X = train.review.values

    y = train.rating.values - 1

    X, y = undersample.fit_resample(X.reshape(-1,1), y)

else:

    X = train.review.values

    y = train.rating.values - 1

    

X = X.flatten()

df = {

    'review_id': list(range(len(y))), 

    'review': X, 

    'rating': y

}

train = pd.DataFrame(data = df)



'''# Augmentation with Amazon dataset

train_amazon = pd.read_csv(os.path.join(base_dir, 'amazon_augmented.csv'), index_col = 0)

train = pd.concat((train, train_amazon))

train.review_id = list(range(train.shape[0]))

train.index = list(range(train.shape[0]))

del train_amazon'''



# Handling emoji

import emoji  # https://pypi.org/project/emoji/



have_emoji_train_idx = []

have_emoji_test_idx = []



for idx, review in enumerate(train['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_train_idx.append(idx)

        

for idx, review in enumerate(test['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_test_idx.append(idx)

        

def emoji_cleaning(text):

    

    # Change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # Delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    return text



train_df_original = train.copy()

test_df_original = test.copy()



# emoji_cleaning

train.loc[have_emoji_train_idx, 'review'] = train.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)

test.loc[have_emoji_test_idx, 'review'] = test.loc[have_emoji_test_idx, 'review'].apply(emoji_cleaning)



# Cleaning reviews

def review_cleaning(text):

    

    # delete lowercase and newline

    text = text.lower()

    text = re.sub(r'\n', '', text)

    

    # change emoticon to text

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

    # delete punctuation

    text = re.sub('[^a-z0-9 ]', ' ', text)

    

    return text



train['review'] = train['review'].apply(review_cleaning)

test['review'] = test['review'].apply(review_cleaning)



# Remove repeated characters in rows

repeated_rows_train = []

repeated_rows_test = []



for idx, review in enumerate(train['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_train.append(idx)

        

for idx, review in enumerate(test['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_test.append(idx)

        

def delete_repeated_char(text):    

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    return text



train.loc[repeated_rows_train, 'review'] = train.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)

test.loc[repeated_rows_test, 'review'] = test.loc[repeated_rows_test, 'review'].apply(delete_repeated_char)



# Contractions

def recover_shortened_words(text):

    

    # put \b (boundary) for avoid the characters in the word to be replaced

    # I only make a few examples here, you can add if you're interested :)

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    

    text = re.sub(r'\bgk\b', 'gak', text)

    

    text = re.sub(r'\bhrs\b', 'harus', text)

    

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    

    text = re.sub(r'\borg\b', 'orang', text)

    

    text = re.sub(r'\bpjg\b', 'panjang', text)

    

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text



train['review'] = train['review'].apply(recover_shortened_words)

test['review'] = test['review'].apply(recover_shortened_words)



train
train.to_csv('train_clean.csv')

test.to_csv('test_clean.csv')



# Load

path = '/kaggle/working'

train = pd.read_csv(os.path.join(path, 'train_clean.csv'), index_col = 0)

test = pd.read_csv(os.path.join(path, 'test_clean.csv'), index_col = 0)



train = train.dropna()
# For RoBERTa, define the model

# Parameters

seed = 42

use_fp16 = False

bs = 16



model_type = 'roberta'

pretrained_model_name = 'roberta-base'



model_class, tokenizer_class, config_class = RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig



# Set seed

def seed_all(seed_value):

    random.seed(seed_value) # Python

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

        

seed_all(seed)



# Define transformer processors

class TransformersBaseTokenizer(BaseTokenizer):

    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):

        self._pretrained_tokenizer = pretrained_tokenizer

        self.max_seq_len = pretrained_tokenizer.max_len

        self.model_type = model_type



    def __call__(self, *args, **kwargs): 

        return self



    def tokenizer(self, t:str) -> List[str]:

        """Limits the maximum sequence length and add the spesial tokens"""

        CLS = self._pretrained_tokenizer.cls_token

        SEP = self._pretrained_tokenizer.sep_token

        if self.model_type in ['roberta']:

            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]

            tokens = [CLS] + tokens + [SEP]

        else:

            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

            if self.model_type in ['xlnet']:

                tokens = tokens + [SEP] +  [CLS]

            else:

                tokens = [CLS] + tokens + [SEP]

        return tokens

    

transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)

fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules = [], post_rules = [])



class TransformersVocab(Vocab):

    def __init__(self, tokenizer: PreTrainedTokenizer):

        super(TransformersVocab, self).__init__(itos = [])

        self.tokenizer = tokenizer

    

    def numericalize(self, t:Collection[str]) -> List[int]:

        "Convert a list of tokens `t` to their ids."

        return self.tokenizer.convert_tokens_to_ids(t)

        #return self.tokenizer.encode(t)



    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

    

    def __getstate__(self):

        return {'itos':self.itos, 'tokenizer':self.tokenizer}



    def __setstate__(self, state:dict):

        self.itos = state['itos']

        self.tokenizer = state['tokenizer']

        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

        

transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)

numericalize_processor = NumericalizeProcessor(vocab = transformer_vocab)



tokenize_processor = TokenizeProcessor(tokenizer = fastai_tokenizer, include_bos = False, include_eos = False)



transformer_processor = [tokenize_processor, numericalize_processor]



# Padding strategy

pad_first = False

pad_idx = transformer_tokenizer.pad_token_id



# Generate Data Bunch

databunch = (TextList.from_df(train, cols = ['review'], processor = transformer_processor)

             .split_none()

             .label_from_df(cols = ['rating'])

             .add_test(test)

             .databunch(bs = bs, pad_first = pad_first, pad_idx = pad_idx))
# Define the RoBERTa model

# defining our model architecture 

class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):

        super(CustomTransformerModel,self).__init__()

        self.transformer = transformer_model

        

    def forward(self, input_ids, attention_mask = None):

        

        # attention_mask

        # Mask to avoid performing attention on padding token indices.

        # Mask values selected in ``[0, 1]``:

        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        attention_mask = (input_ids != pad_idx).type(input_ids.type()) 

        

        logits = self.transformer(input_ids,

                                  attention_mask = attention_mask)[0]   

        return logits

    

# Set configuration for the model

config = config_class.from_pretrained(pretrained_model_name)

config.num_labels = 5

config.use_bfloat16 = use_fp16



transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)



from fastai.callbacks import *

from transformers import AdamW

from functools import partial



CustomAdamW = partial(AdamW, correct_bias = False)



learner = Learner(databunch, 

                  custom_transformer_model, 

                  opt_func = CustomAdamW, 

                  metrics = [accuracy, error_rate])



# Show graph of learner stats and metrics after each epoch.

learner.callbacks.append(ShowGraph(learner))



# Put learn in FP16 precision mode. --> Seems to not working

if use_fp16: learner = learner.to_fp16()



# For roberta-base

list_layers = [learner.model.transformer.roberta.embeddings,

              learner.model.transformer.roberta.encoder.layer[0],

              learner.model.transformer.roberta.encoder.layer[1],

              learner.model.transformer.roberta.encoder.layer[2],

              learner.model.transformer.roberta.encoder.layer[3],

              learner.model.transformer.roberta.encoder.layer[4],

              learner.model.transformer.roberta.encoder.layer[5],

              learner.model.transformer.roberta.encoder.layer[6],

              learner.model.transformer.roberta.encoder.layer[7],

              learner.model.transformer.roberta.encoder.layer[8],

              learner.model.transformer.roberta.encoder.layer[9],

              learner.model.transformer.roberta.encoder.layer[10],

              learner.model.transformer.roberta.encoder.layer[11],

              learner.model.transformer.roberta.pooler]



learner.split(list_layers)
# Loading models and boosting

def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]
class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(Feedforward, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.output_size  = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)

        self.relu = torch.nn.ReLU()

        self.fc2  = torch.nn.Linear(self.hidden_size, self.output_size)

        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x):

        output = self.fc1(x)

        output = self.relu(output)

        output = self.fc2(output)

        output = self.softmax(output)

        return output
class Boosting(object):

    def __init__(self, num_models = 10, num_class = 5):

        self.num_models = num_models

        self.num_class = num_class

        

    def predict(self, train, test, learner, model_path):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        N = train.shape[0]

        

        # Boosting

        print('Start stacking inputs...')

        stack = None

        stack_test = None

        for i in tqdm(range(self.num_models)):

            # Load the model

            learner.load(os.path.join(model_path, 'model_' + str(i)))



            # Predict in-sample (remember to use DatasetType.Fix rather than DatasetType.Train)

            train_preds = get_preds_as_nparray(DatasetType.Fix)

            test_preds = get_preds_as_nparray(DatasetType.Test)

            if (stack is None) & (stack_test is None) :

                stack = train_preds

                stack_test = test_preds

            else:

                stack = np.dstack((stack, train_preds))

                stack_test = np.dstack((stack_test, test_preds))

            

        # Flatten predictions to [rows, members x probabilities]

        del learner

        stack = stack.reshape((stack.shape[0], stack.shape[1] * stack.shape[2]))

        stack = torch.from_numpy(stack).float().to(device)

        

        stack_test = stack_test.reshape((stack_test.shape[0], stack_test.shape[1] * stack_test.shape[2]))

        stack_test = torch.from_numpy(stack_test).float().to(device)

        

        

        # Fitting

        print('Stacking: Done! \n Start fitting...')

        

        # Define the model and send it to GPU

        model = Feedforward(stack.shape[1], 128, 5)

        model.cuda()

        y_true = torch.from_numpy(train.rating.values).to(device)

        

        # Define the loss function and optimizer

        loss = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr = 1e-2)    

        # Training



        model.train()

        epoch = 5000

        for e in range(epoch):

            optimizer.zero_grad()

            # Forward pass

            y_pred = model(stack)

            # Compute Loss

            L = loss(y_pred, y_true)

            

            if (e % 500) == 499:

                print('Epoch {}: train loss: {}'.format(e, L.item()))

            

            # Backward pass

            L.backward()

            optimizer.step()

            

        # Predict out-of-sample

        print('Fitting: Done! \n Start predicting...')

        model.eval()

        test_rating = model(stack_test).detach().cpu().numpy()

                

        test['rating'] = np.argmax(test_rating, axis = 1) + 1

        test = test.drop(['review'], axis = 1)

        print('Predicting: Done!')

        return test
boost = Boosting(num_models = 10, num_class = config.num_labels)

# Store the result

boost.predict(train, test, learner, base_dir).to_csv('predictions_boosting_6.csv', index = False)