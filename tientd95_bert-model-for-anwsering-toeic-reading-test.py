import json
with open('../input/toeic-test/toeic_test.json') as input_json:
    data = json.load(input_json)

# Data is a dictionary contain over 3000 toeic question
# Let read the first question and familiar with format
data['1']
# Convert data dict to list so that we can iterate them
question_infors = []

for key, value in data.items():
    question_infors.append(value)

question_infors[0]
!pip install -U pytorch-pretrained-bert;
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
class TOEICBert():
    """
    Model using pretrained Bert for answering toeic question, running for each example
    Bertmodel: we can choose bert large cased/bert large uncased, etc
    
    Model return the answer for the question based on the highest probability
    """
    def __init__(self, bertmodel):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.bertmodel = bertmodel
        # Initial tokenizer to tokenize the question later
        self.tokenizer = BertTokenizer.from_pretrained(self.bertmodel)
        self.model = BertForMaskedLM.from_pretrained(self.bertmodel).to(self.device)
         # We used pretrained BertForMaskedLM to fill in the blank, do not fine tuning so we set model to eval
        self.model.eval()
        
    def get_score(self,question_tensors, segment_tensors, masked_index, candidate):
        # Tokenize the answer candidate
        candidate_tokens = self.tokenizer.tokenize(candidate)
        # After tokenizing, we convert token to ids, (word to numerical)
        candidate_ids = self.tokenizer.convert_tokens_to_ids(candidate_tokens)
        predictions = self.model(question_tensors, segment_tensors)
        predictions_candidates = predictions[0,masked_index, candidate_ids].mean()
        return predictions_candidates.item()
    
    def predict(self,row):
        # Tokenizing questions, convert '___' to '_' so that we can MASK it
        question_tokens = self.tokenizer.tokenize(row['question'].replace('___', '_'))
        masked_index = question_tokens.index('_')
        # Assign [MASK] to blank that need to be completed
        question_tokens[masked_index] = '[MASK]'
        segment_ids = [0] * len(question_tokens)
        segment_tensors = torch.tensor([segment_ids]).to(self.device)
        question_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        question_tensors = torch.tensor([question_ids]).to(self.device)
        candidates = [row['1'], row['2'], row['3'], row['4']]
        # Return probabilities of answer choice [prob1, prob2, prob3, prob4]
        predict_tensor = torch.tensor([self.get_score(question_tensors, segment_tensors,
                                                masked_index, candidate) for candidate in candidates])
        # Softmax the predict probability to return the index for maximum values
        predict_idx = torch.argmax(predict_tensor).item()
        return candidates[predict_idx]
Bertmodel  = 'bert-large-uncased'
model = TOEICBert(Bertmodel)
count = 0
for question in question_infors:
    anwser_predict = model.predict(question)
    if anwser_predict == question['anwser']:
        count+=1

num_questions = len(question_infors)
print(f'The model predict {round(count/num_questions,2) * 100} % of total {len(question_infors)} questions')
def Answer_toeic(question):    
    predict_anwser = model.predict(question)
    anwser = question['anwser']
    if predict_anwser == anwser:
        print(f'The BertModel answer: {predict_anwser}')
        print('This is right answer')
    else:
        print(f'The BertModel answer: {predict_anwser}')
        print('This is wrong answer')
        
# now we have a TOEIC question on below:
question = {'1': 'different',
 '2': 'differently',
 '3': 'difference',
 '4': 'differences',
 'anwser': 'different',
 'question': 'Matos Realty has developed two ___ methods of identifying undervalued properties.'}

# Check the model
Answer_toeic(question)
