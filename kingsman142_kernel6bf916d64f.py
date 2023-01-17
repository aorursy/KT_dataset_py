import os

import sys

import json

import random

import pandas as pd

import numpy as np



# useful constants

TEST_PATH = os.path.join("../input/google-quest-challenge/test.csv")

USEFUL_COLUMN_NAMES = ["question_body", "answer"]



def preprocess_text(text):

    # TODO: maybe convert all text to lowercase? see if that affects accuracy?

    text = text.replace(".", " [EOS] ")

    text = text.replace("?", " [EOQ] ")

    text = text.replace("(", "")

    text = text.replace(")", "")

    text = text.split(" ")

    return text



# load data

print("Loading data...")

test_data = pd.read_csv(TEST_PATH)

with open("../input/mydata/word2idx.json", "r") as word2idx_file:

    word2idx = json.loads(word2idx_file.read())

    vocab_size = len(word2idx)



# extract only the useful column names for text preprocessing

test_data = test_data[USEFUL_COLUMN_NAMES]

test_questions = []

test_answers = []



# preprocess question and answer text

print("Preprocessing testing text...")

for index, row in test_data.iterrows(): # iterate over testing data

    # read in question and answer on this row

    question, answer = row



    # convert from "This is my question?" to ["This", "is", "my", "question", "[EOQ]"]

    question_tokenized = preprocess_text(question)

    answer_tokenized = preprocess_text(answer)



    # prepare some variables so we can convert ["This", "is", "my", "question", "[EOQ]"] to [23, 486, 3, 54, 128]

    question_numbered = []

    answer_numbered = []



    # convert ["This", "is", "my", "question", "[EOQ]"] to [23, 486, 3, 54, 128]

    for index, word in enumerate(question_tokenized):

        if index < 8172:

            question_numbered.append(word2idx[word] if word in word2idx else word2idx["[NULL]"])

    for index, word in enumerate(answer_tokenized):

        if index < 8172:

            answer_numbered.append(word2idx[word] if word in word2idx else word2idx["[NULL]"])

    question_numbered += [0] * (8172 - len(question_numbered)) # choose 8172 as the max length of a sentence so we can pad the rest of the sentence with delimiters

    answer_numbered += [0] * (8172 - len(answer_numbered))



    test_questions.append(question_numbered)

    test_answers.append(answer_numbered)



# concatenate question/answer text and scores

print("Concatenating three columns together...")

new_test_data = []

for index, row in enumerate(test_questions):

    new_row = [test_questions[index], test_answers[index]]

    new_test_data.append(new_row)



# turn training and test data into DataFrames

new_test_data = pd.DataFrame(new_test_data, columns = ["question", "answer"])

print(new_test_data.shape)



# save new data

print("Saving data...")

new_test_data.to_csv(os.path.join("../test_preprocessed.csv"))

print("\nVocabulary size: {}".format(len(word2idx)))

import torch

import torch.nn as nn



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class QUESTModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, lstm_layers, num_question_output, num_answer_output):

        super(QUESTModel, self).__init__()

        self.hidden_size = hidden_size

        self.embedding_dim = embedding_dim



        self.question_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.embedding_dim)

        self.question_lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = lstm_layers, dropout = 0.0, bidirectional = False)

        self.question_linear = nn.Linear(in_features = hidden_size, out_features = num_question_output)



        self.answer_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.embedding_dim)

        self.answer_lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = lstm_layers, dropout = 0.0, bidirectional = False)

        self.answer_linear = nn.Linear(in_features = hidden_size, out_features = num_answer_output)



    def forward(self, question, answer):

        batch_size = len(question)



        # NOTE: pytorch LSTM units take input in the form of [window_length, batch_size, num_features], which will end up being [WINDOW_SIZE, batch_size, 1] for our dataset

        # reshape question and answer sizes

        #print(question.shape, answer.shape, batch_size)

        question = question.permute(1, 0)

        answer = answer.permute(1, 0)



        question_hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),

                                torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))

        answer_hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(DEVICE),

                              torch.zeros(1, batch_size, self.hidden_size).to(DEVICE))



        question_embed = self.question_embeddings(question)

        #print(question_embed.shape)

        question_lstm_out, _ = self.question_lstm(question_embed, question_hidden_cell)

        question_pred_scores = self.question_linear(question_lstm_out[-1])



        answer_embed = self.answer_embeddings(answer)

        #print(answer_embed.shape)

        answer_lstm_out, _ = self.answer_lstm(answer_embed, answer_hidden_cell)

        answer_pred_scores = self.answer_linear(answer_lstm_out[-1])



        pred_scores = torch.cat((question_pred_scores, answer_pred_scores), dim = 1)

        pred_scores = torch.sigmoid(pred_scores)



        return pred_scores

import json

import torch

import torch.nn as nn

import torch.utils.data as data

import pandas as pd

import ast



class QUESTDataset(data.Dataset):

    def __init__(self, data_path):

        super(QUESTDataset, self).__init__()



        self.data = pd.read_csv(data_path)

        self.train = True if "scores" in self.data.columns else False



    def __getitem__(self, index):

        row = self.data.iloc[index]

        question = torch.as_tensor(ast.literal_eval(row["question"]))

        answer = torch.as_tensor(ast.literal_eval(row["answer"]))

        if self.train:

            scores = torch.as_tensor(ast.literal_eval(row["scores"]))

            return {'question': question, 'answer': answer, 'scores': scores}

        else:

            return {'question': question, 'answer': answer}



    def __len__(self):

        return len(self.data)

import os

import json

import glob

import torch

import torch.nn as nn

import torch.optim as optim

import pandas as pd



HIDDEN_SIZE = 100

LSTM_LAYERS = 1

EMBEDDING_DIM = 50

WORD2IDX_PATH = os.path.join("../input/mydata/word2idx.json")



QUESTION_OUTPUT_SIZE = 21 # 21 features/attributes corresponding to the questions

ANSWER_OUTPUT_SIZE = 9 # 9 features/attributes corresponding to the answers

COLUMN_NAMES = ['question_asker_intent_understanding', 'question_body_critical',

       'question_conversational', 'question_expect_short_answer',

       'question_fact_seeking', 'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']



saved_model_names = glob.glob("../input/models/*")

MODEL_SAVE_DIR = max(saved_model_names, key = os.path.getctime) # get the model that was saved most recently



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# load in word2idx and determine vocab size

with open(WORD2IDX_PATH, "r") as word2idx_file:

    word2idx = json.loads(word2idx_file.read())

    vocab_size = len(word2idx)



# init dataset and loaders for test dataset

test_data = QUESTDataset(data_path = os.path.join("../test_preprocessed.csv"))

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)



# init training modules, such as the model, Adam optimizer and loss function

model = QUESTModel(vocab_size = vocab_size, embedding_dim = EMBEDDING_DIM, hidden_size = HIDDEN_SIZE, lstm_layers = LSTM_LAYERS, num_question_output = QUESTION_OUTPUT_SIZE, num_answer_output = ANSWER_OUTPUT_SIZE)

model.load_state_dict(torch.load(MODEL_SAVE_DIR, map_location=torch.device('cpu')))



# create a 'models' directory to save models

if not os.path.exists("../input/models"):

    print("No pretrained models exist. Please run train.py first...")





# run the model on the testing dataset

print("Testing model...")

test_preds = []

for batch_id, samples in enumerate(test_loader):

    question, answer = samples["question"].to(DEVICE), samples["answer"].to(DEVICE)

    pred_scores = model(question.to(DEVICE), answer.to(DEVICE))

    test_preds.append(pred_scores[0].cpu().detach().numpy())



    if batch_id % 50 == 0:

        print("Done with {}/{} test samples...".format(batch_id+1, len(test_loader)))



# save the predictions to a file

print("Saving predictions...")

test_preds = pd.DataFrame(test_preds, columns = COLUMN_NAMES)

test_preds.to_csv("submission.csv", index = False)
