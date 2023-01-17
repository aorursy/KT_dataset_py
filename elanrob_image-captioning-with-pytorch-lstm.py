import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import torch

import torchvision

from PIL import Image
imagesFilepath = "../input/flickr8k/Flickr_Data/Flickr_Data/Images"

latent_representation_filepath = "/kaggle/working/flickr8k_latent.csv"

useCuda = torch.cuda.is_available()

# Load the googlenet pre-trained neural network

encoder = torchvision.models.googlenet(pretrained=True)

encoder.eval()

# Replace the last linear layer (called 'fc') with a dummy identity layer.

# By doing this, we can output the image latent representation

# without messing with the layers naming convention

encoder.fc = torch.nn.Identity()

number_of_latent_variables = 1024 # Can be found by examining googlenet architecture

if useCuda:

    encoder = encoder.cuda()
preprocess = torchvision.transforms.Compose([

            torchvision.transforms.Resize(256),

            torchvision.transforms.CenterCrop(224),

            torchvision.transforms.ToTensor(),

            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        
# Create a list of jpg image filepaths

found_jpg_filepaths = []

directory_contents = os.listdir(imagesFilepath)

for content in directory_contents:

    filepath = os.path.join(imagesFilepath, content)

    if filepath.endswith('.jpg'):

        found_jpg_filepaths.append(filepath)
if not os.path.isfile(latent_representation_filepath): # No need to run it more than once

    with open(latent_representation_filepath, 'w+') as output_file:

        # Write the file header

        output_file.write('filepath')

        for variableNdx in range(number_of_latent_variables):

            output_file.write(",v{}".format(variableNdx))

        output_file.write("\n")



        # Loop through the images

        for index, image_filepath in enumerate(found_jpg_filepaths):

            output_file.write(image_filepath)

            image = Image.open(image_filepath)

            inputTsr = preprocess(image).unsqueeze(0) # Preprocess and add a dummy mini-batch dimension

            if useCuda:

                inputTsr = inputTsr.cuda()

            with torch.no_grad():

                latentTsr = encoder(inputTsr)[0] # Run a forward pass through the encoder and get rid of the dummy mini-batch dimension

                for valueNdx in range(latentTsr.shape[-1]):

                    output_file.write(",{}".format(latentTsr[valueNdx].item()))

                output_file.write("\n")

            if index % 300 == 0:

                print ("{}/{}".format(index, len(found_jpg_filepaths)), end=" ", flush=True)

latent_representationDf = pd.read_csv(latent_representation_filepath)
descriptions_filepath = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

!head $descriptions_filepath

sample_filename = os.path.basename(found_jpg_filepaths[0])
sample_image_filename = '1000268201_693b08cb0e.jpg'

import IPython.display

IPython.display.Image(os.path.join(imagesFilepath, sample_image_filename))
training_images_filepath = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

validation_images_filepath = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'

!head $training_images_filepath
import en_core_web_sm

import re

import string



nlp = en_core_web_sm.load()



def Tokenize(text, nlp):

    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # Remove punctuation and numbers

    nopunct = regex.sub(" ", text.lower())

    tokens = [token.text for token in nlp.tokenizer(nopunct) if not token.text.isspace()]

    return tokens



def TrainDescriptionTokens(descriptions_filepath, train_images_list, nlp):

    number_of_train_lines = 0

    number_of_not_train_lines = 0

    train_token_to_occurrences_dict = {}

    with open(descriptions_filepath, 'r') as descriptions_file:

        for line in descriptions_file:

            line = line.strip()

            sharp_index = line.find('#')

            if sharp_index == -1:

                raise ValueError("TrainDescriptionTokens(): Could not find '#' in line '{}'".format(line))

            image_filename = line[:sharp_index]

            description = line[sharp_index + 3: ]

            if image_filename in train_images_list:

                # Tokenize the description

                tokens = Tokenize(description, nlp)

                for token in tokens:

                    if token in train_token_to_occurrences_dict:

                        train_token_to_occurrences_dict[token] += 1

                    else:

                        train_token_to_occurrences_dict[token] = 1

                number_of_train_lines += 1

            else:

                number_of_not_train_lines += 1

    print("TrainDescriptionTokens(): number_of_train_lines = {}; number_of_not_train_lines = {}".format(number_of_train_lines, number_of_not_train_lines))

    return train_token_to_occurrences_dict



with open(training_images_filepath, 'r') as train_images_file:

    train_images_list = [line.strip() for line in train_images_file]

train_token_to_occurrences_dict = TrainDescriptionTokens(descriptions_filepath,

                                                             train_images_list, nlp)
print("Before filtering the single occurrences, len(train_token_to_occurrences_dict) = {}".format(len(train_token_to_occurrences_dict)))

single_occurrence_words = []

for word, occurrences in train_token_to_occurrences_dict.items():

    if occurrences < 2:

        single_occurrence_words.append(word)

for word in single_occurrence_words:

    train_token_to_occurrences_dict.pop(word)

print("After filtering the single occurrences, len(train_token_to_occurrences_dict) = {}".format(len(train_token_to_occurrences_dict)))
vocabulary_filepath = '/kaggle/working/vocabulary.csv'

sorted_tokens = sorted(train_token_to_occurrences_dict.items(),

                           key=lambda x: x[1], reverse=True) # Cf. https://careerkarma.com/blog/python-sort-a-dictionary-by-value/

sorted_tokens = [('ENDOFSEQ', 0), ('UNKNOWN', 0), ('NOTSET', 0)] + sorted_tokens



with open(vocabulary_filepath, 'w+') as output_file:

    output_file.write("index,word,frequency\n")

    for index, token in enumerate(sorted_tokens):

        output_file.write("{},{},{}\n".format(index, token[0], token[1]))
def LoadVocabulary(vocabularyFilepath):

    word_to_index_dict = {}

    index_to_word_dict = {}

    vocabDf = pd.read_csv(vocabularyFilepath)

    for i, row in vocabDf.iterrows():

        index = row['index']

        word = row['word']

        word_to_index_dict[word] = index

        index_to_word_dict[index] = word

    return word_to_index_dict, index_to_word_dict



word_to_index_dict, index_to_word_dict = LoadVocabulary(vocabulary_filepath)

print ("word_to_index_dict['apple'] = {}".format(word_to_index_dict['apple']))



def ConvertTokensListToIndices(tokens, word_to_index_dict, maximum_length):

    indices = [word_to_index_dict['NOTSET']] * maximum_length

    for tokenNdx, token in enumerate(tokens):

        index = word_to_index_dict.get(token, word_to_index_dict['UNKNOWN']) # If the word is not in the dictionary, fall back to 'UNKOWN'

        indices[tokenNdx] = index

    if len(tokens) < maximum_length:

        indices[len(tokens)] = word_to_index_dict['ENDOFSEQ']

    return indices



pretokenized_descriptions_filepath = '/kaggle/working/tokenized_descriptions.csv'

description_maximum_length = 40

with open(pretokenized_descriptions_filepath, 'w+') as outputFile:

    # Write the header

    outputFile.write("image")

    for wordNdx in range(description_maximum_length):

        outputFile.write(",w{}".format(wordNdx))

    outputFile.write("\n")



    # Loop through the lines of the descriptions file

    with open(descriptions_filepath, 'r') as descriptionsFile:

        for line in descriptionsFile:

            line = line.strip()

            sharp_index = line.find('#')

            if sharp_index == -1:

                raise ValueError("Could not find '#' in line '{}'".format(line))

            image_filename = line[:sharp_index]

            description = line[sharp_index + 3:]

            # Tokenize the description

            tokens = Tokenize(description, nlp)



            # Convert the list of tokens to a list of indices

            indices = ConvertTokensListToIndices(tokens,

                                                 word_to_index_dict,

                                                 description_maximum_length)

            outputFile.write(image_filename)

            for indexNdx in range(len(indices)):

                outputFile.write(",{}".format(indices[indexNdx]))

            outputFile.write("\n")
tokenized_descriptionsDf = pd.read_csv(pretokenized_descriptions_filepath)

tokenized_descriptionsDf.head()
training_description_indices = []

for t in tokenized_descriptionsDf.itertuples():

    filename = t[1]

    if filename in train_images_list:

        training_description_indices.append(list(t[2:]))

print ("training_description_indices[0:5] = {}".format(training_description_indices[0:5]))
from torch.utils.data import Dataset, DataLoader

import random



class ContextToWordDataset(Dataset):

    def __init__(self,

                 training_descriptions_indices,

                 index_to_word_dict,

                 word_to_index_dict,

                 contextLength):

        self.training_descriptions_indices = training_descriptions_indices

        self.index_to_word_dict = index_to_word_dict

        self.word_to_index_dict = word_to_index_dict

        self.contextLength = contextLength



    def __len__(self):

        return len(self.training_descriptions_indices)



    def __getitem__(self, idx):

        description_indices = self.training_descriptions_indices[idx]

        # Randomly select a target word

        last_acceptable_center_index = len(description_indices) - 1

        if self.word_to_index_dict['ENDOFSEQ'] in description_indices:

            for position, index in enumerate(description_indices):

                if index == self.word_to_index_dict['ENDOFSEQ'] in description_indices:

                    last_acceptable_center_index = position        

        targetNdx = random.choice(range(last_acceptable_center_index + 1))

        # Create a Long tensor with dim (2 * context_length)

        description_indicesTsr = torch.ones((2 * self.contextLength)).long() * self.word_to_index_dict['NOTSET']



        runningNdx = targetNdx - int(self.contextLength)

        counter = 0

        while counter < 2 * self.contextLength:

            if runningNdx != targetNdx:

                if runningNdx >= 0 and runningNdx < len(description_indices):

                    description_indicesTsr[counter] = description_indices[runningNdx]

                counter += 1

            runningNdx += 1

        return (description_indicesTsr, description_indices[targetNdx])

    

train_dataset = ContextToWordDataset(training_description_indices,

                 index_to_word_dict,

                 word_to_index_dict,

                 contextLength=3)
sample_data_0 = train_dataset[0]

sample_words_0 = [index_to_word_dict[i] for i in sample_data_0[0].tolist()]

center_word_0 = index_to_word_dict[sample_data_0[1]]

print ("sample_words_0 = {}; center_word_0 = {}".format(sample_words_0, center_word_0))

sample_data_1 = train_dataset[1]

sample_words_1 = [index_to_word_dict[i] for i in sample_data_1[0].tolist()]

center_word_1 = index_to_word_dict[sample_data_1[1]]

print ("sample_words_1 = {}; center_word_1 = {}".format(sample_words_1, center_word_1))
class CenterWordPredictor(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension):

        super(CenterWordPredictor, self).__init__()

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dimension)

        self.decoderLinear = torch.nn.Linear(embedding_dimension, vocabulary_size)



    def forward(self, contextTsr):

        # contextTsr.shape = (N, context_length), contextTsr.dtype = torch.int64

        embedding = self.embedding(contextTsr)  # (N, context_length, embedding_dimension)

        # Average over context words: (N, context_length, embedding_dimension) -> (N, embedding_dimension)

        embedding = torch.mean(embedding, dim=1)



        # Decoding

        outputTsr = self.decoderLinear(embedding)

        return outputTsr

    

embedding_dimension = 128

word_embedder = CenterWordPredictor(len(word_to_index_dict), embedding_dimension)

if useCuda:

    word_embedder = word_embedder.cuda()
word_embedder_parameters = filter(lambda p: p.requires_grad, word_embedder.parameters())

optimizer = torch.optim.Adam(word_embedder_parameters, lr=0.0001)

lossFcn = torch.nn.CrossEntropyLoss()

train_dataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)



for epoch in range(1, 1000):

    word_embedder.train()

    loss_sum = 0.0

    number_of_batches = 0

    for (description_indicesTsr, target_center_word_ndx) in train_dataLoader:

        if number_of_batches % 20 == 1:

            print (".", end="", flush=True)

        if useCuda:

            description_indicesTsr = description_indicesTsr.cuda()

            target_center_word_ndx = target_center_word_ndx.cuda()

        predicted_center_word_ndx = word_embedder(description_indicesTsr)

        optimizer.zero_grad()

        loss = lossFcn(predicted_center_word_ndx, target_center_word_ndx)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        number_of_batches += 1

    train_loss = loss_sum/number_of_batches

    print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))
words2vec_dictionary_filepath = '/kaggle/working/words2vec.csv'

with open(words2vec_dictionary_filepath, 'w+') as word2vecFile:

        # Write the header

        word2vecFile.write("word")

        for embeddingNdx in range(embedding_dimension):

            word2vecFile.write(",e{}".format(embeddingNdx))

        word2vecFile.write("\n")



        for index, word in index_to_word_dict.items():

            wordEmbeddingList = word_embedder.embedding.weight[index].tolist()

            word2vecFile.write(word)

            for coef in wordEmbeddingList:

                word2vecFile.write(",{}".format(str(coef)))

            word2vecFile.write("\n")
word2vecDf = pd.read_csv(words2vec_dictionary_filepath)

word_to_embedding_dict = {word2vecDf.iloc[i]['word'] : word2vecDf.loc[i, 'e0': 'e{}'.format(embedding_dimension - 1)].tolist()

                              for i in range(len(word2vecDf))}
print ("word_to_embedding_dict['dog'] = {}".format(word_to_embedding_dict['dog']))

print()

print ("word_to_embedding_dict['dance'] = {}".format(word_to_embedding_dict['dance']))
sample_image_filename = '99679241_adc853a5c0.jpg'

IPython.display.Image(os.path.join(imagesFilepath, sample_image_filename))
class CaptionGenerationDataset(Dataset):

    def __init__(self,

                 image_filenames,

                 image_filename_to_latent_variables,

                 image_filename_to_tokenized_descriptions_list,

                 endOfSeqIndex,

                 notSetIndex,

                 vocabulary_size,

                 index_to_word_dict,

                 word_to_embedding_dict

                 ):

        self.image_filenames = image_filenames

        self.image_filename_to_latent_variables = image_filename_to_latent_variables

        self.image_filename_to_tokenized_descriptions_list = image_filename_to_tokenized_descriptions_list

        self.endOfSeqIndex = endOfSeqIndex

        self.notSetIndex = notSetIndex

        self.vocabulary_size = vocabulary_size

        self.index_to_word_dict = index_to_word_dict

        self.word_to_embedding_dict = word_to_embedding_dict



    def __len__(self):

        return len(self.image_filenames)



    def __getitem__(self, idx):

        if idx >= len(self.image_filenames):

            raise IndexError("CaptionGenerationDataset.__getitem__(): Index {} is greater than the number of images ({})".format(idx, len(self.image_filenames)))

        filename = self.image_filenames[idx]

        latent_variables = self.image_filename_to_latent_variables[filename]

        # Build a tensor

        latent_variablesTsr = torch.zeros(len(latent_variables))

        for i in range(len(latent_variables)):

            latent_variablesTsr[i] = latent_variables[i]

        # Randomly choose one of the descriptions

        description = random.choice(self.image_filename_to_tokenized_descriptions_list[filename])

        lastIndex = self.IndexOfEndOfSeq(description)

        chopIndex = random.randint(0, lastIndex) - 1 # The last index of the chopped description

        choppedDescription = [self.notSetIndex] * len(description)

        if chopIndex >= 0:

            for i in range(chopIndex + 1):

                choppedDescription[i] = description[i]



        embedding_dim = len(self.word_to_embedding_dict[ self.index_to_word_dict[0] ]) # Length of the embedding of the 1st word

        embeddedChoppedDescriptionTsr = torch.zeros((len(description), embedding_dim))

        for wordPosn in range(len(choppedDescription)):

            embedding = self.word_to_embedding_dict[ self.index_to_word_dict[choppedDescription[wordPosn]]]

            embeddedChoppedDescriptionTsr[wordPosn] = torch.tensor(embedding)



        next_word_index = description[chopIndex + 1]



        return ( (latent_variablesTsr, embeddedChoppedDescriptionTsr), next_word_index)



    def IndexOfEndOfSeq(self, description_list):

        foundIndex = -1

        for candidateNdx in range(len(description_list)):

            if description_list[candidateNdx] == self.endOfSeqIndex:

                foundIndex = candidateNdx

                break

        if foundIndex == -1:

            return len(description_list) - 1

        else:

            return foundIndex

        

def ImageFilenameToLatentVariables(latent_varDf):

    image_filename_to_latent_variables = {}

    for i, row in latent_varDf.iterrows():

        filepath = latent_varDf.iloc[i][0]

        filename = os.path.basename(filepath)

        latent_variables = list(latent_varDf.iloc[i][1:])

        image_filename_to_latent_variables[filename] = latent_variables

    return image_filename_to_latent_variables



image_filename_to_latent_variables = ImageFilenameToLatentVariables(latent_representationDf)



def ImageFilenameToTokenizedDescriptionsList(descriptionsDF):

    image_filename_to_tokenized_description = {}

    for i, row in descriptionsDF.iterrows():

        filename = descriptionsDF.iloc[i][0]

        tokenized_description = list(descriptionsDF.iloc[i][1:])

        if filename in image_filename_to_tokenized_description:

            image_filename_to_tokenized_description[filename].append(tokenized_description)

        else:

            image_filename_to_tokenized_description[filename] = [tokenized_description]

    return image_filename_to_tokenized_description



image_filename_to_tokenized_descriptions_list = ImageFilenameToTokenizedDescriptionsList(

tokenized_descriptionsDf)



lstm_train_dataset = CaptionGenerationDataset(

        train_images_list,

        image_filename_to_latent_variables,

        image_filename_to_tokenized_descriptions_list,

        word_to_index_dict['ENDOFSEQ'],

        word_to_index_dict['NOTSET'],

        len(word_to_index_dict),

        index_to_word_dict,

        word_to_embedding_dict

    )

with open(validation_images_filepath, 'r') as valid_images_file:

    validation_images_list = [line.strip() for line in valid_images_file]

lstm_validation_dataset = CaptionGenerationDataset(

        validation_images_list,

        image_filename_to_latent_variables,

        image_filename_to_tokenized_descriptions_list,

        word_to_index_dict['ENDOFSEQ'],

        word_to_index_dict['NOTSET'],

        len(word_to_index_dict),

        index_to_word_dict,

        word_to_embedding_dict

    )
class LSTM_fixed_embedding(torch.nn.Module):

    def __init__(self, embedding_dim, lstm_hidden_dim,

                 num_lstm_layers, image_latent_dim,

                 vocab_size,

                 dropoutProportion=0.5):

        super(LSTM_fixed_embedding, self).__init__()

        self.embedding_dim = embedding_dim

        self.lstm = torch.nn.LSTM(embedding_dim, lstm_hidden_dim, num_lstm_layers,

                                  batch_first=True)

        self.dropout = torch.nn.Dropout(dropoutProportion)

        self.linear = torch.nn.Linear(lstm_hidden_dim + image_latent_dim, vocab_size)





    def forward(self, image_latentTsr, embeddedChoppedDescriptionTsr):

        # image_latentTsr.shape = (N, image_latent_dim)

        # embeddedChoppedDescriptionTsr.shape = (N, sequence_length, embedding_dim)

        aggregated_h, (ht, ct) = self.lstm(embeddedChoppedDescriptionTsr)

        # ht.shape = (num_lstm_layers, N, lstm_hidden_dim)

        # ht[-1].shape = (N, lstm_hidden_dim)

        concat_latent = torch.cat( (torch.nn.functional.normalize(ht[-1]),

                                    torch.nn.functional.normalize(image_latentTsr)), dim=1)

        # concat_latent.shape = (N, image_latent_dim + lstm_hidden_dim)

        outputTsr = self.linear(self.dropout(concat_latent))

        # outputTsr.shape = (N, vocab_size)

        return outputTsr



    def Caption(self, latentVariablesTsr, maximumLength,

                         word_to_embedding_dict, index_to_word_dict,

                         endOfSeqIndex, useCuda):

        notSetEmbedding = word_to_embedding_dict['NOTSET']

        embeddedChoppedDescriptionTsr = torch.zeros((maximumLength, self.embedding_dim))

        for i in range(maximumLength):

            embeddedChoppedDescriptionTsr[i] = torch.tensor(notSetEmbedding)

        endOfSeqIsFound = False

        runningNdx = 0

        caption_words = []

        while not endOfSeqIsFound and runningNdx < maximumLength:

            if useCuda:

                latentVariablesTsr = latentVariablesTsr.cuda()

                embeddedChoppedDescriptionTsr = embeddedChoppedDescriptionTsr.cuda()

            outputTsr = self.forward(latentVariablesTsr.unsqueeze(0), embeddedChoppedDescriptionTsr.unsqueeze(0))

            next_word_index = torch.argmax(outputTsr[0]).item()

            caption_words.append(index_to_word_dict[next_word_index])

            next_word = index_to_word_dict[next_word_index]

            embeddedChoppedDescriptionTsr[runningNdx] = torch.tensor(word_to_embedding_dict[next_word])

            runningNdx += 1

            if next_word_index == endOfSeqIndex:

                endOfSeqIsFound = True

        return caption_words



lstm_hidden_dimension = 32

lstm_number_of_layers = 2

dropoutProportion = 0.5

lstm_model = LSTM_fixed_embedding(

        embedding_dim=embedding_dimension,

        lstm_hidden_dim=lstm_hidden_dimension,

        num_lstm_layers=lstm_number_of_layers,

        image_latent_dim=number_of_latent_variables,

        vocab_size=len(word_to_index_dict),

        dropoutProportion=dropoutProportion

    )



if useCuda:

    lstm_model = lstm_model.cuda()
def TestSample(index, validation_images_list, image_filename_to_latent_variables, model,

               index_to_word_dict,

               word_to_embedding_dict,

               sequence_length=40,

               endOfSeqIndex=0,

               useCuda=True):

    sample_filename = validation_images_list[index]

    sample_latentVariablesTsr = torch.FloatTensor(image_filename_to_latent_variables[sample_filename])

    if useCuda:

        sample_latentVariablesTsr = sample_latentVariablesTsr.cuda()

    sample_words = model.Caption(

        latentVariablesTsr=sample_latentVariablesTsr,

        maximumLength=sequence_length,

        word_to_embedding_dict=word_to_embedding_dict,

        index_to_word_dict=index_to_word_dict,

        endOfSeqIndex=endOfSeqIndex,

        useCuda=useCuda

    )

    return sample_words



validation_sample_0_Ndx = 0

validation_sample_100_Ndx = 100

IPython.display.Image(os.path.join(imagesFilepath, validation_images_list[validation_sample_0_Ndx]))
IPython.display.Image(os.path.join(imagesFilepath, validation_images_list[validation_sample_100_Ndx]))
import sys



parameters = filter(lambda p: p.requires_grad, lstm_model.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.0003)

lossFcn = torch.nn.CrossEntropyLoss()

train_dataLoader = DataLoader(lstm_train_dataset, batch_size=16, shuffle=True)

validation_dataLoader = DataLoader(lstm_validation_dataset, batch_size=lstm_validation_dataset.__len__())

best_model_filepath = '/kaggle/working/lstm.pth'



lowestValidationLoss = sys.float_info.max

for epoch in range(1, 500 + 1):

    lstm_model.train()

    loss_sum = 0.0

    numberOfBatches = 0

    for ( (latent_variablesTsr, chopped_descriptionTsr), target_next_word) in train_dataLoader:

        if numberOfBatches % 4 == 1:

            print (".", end="", flush=True)

        if useCuda:

            latent_variablesTsr = latent_variablesTsr.cuda()

            chopped_descriptionTsr = chopped_descriptionTsr.cuda()

            target_next_word = target_next_word.cuda()

        predicted_next_word = lstm_model(latent_variablesTsr, chopped_descriptionTsr)

        optimizer.zero_grad()

        loss = lossFcn(predicted_next_word, target_next_word)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        numberOfBatches += 1

    train_loss = loss_sum/numberOfBatches

    print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))



    # Validation

    lstm_model.eval()

    sample_0_words = TestSample(validation_sample_0_Ndx, validation_images_list, image_filename_to_latent_variables, lstm_model,

       index_to_word_dict,

       word_to_embedding_dict,

       sequence_length=40,

       endOfSeqIndex=0,

       useCuda=useCuda)

    print ("sample_0_words = {}".format(sample_0_words))

    sample_100_words = TestSample(validation_sample_100_Ndx, validation_images_list, image_filename_to_latent_variables, lstm_model,

       index_to_word_dict,

       word_to_embedding_dict,

       sequence_length=40,

       endOfSeqIndex=0,

       useCuda=useCuda)

    print ("sample_100_words = {}".format(sample_100_words))

    

    for ((validation_latent_variablesTsr, validation_chopped_descriptionTsr), validation_target_next_word) in validation_dataLoader:

        if useCuda:

            validation_latent_variablesTsr = validation_latent_variablesTsr.cuda()

            validation_chopped_descriptionTsr = validation_chopped_descriptionTsr.cuda()

            validation_target_next_word = validation_target_next_word.cuda()

        validation_predicted_next_word = lstm_model(validation_latent_variablesTsr, validation_chopped_descriptionTsr)

        validation_loss = lossFcn(validation_predicted_next_word, validation_target_next_word).item()

    print ("validation_loss = {}".format(validation_loss))



    if validation_loss < lowestValidationLoss:

        lowestValidationLoss = validation_loss

        torch.save(lstm_model.state_dict(), best_model_filepath)
test_images_filepath = '../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

with open(test_images_filepath, 'r') as test_images_file:

    test_images_list = [line.strip() for line in test_images_file]

print ("len(test_images_list) = {}".format(len(test_images_list)))
# Load the model that gave the lowest validation loss

lstm_model.load_state_dict(torch.load(best_model_filepath))

# Randomly select some test images

test_image_sample_indices = random.choices(range(len(test_images_list)), k=3)

lstm_model.eval()

test_sample_filepaths = []

test_sample_captions = []

for test_image_sample_index in test_image_sample_indices:

    sample_words = TestSample(test_image_sample_index, test_images_list, image_filename_to_latent_variables, lstm_model,

       index_to_word_dict,

       word_to_embedding_dict,

       sequence_length=40,

       endOfSeqIndex=0,

       useCuda=useCuda) 

    caption = ' '.join(sample_words)

    test_sample_filepaths.append(os.path.join(imagesFilepath, test_images_list[test_image_sample_index]))

    test_sample_captions.append(caption)
print (test_sample_captions[0])

IPython.display.Image(test_sample_filepaths[0])
print (test_sample_captions[1])

IPython.display.Image(test_sample_filepaths[1])
print (test_sample_captions[2])

IPython.display.Image(test_sample_filepaths[2])